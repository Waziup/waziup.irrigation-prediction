#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phenology_engine.py — Growing Degree Days, crop growth stage tracking,
                      dynamic Kc computation, and dynamic irrigation thresholds.

Satellite fusion strategy:
    NDVI  — best for sparse canopy (pre-emergence, development, senescence)
    NDRE  — best for dense canopy (mid-season); does not saturate like NDVI
    SAR   — cloud-cover backup; confirms canopy presence when optical is stale
    GDD   — biological plausibility check; caps satellite weight when the
            observed Kc implies a stage the thermal budget cannot support

"""

import argparse
import logging
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from crops import (
    CropParams,
    CROP_PARAMS,
    get_crop_params,
    STAGE_PRE_EMERGENCE,
    STAGE_DEVELOPMENT,
    STAGE_MID_SEASON,
    STAGE_LATE_SEASON,
    STAGE_POST_MATURITY,
)

log = logging.getLogger(__name__)

# ALGORITHM CONSTANTS

# Kc from NDVI (Glenn et al. 2011 / Kamble et al. 2013)
# Calibrated across irrigated crop studies; linear fit between NDVI and Kc.
NDVI_KC_SLOPE = 1.457   # Kc_NDVI = NDVI_KC_SLOPE * NDVI + NDVI_KC_INTERCEPT
NDVI_KC_INTERCEPT = -0.10
KC_NDVI_UPPER_CAP = 1.35    # global cap before crop-specific ceiling is applied

# Kc from NDRE (Delegido et al. 2013, adapted)
# Slope scaled so NDRE 0.1–0.6 maps to the same Kc range as NDVI 0–1.
NDRE_KC_SLOPE = 2.5     # Kc_NDRE = NDRE_KC_SLOPE * NDRE + NDRE_KC_INTERCEPT
NDRE_KC_INTERCEPT = -0.05
KC_NDRE_UPPER_CAP = 1.35    # same cap as NDVI

# Crop-specific ceiling: max(Kc_satellite) = kc_mid * KC_UPPER_TOLERANCE
# 10% tolerance above Kc_mid accommodates field variability and FAO-56 §7.4
# high-wind / low-humidity upward adjustments.
KC_UPPER_TOLERANCE = 1.10

# Quality weight for satellite blending
# w_base = clip(ndvi * NDVI_WEIGHT_SLOPE, 0, NDVI_WEIGHT_CAP)
# slope=0.85 → weight saturates at NDVI ≈ 0.88 (healthy canopy range)
# cap=0.75   → GDD always contributes ≥ 25% (satellite alone insufficient)
NDVI_WEIGHT_SLOPE = 0.85
NDVI_WEIGHT_CAP = 0.75

# NDRE-to-NDVI proxy scale factor when NDVI is absent at mid-season.
# Empirical: NDRE 0.1–0.6 ≈ NDVI 0.15–0.90 for annual field crops.
NDRE_NDVI_PROXY = 1.5

# Freshness decay thresholds (hours since last observation)
SAT_FRESH_HOURS = 360     # ≤ 15 days  → quality_factor = 1.0
SAT_REDUCED_HOURS = 720     # 15–30 days → quality_factor = 0.7
SAT_LOW_HOURS = 1080    # 30–45 days → quality_factor = 0.4
SAT_REDUCED_FACTOR = 0.7
SAT_LOW_FACTOR = 0.4
# > 45 days -> quality_factor = 0.0 by default.
# Mid-season applies a floor (see MIDSEASON_FRESHNESS_FLOOR).

# Mid-season canopy changes slowly; avoid over-penalizing stale but still
# informative imagery during cloud-cover windows.
MIDSEASON_FRESHNESS_FLOOR = 0.6

# SAR cloud-cover correction
# VV backscatter above this threshold confirms an established crop canopy.
# -10 dB separates sparse vegetation from established crop (Sentinel-1 C-band).
SAR_CANOPY_THRESHOLD_DB = -10.0
# When SAR confirms canopy, cap effective NDVI age at this value so cloud
# gaps don't decay satellite weight to zero while the canopy is unchanged.
SAR_MAX_TRUST_HOURS = 720.0   # 30 days

# GDD plausibility weight thresholds
# Fraction of crop's Kc range used to infer the satellite-implied stage.
# 0.65 × range separates Development from Mid-season signals.
# 0.10 × range separates Pre-emergence from Development signals.
# Proportional to each crop's Kc range so narrow-range crops (olive) behave
# correctly — a fixed +0.10 cbar absolute offset would invert for olive.
PLAUSIBILITY_MID_FRAC = 0.65
PLAUSIBILITY_DEV_FRAC = 0.10

# Mid-season stress safeguard for plausibility gating:
# when GDD says mid-season but the satellite Kc is very low, this can be
# genuine drought stress rather than bad imagery. Keep a partial satellite
# influence instead of hard-zeroing via stage plausibility.
# Margin is a fraction of the Kc range (kc_mid - kc_ini).
MIDSEASON_STRESS_KC_MARGIN = 0.05
MIDSEASON_STRESS_PLAUSIBILITY = 0.7


def compute_daily_gdd(
    tmax: float,
    tmin: float,
    t_base: float,
    t_ceiling: float,
) -> float:
    """
    Compute Growing Degree Days for a single day.

    Uses the modified rectangle method (McMaster & Wilhelm 1997):
        GDD = max(0, (min(Tmax, T_ceiling) + min(Tmin, T_ceiling)) / 2 - T_base)

    Tmax and Tmin are clipped to T_ceiling INDIVIDUALLY before averaging.
    Temperatures above T_ceiling denature developmental enzymes — this
    occurs regardless of nighttime recovery, so per-value clipping is
    physiologically correct.

    Args:
        tmax:      Daily maximum temperature (°C)
        tmin:      Daily minimum temperature (°C)
        t_base:    Base temperature below which no development occurs (°C)
        t_ceiling: Upper cutoff — temperatures above this are clipped (°C)

    Returns:
        GDD for the day (>= 0). Returns 0.0 for NaN inputs.
    """
    if np.isnan(tmax) or np.isnan(tmin):
        return 0.0

    t_eff_max = min(tmax, t_ceiling)
    t_eff_min = min(tmin, t_ceiling)
    t_mean = (t_eff_max + t_eff_min) / 2.0
    return max(0.0, t_mean - t_base)


def compute_gdd_series(
    tmax_series: pd.Series,
    tmin_series: pd.Series,
    planting_date: Union[str, datetime, pd.Timestamp],
    crop_type: str,
    initial_gdd: float = 0.0,
) -> pd.Series:
    """
    Compute cumulative GDD from planting_date forward.

    Production use only — one crop, one known planting date from farms.yaml.
    There is no multi-season reset logic and no estimation from data boundaries.
    Farmers in the target regions (Africa, irrigation-based) do not follow
    temperate annual planting calendars. The planting date is an explicit
    management decision recorded in the farm configuration.

    Rows before planting_date have GDD = 0.0 (sensor may have been running
    before the crop was planted).

    If planting_date falls before the data start, initial_gdd allows for
    heat units accumulated between planting and the first sensor reading.
    This is the only case where initial_gdd is not zero.

    Gap handling:
        Short temperature gaps (<= 7 days) are linearly interpolated before
        GDD computation. Sensor downtime appears as NaN in daily Tmax/Tmin;
        without interpolation those days contribute 0 GDD, systematically
        under-counting heat units. Longer gaps (> 7 days) remain as NaN
        and log a warning.

    Args:
        tmax_series:   Daily maximum temperature series (DatetimeIndex, °C).
        tmin_series:   Daily minimum temperature series (matching index).
        planting_date: Actual planting date from farms.yaml. Must be
                       provided explicitly — never inferred from data.
        crop_type:     Key into CROP_PARAMS.
        initial_gdd:   GDD already accumulated between planting_date and
                       the first row of tmax_series. Use when the sensor
                       started mid-season. Defaults to 0.0.

    Returns:
        pd.Series of cumulative GDD with same index as input.
        Rows before planting_date are 0.0.
    """
    params = get_crop_params(crop_type)
    planting_ts = pd.Timestamp(planting_date)

    # Align tz-awareness — mismatched aware/naive timestamps raise TypeError.
    idx_tz = tmax_series.index.tz
    if idx_tz is not None:
        if planting_ts.tz is None:
            planting_ts = planting_ts.tz_localize(idx_tz)
        elif planting_ts.tz != idx_tz:
            planting_ts = planting_ts.tz_convert(idx_tz)
    elif planting_ts.tz is not None:
        planting_ts = planting_ts.tz_convert("UTC").tz_localize(None)

    # Interpolate short temperature gaps.
    n_gaps_before = int(tmax_series.isna().sum())
    if n_gaps_before > 0:
        tmax_series = tmax_series.interpolate(method="linear", limit=7)
        tmin_series = tmin_series.interpolate(method="linear", limit=7)
        n_gaps_after = int(tmax_series.isna().sum())
        n_filled = n_gaps_before - n_gaps_after
        if n_filled > 0:
            log.info(
                "  Interpolated %s temperature gap days (<= 7-day gaps)",
                n_filled,
            )
        if n_gaps_after > 0:
            log.warning(
                "  %s temperature gap days remain (> 7-day gaps) -- GDD=0 for those days",
                n_gaps_after,
            )

    data_start = tmax_series.index.min()
    if planting_ts > tmax_series.index.max():
        log.warning(
            "  planting_date %s is after end of data — GDD will be all zeros. Check farms.yaml.",
            planting_ts.date(),
        )
        return pd.Series(0.0, index=tmax_series.index, dtype=float)

    if planting_ts < data_start and initial_gdd > 0:
        log.info(
            "  Starting with initial_gdd=%.0f (planting %s → data start %s)",
            initial_gdd,
            planting_ts.date(),
            data_start.date(),
        )

    # Accumulate GDD: zero before planting_date, initial_gdd + daily GDD after.
    cumulative_gdd = pd.Series(0.0, index=tmax_series.index, dtype=float)
    running_sum = initial_gdd if planting_ts < data_start else 0.0

    for date in tmax_series.index:
        if date < planting_ts:
            cumulative_gdd[date] = 0.0
            continue

        running_sum += compute_daily_gdd(
            tmax=tmax_series[date],
            tmin=tmin_series[date],
            t_base=params.t_base,
            t_ceiling=params.t_ceiling,
        )
        cumulative_gdd[date] = running_sum

    return cumulative_gdd


# GROWTH STAGE DETERMINATION

def get_growth_stage(
    cumulative_gdd: float,
    crop_type: str,
) -> int:
    """
    Determine the current growth stage from cumulative GDD.

    Growth stages are strictly forward -- once a stage is entered it cannot
    revert (GDD only accumulates, never decreases within a season).

    Args:
        cumulative_gdd: Accumulated GDD since planting
        crop_type:      Key into CROP_PARAMS

    Returns:
        Integer stage code (0-4). See STAGE_NAMES for human-readable labels.
    """
    params = get_crop_params(crop_type)

    if cumulative_gdd < params.gdd_emergence:
        return STAGE_PRE_EMERGENCE
    elif cumulative_gdd < params.gdd_dev_end:
        return STAGE_DEVELOPMENT
    elif cumulative_gdd < params.gdd_mid_end:
        return STAGE_MID_SEASON
    elif cumulative_gdd < params.gdd_maturity:
        return STAGE_LATE_SEASON
    else:
        return STAGE_POST_MATURITY


def get_growth_stage_series(
    cumulative_gdd_series: pd.Series,
    crop_type: str,
) -> pd.Series:
    """Vectorized growth stage computation for a GDD time series."""
    params = get_crop_params(crop_type)
    gdd = cumulative_gdd_series.values
    stages = np.select(
        [
            gdd < params.gdd_emergence,
            gdd < params.gdd_dev_end,
            gdd < params.gdd_mid_end,
            gdd < params.gdd_maturity,
        ],
        [STAGE_PRE_EMERGENCE, STAGE_DEVELOPMENT,
            STAGE_MID_SEASON, STAGE_LATE_SEASON],
        default=STAGE_POST_MATURITY,
    )
    return pd.Series(stages, index=cumulative_gdd_series.index, dtype=int)

# DYNAMIC IRRIGATION THRESHOLD


def get_dynamic_threshold(
    cumulative_gdd: float,
    crop_type: str,
    trigger_base: float,
) -> float:
    """
    Compute the dynamic irrigation threshold for the current growth stage.

    threshold(t) = trigger_base + delta_offset(growth_stage, crop_type)

    Where:
        trigger_base is the soil-texture baseline irrigation trigger (cbar),
        approximately near field-capacity/depletion setpoints used operationally.
        delta_offset is growth-stage-dependent for tension sensors
        (irrigate when measured tension > threshold):
            Negative delta -> irrigate EARLIER (lower threshold; triggers at
                              less tension). Used during stress-sensitive stages.
            Positive delta -> allow MORE drying (higher threshold; controlled
                              deficit). Used during maturation.

    Example (maize on loam, trigger_base=45 cbar):
        Pre-emergence: 45 + 0   = 45 cbar
        Development:   45 - 5   = 40 cbar  (irrigate earlier -- leaf expansion)
        Mid-season:    45 - 15  = 30 cbar  (irrigate MUCH earlier -- flowering)
        Late-season:   45 + 10  = 55 cbar  (allow drying -- kernel hardening)

    Args:
        cumulative_gdd: Current accumulated GDD
        crop_type:      Key into CROP_PARAMS
        trigger_base:   Soil-type-specific base threshold (cbar). Typical
                values: Sandy=35, Sandy Loam=40, Loam=45, Clay Loam=50

    Returns:
        Dynamic threshold in cbar. Always >= 0.
    """
    params = get_crop_params(crop_type)
    stage = get_growth_stage(cumulative_gdd, crop_type)

    delta_map = {
        STAGE_PRE_EMERGENCE: params.delta_pre_emergence,
        STAGE_DEVELOPMENT:   params.delta_development,
        STAGE_MID_SEASON:    params.delta_mid_season,
        STAGE_LATE_SEASON:   params.delta_late_season,
        STAGE_POST_MATURITY: 0.0,
    }
    delta = delta_map.get(stage, 0.0)
    return max(0.0, trigger_base + delta)


def get_dynamic_threshold_series(
    cumulative_gdd_series: pd.Series,
    crop_type: str,
    trigger_base: float,
) -> pd.Series:
    """Vectorized dynamic threshold computation for a GDD time series."""
    params = get_crop_params(crop_type)
    stages = get_growth_stage_series(cumulative_gdd_series, crop_type).values
    deltas = np.select(
        [
            stages == STAGE_PRE_EMERGENCE,
            stages == STAGE_DEVELOPMENT,
            stages == STAGE_MID_SEASON,
            stages == STAGE_LATE_SEASON,
        ],
        [
            params.delta_pre_emergence,
            params.delta_development,
            params.delta_mid_season,
            params.delta_late_season,
        ],
        default=0.0,
    )
    return pd.Series(
        np.maximum(0.0, trigger_base + deltas),
        index=cumulative_gdd_series.index,
    )

# CROP COEFFICIENT -- GDD-BASED


def _compute_kc_gdd_from_params(cumulative_gdd: float, params: CropParams) -> float:
    """
    Internal: compute Kc_GDD given explicit CropParams.

    Extracted from compute_kc_gdd so tests can exercise arbitrary CropParams
    directly without mutating the global CROP_PARAMS registry. All public
    Kc_GDD functions delegate here.

    Kc curve segments:
      Pre-emergence:  Kc = Kc_ini  (soil evaporation only)
      Development:    Kc rises linearly Kc_ini -> Kc_mid
      Mid-season:     Kc constant at Kc_mid  (full canopy)
      Late-season:    Kc falls linearly Kc_mid -> Kc_end
      Post-maturity:  Kc = Kc_ini  (bare-soil evaporation)
    """
    if cumulative_gdd <= 0 or cumulative_gdd < params.gdd_emergence:
        return params.kc_ini

    if cumulative_gdd < params.gdd_dev_end:
        dev_span = params.gdd_dev_end - params.gdd_emergence
        if dev_span <= 0:
            return params.kc_mid
        frac = (cumulative_gdd - params.gdd_emergence) / dev_span
        return params.kc_ini + frac * (params.kc_mid - params.kc_ini)

    if cumulative_gdd < params.gdd_mid_end:
        return params.kc_mid

    if cumulative_gdd < params.gdd_maturity:
        late_span = params.gdd_maturity - params.gdd_mid_end
        if late_span <= 0:
            return params.kc_end
        frac = (cumulative_gdd - params.gdd_mid_end) / late_span
        return params.kc_mid + frac * (params.kc_end - params.kc_mid)

    # Post-maturity: Kc_ini (not 0) because even fallow/stubble fields
    # lose water to soil evaporation.
    return params.kc_ini


def compute_kc_gdd(
    cumulative_gdd: float,
    crop_type: str,
) -> float:
    """
    Compute the GDD-based crop coefficient using linear interpolation
    between FAO-56 stage Kc values.

    Indexing by GDD instead of calendar days is the physically correct
    approach -- crop development is driven by thermal time, not elapsed time.
    The same maize variety reaches Kc_mid=1.20 at ~700 GDD regardless of
    whether that takes 60 days (Nabeul) or 90 days (Dresden).
    """
    params = get_crop_params(crop_type)
    return _compute_kc_gdd_from_params(cumulative_gdd, params)


def compute_kc_gdd_series(
    cumulative_gdd_series: pd.Series,
    crop_type: str,
) -> pd.Series:
    """Vectorized Kc_GDD computation using numpy operations."""
    params = get_crop_params(crop_type)
    gdd = cumulative_gdd_series.values.astype(float)
    kc = np.full_like(gdd, params.kc_ini)

    dev_span = params.gdd_dev_end - params.gdd_emergence
    if dev_span > 0:
        mask = (gdd >= params.gdd_emergence) & (gdd < params.gdd_dev_end)
        frac = (gdd[mask] - params.gdd_emergence) / dev_span
        kc[mask] = params.kc_ini + frac * (params.kc_mid - params.kc_ini)

    kc[(gdd >= params.gdd_dev_end) & (gdd < params.gdd_mid_end)] = params.kc_mid

    late_span = params.gdd_maturity - params.gdd_mid_end
    if late_span > 0:
        mask = (gdd >= params.gdd_mid_end) & (gdd < params.gdd_maturity)
        frac = (gdd[mask] - params.gdd_mid_end) / late_span
        kc[mask] = params.kc_mid + frac * (params.kc_end - params.kc_mid)

    kc[gdd >= params.gdd_maturity] = params.kc_ini

    return pd.Series(kc, index=cumulative_gdd_series.index)


# CROP COEFFICIENT -- SATELLITE-DERIVED

def compute_kc_ndvi(
    ndvi: Union[float, np.ndarray],
    crop_type: str,
) -> Union[float, np.ndarray]:
    """
    Compute Kc from satellite-observed NDVI (Glenn et al. 2011).

        Kc_NDVI = clip(1.457 * NDVI - 0.10, Kc_ini, 1.35)

    Coefficients calibrated for irrigated crops across multiple studies.
    Physical basis: NDVI linearly correlates with fractional vegetation
    cover, which directly partitions ET0 into transpiration vs soil
    evaporation.

    Limitation: NDVI saturates at NDVI >= 0.7-0.8 in dense canopies
    (mid-season), suppressing sensitivity to real Kc variation. Use
    compute_kc_ndre for mid-season where NDRE is available.

    Returns NaN if NDVI is NaN (missing optical data).
    """
    ndvi_arr = np.atleast_1d(ndvi)
    params = get_crop_params(crop_type)
    # Upper clip: crop-specific ceiling, not global 1.35.

    kc_upper = min(params.kc_mid * KC_UPPER_TOLERANCE, KC_NDVI_UPPER_CAP)
    kc = np.clip(NDVI_KC_SLOPE * ndvi_arr +
                 NDVI_KC_INTERCEPT, params.kc_ini, kc_upper)
    result = np.where(np.isnan(ndvi_arr), np.nan, kc)
    return float(result.squeeze()) if np.ndim(ndvi) == 0 else result


def compute_kc_ndre(
    ndre: Union[float, np.ndarray],
    crop_type: str,
) -> Union[float, np.ndarray]:
    """
    Compute Kc from satellite-observed NDRE (Delegido et al. 2013, adapted).

        Kc_NDRE = clip(2.5 * NDRE - 0.05, Kc_ini, 1.35)

    NDRE (Normalised Difference Red Edge, Sentinel-2 B05/B8A) is sensitive
    to chlorophyll content and LAI in the range where NDVI saturates
    (NDVI >= 0.7). Typical NDRE range for crops: 0.1-0.6.

    Coefficient derivation:
        slope=2.5 ensures the same Kc dynamic range as the NDVI formula
        across the narrower NDRE domain (0.1-0.6 vs 0-1 for NDVI).
        At NDRE=0.45 (full canopy): Kc ~= 1.075, consistent with Kc_mid
        for most annual crops.

    Preferred over NDVI during STAGE_MID_SEASON where NDVI saturates.
    Falls back to NDVI during development and late-season where NDRE
    loses sensitivity to chlorophyll degradation patterns.

    Returns NaN if NDRE is NaN.
    """
    ndre_arr = np.atleast_1d(ndre)
    params = get_crop_params(crop_type)
    # Same crop-specific ceiling as compute_kc_ndvi. See note there.
    kc_upper = min(params.kc_mid * KC_UPPER_TOLERANCE, KC_NDRE_UPPER_CAP)
    kc = np.clip(NDRE_KC_SLOPE * ndre_arr +
                 NDRE_KC_INTERCEPT, params.kc_ini, kc_upper)
    result = np.where(np.isnan(ndre_arr), np.nan, kc)
    return float(result.squeeze()) if np.ndim(ndre) == 0 else result


def _select_satellite_kc(
    cumulative_gdd: float,
    ndvi: float,
    ndre: float,
    crop_type: str,
) -> tuple:
    """
    Stage-aware selection between NDVI- and NDRE-derived Kc.

    Selection logic:
        STAGE_MID_SEASON:
            NDRE preferred -- NDVI saturates at dense canopy (>= 0.7).
            Falls back to NDVI if NDRE is unavailable.
        All other stages:
            NDVI preferred -- better signal for sparse canopy (development)
            and during senescence where NDRE shifts non-linearly with
            chlorophyll degradation.

    Args:
        cumulative_gdd: Current accumulated GDD
        ndvi:           Latest NDVI value (NaN if unavailable)
        ndre:           Latest NDRE value (NaN if unavailable)
        crop_type:      Key into CROP_PARAMS

    Returns:
        (kc_satellite, source) where source is 'ndre', 'ndvi', or 'none'.
        Returns (np.nan, 'none') when no valid satellite data is available.
    """
    stage = get_growth_stage(cumulative_gdd, crop_type)

    if stage == STAGE_MID_SEASON and not np.isnan(ndre):
        kc_sat = compute_kc_ndre(ndre, crop_type)
        if not np.isnan(kc_sat):
            return kc_sat, "ndre"

    if not np.isnan(ndvi):
        kc_sat = compute_kc_ndvi(ndvi, crop_type)
        if not np.isnan(kc_sat):
            return kc_sat, "ndvi"

    return np.nan, "none"


def _gdd_plausibility_weight(
    kc_satellite: float,
    cumulative_gdd: float,
    crop_type: str,
) -> float:
    """
    Penalise satellite Kc when it implies a growth stage that GDD says
    is thermally impossible.

    The satellite reports observed canopy state; GDD reports the thermal
    budget accumulated since planting. If the satellite signals a dense,
    full-canopy Kc (~= Kc_mid) but GDD indicates only 200 heat units have
    accumulated since planting, something is wrong -- likely cloud/shadow
    contamination, field misidentification, or sensor artefact.

    Stage divergence -> penalty:
        0   stages ahead of GDD: weight = 1.0  (plausible)
        1   stage  ahead of GDD: weight = 0.5  (possible microclimate/early season)
        2+  stages ahead of GDD: weight = 0.0  (biologically implausible)

    The satellite-implied stage is inferred from Kc relative to the
    crop's Kc_ini / Kc_mid parameters:
        Kc <= Kc_ini + 0.10                   -> implies Pre-emergence
        Kc_ini + 0.10 < Kc < 0.65 * Kc_mid   -> implies Development
        Kc >= 0.65 * Kc_mid                   -> implies Mid-season or later

    Args:
        kc_satellite:   Satellite-derived Kc value (NDVI or NDRE path)
        cumulative_gdd: Current accumulated GDD
        crop_type:      Key into CROP_PARAMS

    Returns:
        Plausibility weight in [0, 1].
    """
    if np.isnan(kc_satellite):
        return 0.0

    params = get_crop_params(crop_type)
    gdd_stage = get_growth_stage(cumulative_gdd, crop_type)

    kc_range = params.kc_mid - params.kc_ini
    stress_kc_floor = params.kc_ini + MIDSEASON_STRESS_KC_MARGIN * kc_range

    # Stress safeguard: if thermal stage is mid-season but observed canopy Kc
    # is near Kc_ini, avoid discarding the satellite signal entirely.
    if gdd_stage == STAGE_MID_SEASON and kc_satellite < stress_kc_floor:
        return MIDSEASON_STRESS_PLAUSIBILITY

    # Threshold to distinguish Mid-season from Development implied by satellite.
    # Uses 65% of the crop's Kc range (proportional, not a fixed offset).
    kc_thresh_mid = params.kc_ini + PLAUSIBILITY_MID_FRAC * kc_range

    # Threshold to distinguish Development from Pre-emergence.
    # Uses 10% of the crop's Kc range — proportional so it stays below
    # kc_thresh_mid regardless of how narrow the Kc range is.
    #
    # The original hardcoded `kc_ini + 0.10` breaks crops with small Kc
    # ranges: for olive (range=0.15), kc_ini+0.10=0.65 exceeds
    # kc_thresh_mid=0.6475, inverting the thresholds and creating a dead
    # zone [0.55, 0.6475) where valid Development Kc values are misclassified
    # as Pre-emergence, suppressing the plausibility penalty when the satellite
    # shows more canopy than GDD supports.
    kc_thresh_dev = params.kc_ini + PLAUSIBILITY_DEV_FRAC * kc_range

    if kc_satellite >= kc_thresh_mid:
        sat_implied_stage = STAGE_MID_SEASON
    elif kc_satellite > kc_thresh_dev:
        sat_implied_stage = STAGE_DEVELOPMENT
    else:
        sat_implied_stage = STAGE_PRE_EMERGENCE

    stage_lead = sat_implied_stage - gdd_stage
    if stage_lead <= 0:
        return 1.0
    elif stage_lead == 1:
        return 0.5
    else:
        return 0.0


def sar_adjusted_ndvi_age(
    ndvi_age_series: pd.Series,
    vv_db_series: pd.Series,
    vv_canopy_threshold_db: float = SAR_CANOPY_THRESHOLD_DB,
    max_trust_hours: float = SAR_MAX_TRUST_HOURS,
) -> pd.Series:
    """
    Reduce effective NDVI staleness when SAR confirms the canopy is present.

    When optical sensors are blocked by cloud cover, NDVI age increases
    even though the canopy itself has not changed. SAR (VV polarisation,
    C-band) is cloud-transparent and can confirm whether a canopy is still
    present.

    If VV backscatter exceeds vv_canopy_threshold_db (established crop
    present), the stale NDVI is likely representative -- cloud cover, not
    senescence, caused the optical gap. The effective age is capped at
    max_trust_hours (default 30 days) rather than growing unbounded.

    This prevents the quality decay factor from zeroing out the satellite
    weight during legitimate cloud-cover windows, while still allowing
    normal decay when SAR also shows backscatter has dropped (senescence
    or harvest has occurred).

    Typical VV backscatter (C-band, 10 m Sentinel-1):
        Bare soil:            -15 to -12 dB
        Sparse vegetation:    -13 to -10 dB
        Established crop:     -10 to  -6 dB
        Dense maize at peak:   -8 to  -4 dB

    Args:
        ndvi_age_series:        Hours since last valid NDVI observation
        vv_db_series:           SAR VV backscatter in dB (NaN = no SAR data)
        vv_canopy_threshold_db: dB threshold above which canopy is confirmed
        max_trust_hours:        Age cap applied when SAR confirms canopy

    Returns:
        Age series with staleness caps applied where SAR confirms canopy.
        Rows where SAR is NaN are unchanged.
    """
    adjusted = ndvi_age_series.copy()
    sar_present = ~vv_db_series.isna()
    canopy_confirmed = sar_present & (vv_db_series > vv_canopy_threshold_db)

    # Guard: only cap rows where a real NDVI observation has previously occurred
    # (finite age). Capping inf would manufacture trust for rows that have
    # never had an NDVI observation — SAR confirms the canopy is present, but
    # it cannot substitute for an actual optical reading. An age of inf means
    # "no observation yet", not "observation is very stale".
    has_prior_observation = ~np.isinf(ndvi_age_series)
    canopy_confirmed = canopy_confirmed & has_prior_observation

    adjusted[canopy_confirmed] = adjusted[canopy_confirmed].clip(
        upper=max_trust_hours)

    n_adjusted = int(canopy_confirmed.sum())
    if n_adjusted > 0:
        log.info(
            "  SAR age correction: %s rows capped at %.0fh (VV > %s dB confirms canopy despite stale optical)",
            n_adjusted,
            max_trust_hours,
            vv_canopy_threshold_db,
        )
    return adjusted


# BLENDED DYNAMIC Kc
def compute_kc_dynamic(
    cumulative_gdd: float,
    crop_type: str,
    ndvi: float = np.nan,
    ndvi_age_hours: float = np.inf,
    ndre: float = np.nan,
) -> float:
    """
    Compute the blended dynamic crop coefficient.

        Kc_dynamic = (1 - w) * Kc_GDD + w * Kc_satellite

    Where:
        Kc_satellite  is NDRE (mid-season) or NDVI (all other stages),
                      selected by _select_satellite_kc.
        w             is a quality weight combining three factors:
                          signal strength  -- scales with NDVI magnitude
                                             (more canopy -> more confident)
                          freshness decay  -- reduces w as observation ages
                          plausibility     -- zeros w when satellite implies
                                             a stage GDD says is impossible

    Falls back entirely to GDD-based Kc when satellite data is absent
    or biologically inconsistent with thermal development.

    For stale observations, satellite influence decays by age, with a
    conservative freshness floor applied during mid-season where canopy
    state changes more slowly.

    Args:
        cumulative_gdd:  Current accumulated GDD
        ndvi:            Latest NDVI value (NaN if unavailable)
        ndvi_age_hours:  Hours since the last real NDVI observation
        crop_type:       Key into CROP_PARAMS
        ndre:            Latest NDRE value (NaN if unavailable).
                         Used instead of NDVI during STAGE_MID_SEASON.

    Returns:
        Kc_dynamic in range [Kc_ini, ~1.35].
    """
    kc_gdd = compute_kc_gdd(cumulative_gdd, crop_type)

    # Age information is required for any satellite blending: without it we
    # cannot assess freshness, so we fall back to GDD entirely.
    if np.isinf(ndvi_age_hours):
        return kc_gdd

    # Determine which satellite source is usable at the current stage.
    # NDRE is only meaningful at mid-season (other stages → fall back to NDVI).
    stage = get_growth_stage(cumulative_gdd, crop_type)
    ndre_usable = not np.isnan(ndre) and stage == STAGE_MID_SEASON
    ndvi_usable = not np.isnan(ndvi)

    if not ndvi_usable and not ndre_usable:
        return kc_gdd

    kc_sat, _ = _select_satellite_kc(
        cumulative_gdd, ndvi, ndre, crop_type)
    if np.isnan(kc_sat):
        return kc_gdd

    # Signal-strength weight.
    # Primary: NDVI (all stages). Its magnitude tracks fractional cover and
    # is the validated signal for scaling the quality weight.
    # Fallback: when NDVI is absent but NDRE is available at mid-season, derive
    # a canopy-density proxy from NDRE. At mid-season, NDRE 0.1-0.6 corresponds
    # roughly to NDVI 0.3-0.85 for annual field crops (empirical scale ≈ ×1.5).
    # This is conservative: the proxy gives ~0.6 weight at NDRE=0.45, not 0.75.
    if ndvi_usable:
        ndvi_for_weight = ndvi
    else:
        # NDRE-only at mid-season
        ndvi_for_weight = float(np.clip(ndre * NDRE_NDVI_PROXY, 0.0, 1.0))

    w_base = np.clip(ndvi_for_weight * NDVI_WEIGHT_SLOPE, 0.0, NDVI_WEIGHT_CAP)

    # Freshness decay.
    if ndvi_age_hours <= SAT_FRESH_HOURS:       # <= 15 days: full trust
        quality_factor = 1.0
    elif ndvi_age_hours <= SAT_REDUCED_HOURS:   # 15-30 days: reduced
        quality_factor = SAT_REDUCED_FACTOR
    elif ndvi_age_hours <= SAT_LOW_HOURS:       # 30-45 days: low
        quality_factor = SAT_LOW_FACTOR
    else:                           # > 45 days: no trust
        quality_factor = 0.0

    # Stage-conditional floor: retain minimum freshness confidence at
    # mid-season where canopy state is relatively stable.
    if stage == STAGE_MID_SEASON:
        quality_factor = max(quality_factor, MIDSEASON_FRESHNESS_FLOOR)

    # GDD plausibility check.
    plausibility = _gdd_plausibility_weight(kc_sat, cumulative_gdd, crop_type)

    w = w_base * quality_factor * plausibility
    return (1.0 - w) * kc_gdd + w * kc_sat


def compute_kc_dynamic_series(
    gdd_series: pd.Series,
    ndvi_series: pd.Series,
    age_series: pd.Series,
    crop_type: str,
    ndre_series: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Vectorized Kc_dynamic computation for entire DataFrame columns.

    Args:
        gdd_series:   Cumulative GDD series
        ndvi_series:  NDVI observations (NaN where unavailable)
        age_series:   Hours since last valid NDVI observation
        crop_type:    Key into CROP_PARAMS
        ndre_series:  NDRE observations (NaN where unavailable). When
                      provided, NDRE replaces NDVI during STAGE_MID_SEASON.

    Returns:
        Series of Kc_dynamic values with same index as gdd_series.
    """
    params = get_crop_params(crop_type)

    kc_gdd_vals = compute_kc_gdd_series(gdd_series, crop_type).values
    ndvi_vals = ndvi_series.values.astype(float)
    age_vals = age_series.values.astype(float)

    # Compute stages once; reused for NDRE selection, weight proxy, and plausibility.
    stages_arr = get_growth_stage_series(gdd_series, crop_type).values
    mid_mask_base = stages_arr == STAGE_MID_SEASON

    # Stage-aware satellite Kc: default to NDVI.
    # np.where returns a new array, so kc_sat is always writeable.
    kc_sat = compute_kc_ndvi(ndvi_vals, crop_type).copy()

    # Replace with NDRE during mid-season where available.
    ndre_vals = None
    if ndre_series is not None:
        ndre_vals = ndre_series.values.astype(float)
        mid_mask = mid_mask_base & ~np.isnan(ndre_vals)
        if mid_mask.any():
            kc_sat[mid_mask] = compute_kc_ndre(ndre_vals, crop_type)[mid_mask]
            log.info(
                "  NDRE used for Kc in %s mid-season rows (%.1f%% of total)",
                mid_mask.sum(),
                mid_mask.mean() * 100.0,
            )

    # Signal-strength weight.
    # Primary: NDVI. When NDVI is absent but NDRE is available at mid-season,
    # derive an NDVI-equivalent proxy (NDRE * 1.5 ≈ NDVI for dense canopy).
    # This matches the scalar path in compute_kc_dynamic.
    if ndre_vals is not None:
        ndre_proxy = np.clip(ndre_vals * NDRE_NDVI_PROXY, 0.0, 1.0)
        ndvi_for_weight = np.where(
            np.isnan(ndvi_vals) & mid_mask_base & ~np.isnan(ndre_vals),
            ndre_proxy,
            np.where(np.isnan(ndvi_vals), 0.0, ndvi_vals),
        )
    else:
        ndvi_for_weight = np.where(np.isnan(ndvi_vals), 0.0, ndvi_vals)
    w_base = np.clip(ndvi_for_weight * NDVI_WEIGHT_SLOPE, 0.0, NDVI_WEIGHT_CAP)

    # Freshness decay
    quality = np.where(
        age_vals <= SAT_FRESH_HOURS, 1.0,
        np.where(age_vals <= SAT_REDUCED_HOURS, SAT_REDUCED_FACTOR,
                 np.where(age_vals <= SAT_LOW_HOURS, SAT_LOW_FACTOR, 0.0))
    )

    # Match scalar behavior: floor freshness during mid-season.
    quality = np.where(
        mid_mask_base,
        np.maximum(quality, MIDSEASON_FRESHNESS_FLOOR),
        quality,
    )

    # GDD plausibility (vectorized) — mirrors _gdd_plausibility_weight scalar logic.
    # Uses proportional thresholds so narrow-Kc crops (olive) behave correctly.
    kc_range = params.kc_mid - params.kc_ini
    kc_thresh_mid_v = params.kc_ini + PLAUSIBILITY_MID_FRAC * kc_range
    kc_thresh_dev_v = params.kc_ini + PLAUSIBILITY_DEV_FRAC * kc_range
    sat_implied = np.where(
        kc_sat >= kc_thresh_mid_v, STAGE_MID_SEASON,
        np.where(kc_sat > kc_thresh_dev_v, STAGE_DEVELOPMENT,
                 STAGE_PRE_EMERGENCE)
    )
    # stages_arr already computed above
    stage_lead = sat_implied - stages_arr
    plausibility = np.where(stage_lead <= 0, 1.0,
                            np.where(stage_lead == 1, 0.5, 0.0))

    # Mirror scalar stress safeguard for consistency with compute_kc_dynamic.
    stress_kc_floor_v = params.kc_ini + MIDSEASON_STRESS_KC_MARGIN * kc_range
    stress_mask = (
        (stages_arr == STAGE_MID_SEASON)
        & (kc_sat < stress_kc_floor_v)
    )
    plausibility = np.where(
        stress_mask,
        MIDSEASON_STRESS_PLAUSIBILITY,
        plausibility,
    )

    w = w_base * quality * plausibility
    # Zero out weight where: NDVI is absent AND no NDRE proxy was used,
    # age is unknown (inf), or no valid satellite Kc could be derived.
    # The NDRE-proxy rows already have non-zero w_base, so they are not zeroed here.
    w[np.isinf(age_vals) | np.isnan(kc_sat)] = 0.0
    # For rows where both NDVI and NDRE-proxy are absent, w_base is already 0,
    # but be explicit: zero weight where ndvi is NaN and ndre proxy wasn't applied.
    if ndre_vals is not None:
        no_satellite = np.isnan(ndvi_vals) & (
            ~mid_mask_base | np.isnan(ndre_vals)
        )
    else:
        no_satellite = np.isnan(ndvi_vals)
    w[no_satellite] = 0.0

    kc_dynamic = np.where(w == 0, kc_gdd_vals, (1.0 - w)
                          * kc_gdd_vals + w * kc_sat)
    return pd.Series(kc_dynamic, index=gdd_series.index)

# CLI


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Phenology engine: GDD, growth stages, dynamic Kc and thresholds"
    )
    parser.add_argument("--list-crops", action="store_true",
                        help="Print all crop parameters")
    args = parser.parse_args()

    if args.list_crops:
        print(f"\n{'Crop':12s} {'T_base':>6s} {'T_ceil':>6s} {'Emerge':>7s} "
              f"{'DevEnd':>7s} {'MidEnd':>7s} {'Mature':>7s} "
              f"{'Kc_ini':>6s} {'Kc_mid':>6s} {'Kc_end':>6s}")
        print("-" * 90)
        for key, p in CROP_PARAMS.items():
            print(f"{key:12s} {p.t_base:6.1f} {p.t_ceiling:6.1f} "
                  f"{p.gdd_emergence:7.0f} {p.gdd_dev_end:7.0f} "
                  f"{p.gdd_mid_end:7.0f} {p.gdd_maturity:7.0f} "
                  f"{p.kc_ini:6.2f} {p.kc_mid:6.2f} {p.kc_end:6.2f}")
    else:
        print("Use --list-crops to see crop parameters.")
        print("Run tests with:  python test_phenology.py")


if __name__ == "__main__":
    main()
