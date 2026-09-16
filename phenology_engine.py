#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phenology_engine.py — Growing Degree Days, crop growth stage tracking,
                      dynamic Kc computation, and stage-aware tension triggers.

Satellite fusion strategy:
    NDVI  — conservative production input across crop stages
    NDRE  — observational by default; experimental Kc input only when enabled
    SAR   — optional future input; not supplied by the current runtime source
    GDD   — biological plausibility check; caps satellite weight when the
            observed Kc implies a stage the thermal budget cannot support

"""

import argparse
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

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
    STAGE_NAMES,
)

log = logging.getLogger(__name__)

# ALGORITHM CONSTANTS

# Kc from NDVI (Glenn et al. 2011 / Kamble et al. 2013)
# Calibrated across irrigated crop studies; linear fit between NDVI and Kc.
NDVI_KC_SLOPE = 1.457   # Kc_NDVI = NDVI_KC_SLOPE * NDVI + NDVI_KC_INTERCEPT
NDVI_KC_INTERCEPT = -0.1725
KC_NDVI_UPPER_CAP = 1.35    # global cap before crop-specific ceiling is applied

# Experimental Kc from NDRE. Delegido et al. (2011) supports red-edge use for
# LAI/chlorophyll retrieval, not this Kc equation. This internal range mapping
# is disabled in production fusion unless explicitly enabled for a field trial.
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
# cap=0.35   → uncalibrated EO remains a correction to the GDD prior
NDVI_WEIGHT_SLOPE = 0.85
NDVI_WEIGHT_CAP = 0.35

# NDRE-to-NDVI proxy scale factor when NDVI is absent at mid-season.
# Empirical: NDRE 0.1–0.6 ≈ NDVI 0.15–0.90 for annual field crops.
NDRE_NDVI_PROXY = 1.5

# Satellite freshness is supplied as a 0..1 quality score derived from the
# provider's observed acquisition gaps. Crop response timing remains relevant
# to tension-trend validation, but no crop-specific imagery expiry is encoded.

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

_FORECAST_HORIZON_RE = re.compile(
    r"^(\d+(?:\.\d+)?)(?:h|hours?)?$", re.IGNORECASE)


def phenology_season_start(
    timestamp: Union[str, datetime, pd.Timestamp],
    planting_date: Union[str, datetime, pd.Timestamp],
    crop_type: str,
) -> pd.Timestamp:
    """Return the active season start for annual or perennial crops."""
    value = pd.Timestamp(timestamp)
    planted = pd.Timestamp(planting_date)
    if value.tzinfo is not None and planted.tzinfo is None:
        planted = planted.tz_localize(value.tzinfo)
    elif value.tzinfo is None and planted.tzinfo is not None:
        planted = planted.tz_localize(None)
    elif value.tzinfo is not None and planted.tzinfo is not None:
        planted = planted.tz_convert(value.tzinfo)
    planted = planted.normalize()
    if not get_crop_params(crop_type).is_perennial or value < planted:
        return planted
    try:
        anniversary = planted.replace(year=value.year)
    except ValueError:
        anniversary = planted.replace(year=value.year, day=28)
    if value.normalize() < anniversary:
        try:
            anniversary = planted.replace(year=value.year - 1)
        except ValueError:
            anniversary = planted.replace(year=value.year - 1, day=28)
    return anniversary.normalize()


def compute_daily_gdd(
    tmax: float,
    tmin: float,
    t_base: float,
    t_ceiling: float,
) -> float:
    """
    Compute Growing Degree Days for a single day.

    Uses a bounded modified-average method:
        T* = clip(T, T_base, T_ceiling)
        GDD = max(0, (Tmax* + Tmin*) / 2 - T_base)

    Tmax and Tmin are clipped to both crop bounds before averaging. This is the
    rule used by runtime, training, and forecast code. The selected method and
    its GDD breakpoints must be calibrated together; GDD conventions are not
    interchangeable.

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

    t_eff_max = float(np.clip(tmax, t_base, t_ceiling))
    t_eff_min = float(np.clip(tmin, t_base, t_ceiling))
    t_mean = (t_eff_max + t_eff_min) / 2.0
    return max(0.0, t_mean - t_base)


def daily_temperature_extrema(
    temperature_series: pd.Series,
    timezone_name: Optional[str] = None,
    through=None,
) -> Tuple[pd.Series, pd.Series]:
    """Aggregate temperatures by local calendar day, optionally only to now."""
    values = pd.to_numeric(temperature_series, errors="coerce").copy()
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("temperature series must use a DatetimeIndex")
    if timezone_name:
        # Naive training indices already represent plot-local wall time. Aware
        # provider indices are converted from their source zone.
        if values.index.tz is not None:
            values.index = values.index.tz_convert(timezone_name)
    values = values.sort_index()
    if through is not None:
        cutoff = pd.Timestamp(through)
        if values.index.tz is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        elif values.index.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(values.index.tz)
        elif values.index.tz is not None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(values.index.tz)
        values = values[values.index <= cutoff]
    if values.dropna().empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    daily = values.resample("D")
    return daily.max(), daily.min()


def compute_gdd_series(
    tmax_series: pd.Series,
    tmin_series: pd.Series,
    planting_date: Union[str, datetime, pd.Timestamp],
    crop_type: str,
    initial_gdd: float = 0.0,
) -> pd.Series:
    """
    Compute cumulative GDD from planting_date forward.

    Annual crops accumulate from their explicit planting date. Perennial crops
    use that date's month/day as their configured annual phenological-season
    start and reset GDD at each anniversary.

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
    perennial_season_start = None

    for date in tmax_series.index:
        if date < planting_ts:
            cumulative_gdd[date] = 0.0
            continue

        if params.is_perennial:
            anniversary = phenology_season_start(
                date, planting_ts, crop_type)
            if perennial_season_start is None:
                perennial_season_start = anniversary
            elif anniversary != perennial_season_start:
                running_sum = 0.0
                perennial_season_start = anniversary

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

    GDD can be a better phenology index than fixed calendar days, but the
    breakpoints are cultivar- and environment-dependent.  Values in crops.py
    are priors and require local calibration before operational use.
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

        Kc_NDVI = clip(1.457 * NDVI - 0.1725, Kc_ini, 1.35)

    Coefficients calibrated for irrigated crops across multiple studies.
    Physical basis: NDVI linearly correlates with fractional vegetation
    cover, which directly partitions ET0 into transpiration vs soil
    evaporation.

    Limitation: NDVI saturates at NDVI >= 0.7-0.8 in dense canopies
    (mid-season), suppressing sensitivity to real Kc variation. NDRE may be
    evaluated there, but its internal Kc mapping is experimental and opt-in.

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
    Compute an experimental Kc from satellite-observed NDRE.

        Kc_NDRE = clip(2.5 * NDRE - 0.05, Kc_ini, 1.35)

    NDRE (Normalised Difference Red Edge, Sentinel-2 B05/B8A) is sensitive
    to chlorophyll content and LAI in the range where NDVI saturates
    (NDVI >= 0.7). Typical NDRE range for crops: 0.1-0.6.

    Coefficient derivation:
        slope=2.5 ensures the same Kc dynamic range as the NDVI formula
        across the narrower NDRE domain (0.1-0.6 vs 0-1 for NDVI).
        At NDRE=0.45 (full canopy): Kc ~= 1.075, consistent with Kc_mid
        for most annual crops.

    This range mapping is not a published Delegido Kc relationship and is
    disabled by default in dynamic Kc fusion. It exists only for explicitly
    configured, locally evaluated field trials.

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
    allow_experimental_ndre: bool = False,
) -> tuple:
    """
    Stage-aware selection between NDVI- and NDRE-derived Kc.

    Selection logic:
        NDVI is the default in every stage. When the experimental flag is
        enabled, valid NDRE replaces it only at mid-season.

    Args:
        cumulative_gdd: Current accumulated GDD
        ndvi:           Latest NDVI value (NaN if unavailable)
        ndre:           Latest NDRE value (NaN if unavailable)
        crop_type:      Key into CROP_PARAMS
        allow_experimental_ndre: Permit the unvalidated NDRE Kc mapping.

    Returns:
        (kc_satellite, source) where source is 'ndre', 'ndvi', or 'none'.
        Returns (np.nan, 'none') when no valid satellite data is available.
    """
    stage = get_growth_stage(cumulative_gdd, crop_type)

    if allow_experimental_ndre and stage == STAGE_MID_SEASON and not np.isnan(ndre):
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


def _stage_validation_sensitivity(stage: int) -> float:
    """Canopy consistency is informative during active crop development."""
    return 0.0 if stage == STAGE_POST_MATURITY else 1.0


def _freshness_quality(
    age_hours: float,
    stage: int,
    observation_quality: Optional[float] = None,
) -> float:
    """Use source-derived freshness; unknown cadence trusts only an exact observation."""
    del stage  # Freshness is an acquisition property, not a crop-stage duration.
    if not np.isfinite(age_hours):
        return 0.0
    if observation_quality is None or not np.isfinite(observation_quality):
        return 1.0 if age_hours <= 0 else 0.0
    return float(np.clip(observation_quality, 0.0, 1.0))


def _history_freshness_quality(
    history: pd.DataFrame,
    source: str,
    age_hours: float,
) -> float:
    """Build an empirical freshness score from source acquisition intervals."""
    if history.empty or not np.isfinite(age_hours):
        return 0.0
    availability_column = f"has_{source}"
    value_column = f"sat_{source}"
    if availability_column in history.columns:
        mask = history[availability_column].fillna(False).astype(bool)
    elif value_column in history.columns:
        mask = pd.to_numeric(history[value_column], errors="coerce").notna()
    else:
        return 0.0
    dates = pd.DatetimeIndex(history.loc[mask, "timestamp"].dropna().unique()).sort_values()
    if age_hours <= 0:
        return 1.0
    if len(dates) < 2:
        return 0.0
    gaps = np.diff(dates.asi8) / 3.6e12
    gaps = gaps[gaps > 0]
    return float(np.mean(gaps >= age_hours)) if len(gaps) else 0.0


def _parse_forecast_horizon_hours(label: str) -> Optional[float]:
    match = _FORECAST_HORIZON_RE.match(str(label).strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _fit_slope(points: List[Tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    x_values = np.asarray([point[0] for point in points], dtype=float)
    y_values = np.asarray([point[1] for point in points], dtype=float)
    if np.allclose(x_values, x_values[0]):
        return 0.0
    slope = np.polyfit(x_values, y_values, 1)[0]
    if np.isnan(slope):
        return None
    return float(slope)


def _select_satellite_validation_value(row: pd.Series, stage: int) -> Tuple[float, str]:
    ndre = row.get("sat_ndre", np.nan)
    ndvi = row.get("sat_ndvi", np.nan)

    if stage == STAGE_MID_SEASON and not pd.isna(ndre):
        return float(ndre), "ndre"
    if not pd.isna(ndvi):
        return float(ndvi), "ndvi"
    if not pd.isna(ndre):
        return float(ndre), "ndre"
    return float("nan"), "none"


def _legacy_satellite_tension_consistency(
    crop_type: str,
    growth_stage: int,
    gdd_cumulative: float,
    current_tension: float,
    stress_threshold_cbar: float,
    tension_forecast: Dict[str, float],
    satellite_history: Optional[pd.DataFrame] = None,
    satellite_ndvi: float = np.nan,
    satellite_ndre: float = np.nan,
    satellite_ndvi_age_hours: float = np.inf,
    satellite_ndre_age_hours: float = np.inf,
    satellite_ndvi_quality: Optional[float] = None,
    satellite_ndre_quality: Optional[float] = None,
    satellite_data_age_hours: float = np.inf,
    satellite_vv_db: float = np.nan,
    et0_today_mm: Optional[float] = None,
    et0_baseline_mm: Optional[float] = None,
    et0_std_mm: Optional[float] = None,
) -> Dict[str, object]:
    """Compare tension-model direction against the satellite-observed canopy state.

    The implementation is intentionally one-directional: it scores agreement and
    logs explainability factors, but it does not feed back into the threshold or
    the model.
    """
    params = get_crop_params(crop_type)
    stage_name = STAGE_NAMES.get(growth_stage, str(growth_stage))

    current_tension = float(current_tension)
    stress_threshold_cbar = float(stress_threshold_cbar)
    lag_hours = max(24.0, float(params.validation_lag_days) * 24.0)

    forecast_points: List[Tuple[float, float]] = [(0.0, current_tension)]
    for label, value in (tension_forecast or {}).items():
        horizon_hours = _parse_forecast_horizon_hours(label)
        if horizon_hours is None:
            continue
        try:
            tension_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(tension_value):
            continue
        forecast_points.append((float(horizon_hours), tension_value))
    forecast_points = sorted(forecast_points, key=lambda item: item[0])

    latest_satellite_history = pd.DataFrame()
    if satellite_history is not None and not satellite_history.empty:
        latest_satellite_history = satellite_history.copy()
        if "timestamp" in latest_satellite_history.columns:
            latest_satellite_history["timestamp"] = pd.to_datetime(
                latest_satellite_history["timestamp"], utc=True, errors="coerce")
            latest_satellite_history = latest_satellite_history.dropna(subset=[
                                                                       "timestamp"])
            latest_satellite_history = latest_satellite_history.sort_values(
                "timestamp")
            latest_satellite_history = latest_satellite_history.reset_index(
                drop=True)

    satellite_points: List[Tuple[float, float]] = []
    satellite_source = "none"
    latest_satellite_age_hours = float(satellite_data_age_hours)
    if not np.isfinite(latest_satellite_age_hours) or np.isinf(latest_satellite_age_hours):
        latest_satellite_age_hours = float(satellite_ndvi_age_hours)

    if not latest_satellite_history.empty:
        ref_timestamp = latest_satellite_history["timestamp"].max()
        lag_cutoff = ref_timestamp - pd.Timedelta(hours=lag_hours)
        lagged_history = latest_satellite_history[latest_satellite_history["timestamp"] <= lag_cutoff]

        for _, row in lagged_history.iterrows():
            value, source = _select_satellite_validation_value(
                row, growth_stage)
            if np.isnan(value):
                continue
            timestamp = row.get("timestamp")
            if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
                continue
            satellite_points.append((timestamp.value / 3.6e12, float(value)))
            satellite_source = source

        if satellite_points:
            latest_row = lagged_history.iloc[-1]
            latest_age = latest_row.get(
                "satellite_data_age", latest_satellite_age_hours)
            try:
                latest_satellite_age_hours = float(latest_age)
            except (TypeError, ValueError):
                pass

    if np.isnan(satellite_ndvi) and np.isnan(satellite_ndre):
        satellite_value = float("nan")
    else:
        satellite_value, satellite_source = _select_satellite_validation_value(
            pd.Series({"sat_ndvi": satellite_ndvi,
                      "sat_ndre": satellite_ndre}),
            growth_stage,
        )

    if np.isnan(satellite_value) and not np.isnan(satellite_ndvi):
        satellite_value = float(satellite_ndvi)
        satellite_source = "ndvi"

    satellite_kc = np.nan
    if not np.isnan(satellite_value):
        if satellite_source == "ndre":
            satellite_kc = float(compute_kc_ndre(satellite_value, crop_type))
        else:
            satellite_kc = float(compute_kc_ndvi(satellite_value, crop_type))

    kc_gdd = float(compute_kc_gdd(gdd_cumulative, crop_type))

    satellite_deficit = float("nan")
    if not np.isnan(satellite_kc):
        satellite_deficit = kc_gdd - satellite_kc

    tension_trend = _fit_slope(forecast_points)
    satellite_trend = _fit_slope(satellite_points)

    tension_scale = max(2.0, abs(stress_threshold_cbar) * 0.08)
    kc_range = max(0.05, params.kc_mid - params.kc_ini)
    satellite_scale = max(0.02, kc_range * 0.35)

    tension_signal = 0.0
    if tension_trend is not None:
        tension_signal = float(
            np.clip(tension_trend / tension_scale, -1.0, 1.0))

    satellite_signal = 0.0
    if satellite_trend is not None:
        satellite_signal = float(
            np.clip((-satellite_trend) / satellite_scale, -1.0, 1.0))

    if tension_trend is None or satellite_trend is None:
        direction_agreement = 0.5
    else:
        direction_agreement = 0.5 + 0.5 * tension_signal * satellite_signal

    response_maturity = 1.0
    if np.isfinite(latest_satellite_age_hours):
        response_maturity = float(np.clip(
            1.0 - (latest_satellite_age_hours / max(lag_hours * 2.0, lag_hours + 1.0)), 0.0, 1.0))

    stage_sensitivity = _stage_validation_sensitivity(growth_stage)
    et0_adjustment = 0.0
    if et0_today_mm is not None and et0_baseline_mm is not None and et0_std_mm is not None:
        try:
            et0_today_mm = float(et0_today_mm)
            et0_baseline_mm = float(et0_baseline_mm)
            et0_std_mm = float(et0_std_mm)
            if np.isfinite(et0_today_mm) and np.isfinite(et0_baseline_mm) and np.isfinite(et0_std_mm) and et0_std_mm > 0:
                et0_z = (et0_today_mm - et0_baseline_mm) / et0_std_mm
                et0_adjustment = float(np.clip(et0_z * 0.08, -0.18, 0.18))
        except (TypeError, ValueError):
            et0_adjustment = 0.0
    adjusted_sensitivity = float(
        np.clip(stage_sensitivity * (1.0 + et0_adjustment), 0.0, 1.0))

    # The catalog's real acquisition gaps determine freshness. This avoids
    # treating crop growth duration as if it were satellite revisit cadence.
    if satellite_source == "ndre":
        latest_satellite_age_hours = float(satellite_ndre_age_hours)
        source_quality = satellite_ndre_quality
    else:
        latest_satellite_age_hours = float(satellite_ndvi_age_hours)
        source_quality = satellite_ndvi_quality
    empirical_quality = _history_freshness_quality(
        latest_satellite_history,
        satellite_source if satellite_source in ("ndvi", "ndre") else "ndvi",
        latest_satellite_age_hours,
    )
    data_quality = _freshness_quality(
        latest_satellite_age_hours,
        growth_stage,
        empirical_quality if not latest_satellite_history.empty else source_quality,
    )

    valid_forecast_points = [
        point for point in forecast_points if np.isfinite(point[1])]
    valid_satellite_points = [
        point for point in satellite_points if np.isfinite(point[1])]
    sufficient_data = len(valid_forecast_points) >= 3 and len(
        valid_satellite_points) >= 3

    validation_score = 0.5 + (direction_agreement - 0.5) * adjusted_sensitivity
    overall_confidence = validation_score * \
        data_quality if sufficient_data else None

    if sufficient_data:
        if direction_agreement >= 0.7:
            direction_text = "agreement is strong"
        elif direction_agreement >= 0.4:
            direction_text = "agreement is moderate"
        else:
            direction_text = "agreement is weak"
        reason = (
            f"{stage_name} satellite check for {crop_type}: {direction_text}, "
            f"{satellite_source} trend compared with the tension forecast."
        )
    else:
        reason = (
            f"Insufficient data for {stage_name} satellite validation: "
            f"forecast_points={len(valid_forecast_points)}, satellite_points={len(valid_satellite_points)}."
        )

    factors = {
        "stage_sensitivity": round(stage_sensitivity, 3),
        "et0_adjustment": round(et0_adjustment, 3),
        "direction_agreement": round(direction_agreement, 3),
        "tension_trend": None if tension_trend is None else round(tension_trend, 4),
        "satellite_trend": None if satellite_trend is None else round(satellite_trend, 4),
        "satellite_source": satellite_source,
        "satellite_value": None if np.isnan(satellite_value) else round(float(satellite_value), 4),
        "satellite_kc": None if np.isnan(satellite_kc) else round(float(satellite_kc), 4),
        "satellite_deficit": None if np.isnan(satellite_deficit) else round(float(satellite_deficit), 4),
        "satellite_vv_db": None if np.isnan(satellite_vv_db) else round(float(satellite_vv_db), 2),
        "lag_hours": round(lag_hours, 1),
        "response_maturity": round(response_maturity, 3),
        "latest_satellite_age_hours": None if np.isinf(latest_satellite_age_hours) else round(float(latest_satellite_age_hours), 1),
        "tension_points": len(valid_forecast_points),
        "satellite_points": len(valid_satellite_points),
        "kc_gdd": round(kc_gdd, 4),
        "data_quality": round(data_quality, 3),
    }

    return {
        "available": True,
        "insufficient_data": not sufficient_data,
        "crop_type": crop_type,
        "growth_stage": stage_name,
        "growth_stage_code": int(growth_stage),
        "direction_agreement": round(float(direction_agreement), 3),
        "validation_score": round(float(validation_score), 3),
        "data_quality": round(float(data_quality), 3),
        "overall_confidence": None if overall_confidence is None else round(float(overall_confidence), 3),
        "factors": factors,
        "reason": reason,
    }


def _check_canopy_index(
    crop_type: str,
    growth_stage: int,
    gdd_cumulative: float,
    current_tension: float = np.nan,
    stress_threshold_cbar: float = np.nan,
    tension_forecast: Optional[Dict[str, float]] = None,
    satellite_history: Optional[pd.DataFrame] = None,
    satellite_ndvi: float = np.nan,
    satellite_ndre: float = np.nan,
    satellite_ndvi_age_hours: float = np.inf,
    satellite_ndre_age_hours: float = np.inf,
    satellite_ndvi_quality: Optional[float] = None,
    satellite_ndre_quality: Optional[float] = None,
    satellite_data_age_hours: float = np.inf,
    satellite_vv_db: float = np.nan,
    et0_today_mm: Optional[float] = None,
    et0_baseline_mm: Optional[float] = None,
    et0_std_mm: Optional[float] = None,
    index_name: str = "ndvi",
) -> Dict[str, object]:
    """Assess whether observed canopy development matches the crop stage.

    Tension arguments remain in the signature for API compatibility but are
    deliberately not used: optical canopy history is not evidence for the
    accuracy of a short-horizon soil-tension forecast.
    """
    del current_tension, stress_threshold_cbar, tension_forecast
    del satellite_data_age_hours, satellite_vv_db
    del et0_today_mm, et0_baseline_mm, et0_std_mm

    stage_name = STAGE_NAMES.get(growth_stage, str(growth_stage))
    history = satellite_history.copy() if isinstance(
        satellite_history, pd.DataFrame) else pd.DataFrame()
    if not history.empty and "timestamp" in history:
        history["timestamp"] = pd.to_datetime(
            history["timestamp"], utc=True, errors="coerce")
        history = history.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    elif "timestamp" not in history:
        history = pd.DataFrame()

    source = index_name
    value_column = f"sat_{source}"
    values = pd.Series(dtype=float)
    if not history.empty and value_column in history:
        values = pd.to_numeric(history[value_column], errors="coerce")
    values = values.where(values.between(-1.0, 1.0))

    points = []
    if not history.empty and not values.empty:
        for timestamp, value in zip(history["timestamp"], values):
            if pd.notna(value):
                points.append((timestamp, float(value)))
    if points:
        origin = points[0][0]
        trend_points = [
            ((timestamp - origin).total_seconds() / 86400.0, value)
            for timestamp, value in points
        ]
        trend_per_day = _fit_slope(trend_points)
    else:
        trend_per_day = None

    trend_threshold = 0.003
    if trend_per_day is None:
        observed_direction = "unknown"
    elif trend_per_day > trend_threshold:
        observed_direction = "increasing"
    elif trend_per_day < -trend_threshold:
        observed_direction = "decreasing"
    else:
        observed_direction = "stable"

    expected_direction = {
        STAGE_PRE_EMERGENCE: "increasing",
        STAGE_DEVELOPMENT: "increasing",
        STAGE_MID_SEASON: "stable",
        STAGE_LATE_SEASON: "decreasing",
        STAGE_POST_MATURITY: "stable",
    }.get(growth_stage, "unknown")
    sufficient_data = len(points) >= 3
    if not sufficient_data or observed_direction == "unknown":
        direction_agreement = 0.5
    elif observed_direction == expected_direction:
        direction_agreement = 0.8
    elif "stable" in (observed_direction, expected_direction):
        direction_agreement = 0.5
    else:
        direction_agreement = 0.2

    latest_value = (
        float(satellite_ndvi) if source == "ndvi" and np.isfinite(satellite_ndvi)
        else float(satellite_ndre) if source == "ndre" and np.isfinite(satellite_ndre)
        else points[-1][1] if points else np.nan
    )
    if not -1 <= latest_value <= 1:
        latest_value = np.nan
    kc_gdd = float(compute_kc_gdd(gdd_cumulative, crop_type))
    kc_eo = (
        float(compute_kc_ndvi(latest_value, crop_type))
        if source == "ndvi" and np.isfinite(latest_value) else np.nan
    )
    canopy_gap_kc = (
        float(kc_eo - kc_gdd) if np.isfinite(kc_eo) else np.nan)
    if source == "ndre" and sufficient_data:
        # Red-edge trend is evidence of canopy change, not an absolute Kc
        # calibration or a diagnosis of water stress.
        canopy_status = ("trend_consistent_with_stage"
                         if observed_direction == expected_direction
                         else "trend_differs_from_stage")
    elif not np.isfinite(canopy_gap_kc):
        canopy_status = "unknown"
    elif canopy_gap_kc < -0.15:
        canopy_status = "below_gdd_expectation"
    elif canopy_gap_kc > 0.15:
        canopy_status = "above_gdd_expectation"
    else:
        canopy_status = "consistent_with_gdd"

    source_history = (
        history.loc[values.notna()]
        if not history.empty and not values.empty else pd.DataFrame()
    )
    latest_row = (
        source_history.iloc[-1]
        if not source_history.empty else pd.Series(dtype=object)
    )
    if source == "ndvi":
        age_hours = float(satellite_ndvi_age_hours)
        supplied_quality = satellite_ndvi_quality
    else:
        age_hours = float(satellite_ndre_age_hours)
        supplied_quality = satellite_ndre_quality
    quality_column = f"sat_{source}_quality"
    history_quality = latest_row.get(quality_column, np.nan)
    quality_candidate = supplied_quality
    if quality_candidate is None or not np.isfinite(quality_candidate):
        quality_candidate = history_quality
    data_quality = _freshness_quality(
        age_hours, growth_stage, quality_candidate)
    overall_confidence = (
        float(direction_agreement * data_quality) if sufficient_data else None)

    if not sufficient_data:
        reason = (
            f"Insufficient {source.upper()} history for canopy trend assessment: "
            f"observations={len(points)}, required=3."
        )
    else:
        reason = (
            f"{stage_name} canopy is {observed_direction}; "
            f"the stage expectation is {expected_direction}."
        )

    return {
        "available": True,
        "assessment_type": "canopy_consistency",
        "tension_comparison_performed": False,
        "insufficient_data": not sufficient_data,
        "crop_type": crop_type,
        "growth_stage": stage_name,
        "growth_stage_code": int(growth_stage),
        # Retained for clients on schema 1.0; it now means canopy-stage
        # direction agreement, not agreement with the tension forecast.
        "direction_agreement": round(float(direction_agreement), 3),
        "validation_score": round(float(direction_agreement), 3),
        "data_quality": round(float(data_quality), 3),
        "overall_confidence": (
            None if overall_confidence is None
            else round(float(overall_confidence), 3)
        ),
        "canopy_status": canopy_status,
        "factors": {
            "satellite_source": source,
            "satellite_value": (
                round(float(latest_value), 4) if np.isfinite(latest_value) else None),
            "satellite_points": len(points),
            "trend_per_day": (
                round(float(trend_per_day), 5)
                if trend_per_day is not None else None),
            "observed_direction": observed_direction,
            "expected_direction": expected_direction,
            "kc_gdd": round(kc_gdd, 4),
            "kc_eo": round(float(kc_eo), 4) if np.isfinite(kc_eo) else None,
            "canopy_gap_kc": (
                round(float(canopy_gap_kc), 4)
                if np.isfinite(canopy_gap_kc) else None),
            "latest_satellite_age_hours": (
                round(age_hours, 1) if np.isfinite(age_hours) else None),
            "data_quality": round(float(data_quality), 3),
        },
        "reason": reason,
    }


def check_canopy_consistency(*args, **kwargs) -> Dict[str, object]:
    """Assess each index independently; never substitute NDRE values for NDVI."""
    indices = {name: _check_canopy_index(*args, **kwargs, index_name=name)
               for name in ("ndvi", "ndre")}
    usable = [result for result in indices.values()
              if not result["insufficient_data"]]
    primary = (usable[0] if usable else indices["ndvi"])
    result = {**primary, "factors": dict(primary["factors"]), "indices": indices}
    if len(usable) == 2:
        result["factors"]["satellite_source"] = "ndvi+ndre"
        for key in ("direction_agreement", "validation_score", "data_quality"):
            result[key] = round(float(np.mean([item[key] for item in usable])), 3)
        result["overall_confidence"] = round(float(np.mean([
            item["overall_confidence"] for item in usable])), 3)
        result["reason"] = " ".join(
            f"{name.upper()}: {item['factors']['observed_direction']}."
            for name, item in indices.items())
        if (indices["ndvi"]["factors"]["observed_direction"] !=
                indices["ndre"]["factors"]["observed_direction"]):
            result["canopy_status"] = "mixed_index_trends"
    return result


def check_satellite_tension_consistency(*args, **kwargs) -> Dict[str, object]:
    """Backward-compatible alias for the non-causal canopy assessment."""
    return check_canopy_consistency(*args, **kwargs)


# BLENDED DYNAMIC Kc
def _compute_single_index_kc_diagnostics(
    cumulative_gdd: float,
    crop_type: str,
    ndvi: float = np.nan,
    ndvi_age_hours: float = np.inf,
    ndre: float = np.nan,
    ndre_age_hours: float = np.inf,
    ndvi_quality: Optional[float] = None,
    ndre_quality: Optional[float] = None,
    allow_experimental_ndre: bool = False,
) -> Dict[str, object]:
    """Return the Kc result and every EO factor that influenced it."""
    kc_gdd = float(compute_kc_gdd(cumulative_gdd, crop_type))
    stage = get_growth_stage(cumulative_gdd, crop_type)
    ndvi_usable = bool(np.isfinite(ndvi))
    ndre_usable = bool(
        allow_experimental_ndre
        and np.isfinite(ndre)
        and stage == STAGE_MID_SEASON
    )
    if not ndvi_usable and not ndre_usable:
        return {
            "kc_dynamic": kc_gdd, "kc_gdd": kc_gdd, "kc_eo": None,
            "eo_source": "none", "eo_weight": 0.0, "quality_factor": 0.0,
            "plausibility_factor": 0.0, "eo_kc_delta": 0.0,
            "experimental_ndre_enabled": bool(allow_experimental_ndre),
            "reason": "No usable EO observation; GDD-only Kc.",
        }

    kc_sat, satellite_source = _select_satellite_kc(
        cumulative_gdd, ndvi, ndre, crop_type, allow_experimental_ndre)
    if np.isnan(kc_sat):
        return {
            "kc_dynamic": kc_gdd, "kc_gdd": kc_gdd, "kc_eo": None,
            "eo_source": "none", "eo_weight": 0.0, "quality_factor": 0.0,
            "plausibility_factor": 0.0, "eo_kc_delta": 0.0,
            "experimental_ndre_enabled": bool(allow_experimental_ndre),
            "reason": "EO index could not produce a valid Kc; GDD-only Kc.",
        }

    if satellite_source == "ndre" and ndvi_usable:
        ndre_freshness = _freshness_quality(
            ndre_age_hours, stage, ndre_quality)
        ndvi_freshness = _freshness_quality(
            ndvi_age_hours, stage, ndvi_quality)
        if ndre_freshness <= 0.0 < ndvi_freshness:
            kc_sat = float(compute_kc_ndvi(ndvi, crop_type))
            satellite_source = "ndvi"

    ndvi_for_weight = (
        float(ndvi) if ndvi_usable
        else float(np.clip(ndre * NDRE_NDVI_PROXY, 0.0, 1.0))
    )
    base_weight = float(np.clip(
        ndvi_for_weight * NDVI_WEIGHT_SLOPE, 0.0, NDVI_WEIGHT_CAP))
    selected_age = (
        ndre_age_hours if satellite_source == "ndre" else ndvi_age_hours)
    selected_quality = (
        ndre_quality if satellite_source == "ndre" else ndvi_quality)
    quality_factor = _freshness_quality(
        selected_age, stage, selected_quality)
    plausibility = _gdd_plausibility_weight(
        float(kc_sat), cumulative_gdd, crop_type)
    weight = float(base_weight * quality_factor * plausibility)
    kc_dynamic = float((1.0 - weight) * kc_gdd + weight * kc_sat)
    return {
        "kc_dynamic": kc_dynamic,
        "kc_gdd": kc_gdd,
        "kc_eo": float(kc_sat),
        "eo_source": satellite_source,
        "eo_weight": weight,
        "quality_factor": float(quality_factor),
        "plausibility_factor": float(plausibility),
        "eo_kc_delta": float(kc_dynamic - kc_gdd),
        "experimental_ndre_enabled": bool(allow_experimental_ndre),
        "reason": (
            f"{satellite_source.upper()} adjusted the GDD Kc with "
            f"{weight:.1%} effective weight."
            if weight > 0 else
            "EO observation was rejected by quality/plausibility gating; GDD-only Kc."
        ),
    }


def compute_kc_dynamic_diagnostics(
    cumulative_gdd: float, crop_type: str,
    ndvi: float = np.nan, ndvi_age_hours: float = np.inf,
    ndre: float = np.nan, ndre_age_hours: float = np.inf,
    ndvi_quality: Optional[float] = None,
    ndre_quality: Optional[float] = None,
    allow_experimental_ndre: bool = False,
) -> Dict[str, object]:
    """Fuse independent estimates without counting correlated indices twice.

    NDRE availability is independent of permission to use its experimental
    calibration. A missing/invalid age must never borrow the other index's age.
    """
    components = {}
    for name, value, age, quality in (
        ("ndvi", ndvi, ndvi_age_hours, ndvi_quality),
        ("ndre", ndre, ndre_age_hours, ndre_quality),
    ):
        valid = bool(np.isfinite(value) and -1 <= value <= 1)
        item = _compute_single_index_kc_diagnostics(
            cumulative_gdd, crop_type,
            **{name: float(value) if valid else np.nan,
               f"{name}_age_hours": age if np.isfinite(age) and age >= 0 else np.inf,
               f"{name}_quality": quality},
            allow_experimental_ndre=allow_experimental_ndre,
        )
        item["observation_available"] = valid
        if name == "ndre" and valid and not allow_experimental_ndre:
            item["reason"] = "NDRE observed; Kc calibration not enabled."
        components[name] = item
    active = [item for item in components.values() if item["eo_weight"] > 0]
    result = dict(active[0] if active else components["ndvi"])
    if active:
        total = sum(item["eo_weight"] for item in active)
        weight = max(item["eo_weight"] for item in active)
        kc_eo = sum(item["kc_eo"] * item["eo_weight"] for item in active) / total
        delta = weight * (kc_eo - result["kc_gdd"])
        result.update(kc_eo=kc_eo, eo_weight=weight, eo_kc_delta=delta,
                      kc_dynamic=result["kc_gdd"] + delta,
                      eo_source="+".join(item["eo_source"] for item in active))
        result["reason"] = f"{result['eo_source'].upper()} adjusted GDD Kc with {weight:.1%} weight."
    elif components["ndre"]["observation_available"] and not allow_experimental_ndre:
        result["reason"] = "GDD-only Kc; NDRE observed but Kc calibration not enabled."
    result["indices"] = components
    return result


def compute_kc_dynamic(
    cumulative_gdd: float,
    crop_type: str,
    ndvi: float = np.nan,
    ndvi_age_hours: float = np.inf,
    ndre: float = np.nan,
    ndre_age_hours: float = np.inf,
    ndvi_quality: Optional[float] = None,
    ndre_quality: Optional[float] = None,
    allow_experimental_ndre: bool = False,
) -> float:
    """
    Compute the blended dynamic crop coefficient.

        Kc_dynamic = (1 - w) * Kc_GDD + w * Kc_satellite

    Where:
        Kc_satellite  is NDVI by default. Under the experimental flag, a
                      quality-weighted NDVI/NDRE estimate is used at mid-season.
        w             is a quality weight combining three factors:
                          signal strength  -- scales with NDVI magnitude
                                             (more canopy -> more confident)
                          freshness       -- empirical provider cadence score
                          plausibility     -- zeros w when satellite implies
                                             a stage GDD says is impossible

    Falls back entirely to GDD-based Kc when satellite data is absent
    or biologically inconsistent with thermal development.

    Satellite influence follows the actual acquisition intervals reported by
    the provider. When cadence quality is absent, only a current observation
    is trusted and the calculation otherwise falls back to GDD.

    Args:
        cumulative_gdd:  Current accumulated GDD
        ndvi:            Latest NDVI value (NaN if unavailable)
        ndvi_age_hours:  Hours since the last real NDVI observation
        crop_type:       Key into CROP_PARAMS
        ndre:            Latest NDRE value (NaN if unavailable).
                         Used for Kc only at mid-season when experimental use
                         is explicitly enabled.
        ndre_age_hours:  Hours since the NDRE observation.
        ndvi_quality:    NDVI freshness score derived from observed cadence.
        ndre_quality:    NDRE freshness score derived from observed cadence.
        allow_experimental_ndre: Enable the unvalidated NDRE-to-Kc mapping.

    Returns:
        Kc_dynamic in range [Kc_ini, ~1.35].
    """
    return float(compute_kc_dynamic_diagnostics(
        cumulative_gdd=cumulative_gdd,
        crop_type=crop_type,
        ndvi=ndvi,
        ndvi_age_hours=ndvi_age_hours,
        ndre=ndre,
        ndre_age_hours=ndre_age_hours,
        ndvi_quality=ndvi_quality,
        ndre_quality=ndre_quality,
        allow_experimental_ndre=allow_experimental_ndre,
    )["kc_dynamic"])


def compute_kc_dynamic_series(
    gdd_series: pd.Series,
    ndvi_series: pd.Series,
    age_series: pd.Series,
    crop_type: str,
    ndre_series: Optional[pd.Series] = None,
    ndvi_quality_series: Optional[pd.Series] = None,
    ndre_age_series: Optional[pd.Series] = None,
    ndre_quality_series: Optional[pd.Series] = None,
    allow_experimental_ndre: bool = False,
) -> pd.Series:
    """Use the same per-index quality and fusion rules as the runtime path."""
    def values(series, default):
        if series is None:
            return [default] * len(gdd_series)
        return series.reindex(gdd_series.index).fillna(default).to_numpy(dtype=float)

    rows = zip(
        gdd_series.to_numpy(dtype=float),
        values(ndvi_series, np.nan), values(age_series, np.inf),
        values(ndre_series, np.nan), values(ndre_age_series, np.inf),
        values(ndvi_quality_series, np.nan), values(ndre_quality_series, np.nan),
    )
    return pd.Series([
        compute_kc_dynamic(
            gdd, crop_type, ndvi, ndvi_age, ndre, ndre_age,
            ndvi_quality, ndre_quality, allow_experimental_ndre,
        )
        for gdd, ndvi, ndvi_age, ndre, ndre_age, ndvi_quality, ndre_quality in rows
    ], index=gdd_series.index)

# CLI


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Phenology engine: GDD, growth stages, dynamic Kc and stage-aware tension triggers"
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
