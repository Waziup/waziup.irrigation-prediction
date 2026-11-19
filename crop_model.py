import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from crops import (
    STAGE_NAMES,
    STAGE_PRE_EMERGENCE,
    get_crop_params,
)
from farm_config import FarmConfig
from weather_quality import deduplicate_weather_frame, complete_daily_et0
from phenology_engine import (
    compute_gdd_series,
    daily_temperature_extrema,
    compute_kc_gdd,
    compute_kc_dynamic_diagnostics,
    compute_kc_gdd_series,
    compute_kc_ndvi,
    compute_kc_ndre,
    compute_kc_dynamic_series,
    get_growth_stage,
    get_growth_stage_series,
    phenology_season_start,
    sar_adjusted_ndvi_age,
    KC_UPPER_TOLERANCE,
)

log = logging.getLogger(__name__)


@dataclass
class CropState:

    growth_stage: int
    growth_stage_name: str
    stress_threshold_cbar: float    # cbar — irrigate when tension reaches/exceeds this
    kc: float                       # blended Kc_dynamic
    gdd_cumulative: float
    crop_type: str
    # crop water demand (mm/day); 0.0 when ET0 unavailable
    etc_daily_mm: float = 0.0
    satellite_validation: Optional[dict] = None
    et0_today_mm: Optional[float] = None
    et0_baseline_mm: Optional[float] = None
    et0_std_mm: Optional[float] = None
    # Per-index source age/quality are kept separate because NDRE cadence can
    # differ materially from NDVI cadence in the EO catalog.
    sat_ndvi_quality: float = 0.0
    sat_ndre_quality: float = 0.0
    sat_ndre_age_hours: float = float("inf")
    kc_gdd: float = 0.0
    kc_eo: Optional[float] = None
    eo_source: str = "none"
    eo_weight: float = 0.0
    eo_quality_factor: float = 0.0
    eo_plausibility_factor: float = 0.0
    eo_kc_delta: float = 0.0
    eo_etc_delta_mm: float = 0.0
    eo_reason: str = ""
    eo_indices: Optional[dict] = None
    threshold_mode: str = "static"
    threshold_reason: str = ""
    threshold_details: Optional[dict] = None
    season_active: bool = True
    season_status: str = "active"
    season_end_reason: Optional[str] = None


_STAGE_THRESHOLD_KEYS = {
    0: "pre_emergence",
    1: "development",
    2: "mid_season",
    3: "late_season",
    4: "post_maturity",
}


def _positive_finite(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return result


def _nonnegative_finite(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite value") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative finite value")
    return result


def _retention_curve_points(raw_curve) -> tuple[np.ndarray, np.ndarray]:
    """Validate field tension/VWC observations for monotonic interpolation."""
    points = []
    for item in raw_curve or []:
        try:
            if isinstance(item, dict):
                tension = float(item.get("Soil tension", item.get("tension_cbar")))
                vwc = float(item.get("VWC", item.get("vwc")))
            else:
                tension, vwc = map(float, item[:2])
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise ValueError("Every retention point must contain numeric cbar and VWC") from exc
        if np.isfinite(tension) and tension >= 0 and np.isfinite(vwc) and 0 < vwc < 1:
            points.append((tension, vwc))
        else:
            raise ValueError("Retention points require finite nonnegative cbar and VWC fractions between 0 and 1")
    points = sorted(points)
    if len(points) < 3:
        raise ValueError(
            "dynamic mode requires at least three valid field retention-curve points")
    tensions = np.asarray([point[0] for point in points], dtype=float)
    vwcs = np.asarray([point[1] for point in points], dtype=float)
    if np.any(np.diff(tensions) <= 0) or np.any(np.diff(vwcs) >= 0):
        raise ValueError(
            "soil-water retention VWC must decrease as tension increases")
    return tensions, vwcs


def _tension_for_vwc(target_vwc: float, raw_curve) -> float:
    tensions, vwcs = _retention_curve_points(raw_curve)
    if target_vwc < vwcs[-1] or target_vwc > vwcs[0]:
        raise ValueError(
            "the retention curve does not bracket the dynamic target water content")
    # Matric tension spans orders of magnitude. Interpolate log(1+tension)
    # against water content, avoiding the oscillation/overshoot of a cubic fit.
    log_tension = np.log1p(tensions)
    return float(np.expm1(np.interp(
        target_vwc, vwcs[::-1], log_tension[::-1])))


def _vwc_for_tension(target_tension: float, raw_curve) -> float:
    """Interpolate VWC at a configured tension on the field soil curve."""
    tensions, vwcs = _retention_curve_points(raw_curve)
    tension = _nonnegative_finite(target_tension, "soil tension boundary")
    if tension < tensions[0] or tension > tensions[-1]:
        raise ValueError(
            "the retention curve does not bracket the configured soil-tension "
            "boundaries")
    return float(np.interp(np.log1p(tension), np.log1p(tensions), vwcs))


def _stage_calibrated_threshold(farm, gdd_cumulative: float) -> Optional[float]:
    raw = getattr(farm, "stage_thresholds_cbar", None)
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        values = {
            key: _positive_finite(raw[key], f"stage_thresholds_cbar.{key}")
            for key in _STAGE_THRESHOLD_KEYS.values()
        }
    except KeyError as exc:
        raise ValueError(
            "stage_thresholds_cbar must calibrate all five crop stages") from exc
    params = get_crop_params(farm.crop_type)
    anchors = np.asarray([
        0.0, params.gdd_emergence, params.gdd_dev_end,
        params.gdd_mid_end, params.gdd_maturity,
    ], dtype=float)
    thresholds = np.asarray([
        values["pre_emergence"], values["development"],
        values["mid_season"], values["late_season"],
        values["post_maturity"],
    ], dtype=float)
    return float(np.interp(float(gdd_cumulative), anchors, thresholds))


def _stage_calibrated_depletion_fraction(
    farm,
    gdd_cumulative: float,
) -> Optional[float]:
    """Smoothly interpolate local p values so stage boundaries cannot jump."""
    raw = getattr(farm, "stage_depletion_fractions", None)
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        values = {
            key: float(raw[key]) for key in _STAGE_THRESHOLD_KEYS.values()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "stage_depletion_fractions must calibrate all five crop stages"
        ) from exc
    invalid = [
        key for key, value in values.items()
        if not np.isfinite(value) or not 0 < value < 1
    ]
    if invalid:
        raise ValueError(
            "stage depletion fractions must be between 0 and 1: "
            + ", ".join(sorted(invalid)))
    params = get_crop_params(farm.crop_type)
    anchors = np.asarray([
        0.0, params.gdd_emergence, params.gdd_dev_end,
        params.gdd_mid_end, params.gdd_maturity,
    ], dtype=float)
    fractions = np.asarray([
        values["pre_emergence"], values["development"],
        values["mid_season"], values["late_season"],
        values["post_maturity"],
    ], dtype=float)
    return float(np.interp(float(gdd_cumulative), anchors, fractions))


def _depletion_policy(farm, gdd, etc):
    """FAO-56 Table 22 and its daily ETc adjustment (Chapter 8).

    Stage affects ETc through the crop Kc curve. No universal phase multipliers
    or irrigation-method caps are applied. Explicit site calibrations remain
    supported separately from the published default.
    """
    params = get_crop_params(farm.crop_type)
    configured = getattr(farm, "depletion_fraction", None)
    stage_p = _stage_calibrated_depletion_fraction(farm, gdd)
    reference = float(stage_p if stage_p is not None else (
        params.depletion_fraction if configured in (None, "") else configured))
    if not np.isfinite(reference) or not 0 < reference < 1:
        raise ValueError("depletion_fraction must be between 0 and 1")
    calibrated = stage_p is not None or configured not in (None, "")
    rice_default = farm.crop_type == "rice" and not calibrated
    # Revised FAO-56 Table 8.2 gives flooded rice an explicit 0.80 theta_sat
    # threshold, rather than an ETc-adjusted fraction of FC-to-WP water.
    adjusted = reference if rice_default else float(
        np.clip(reference + 0.04 * (5.0 - etc), 0.1, 0.8))
    policy_name = ("calibrated_stage" if stage_p is not None else
                   "calibrated_crop" if calibrated else
                   "rice_saturation_estimate" if rice_default else "fao56_table22")
    reference_source = ("site_configuration" if calibrated else
                        "FAO-56 Rev.1 (2025), Table 8.2" if rice_default
                        else "FAO-56 Table 22")
    return reference, adjusted, {
        "depletion_policy": policy_name,
        "depletion_reference_source": reference_source,
        "depletion_adjustment_source": (
            "Fixed saturation-relative rice threshold" if rice_default else
            "FAO-56 Chapter 8, Table 22 footnote 2"),
        "reference_url": (
            "https://agrhysmo.agr.unipi.it/wp-content/uploads/2025/09/FAO56%202025.pdf"
            if rice_default else "https://www.fao.org/4/x0490e/x0490e0e.htm"),
        "stage_policy": "field_calibrated" if stage_p is not None else "stage_kc_and_daily_etc",
        "estimated": not calibrated or (getattr(farm, "soil_calibration", {}) or {}).get("source", "estimate") == "estimate",
    }


def _rice_saturation_threshold_details(farm, gdd, etc_daily_mm):
    """Estimated rice soil-tension policy, not a ponded-water balance.

    FAO-56 Rev.1 (2025), Table 8.2 specifies 0.80 theta_sat for flooded rice.
    The application maps that water-content target to the saved soil curve.
    No general FC-to-WP or daily ETc p adjustment is applied. Original curves
    omit the saturation endpoint; their wettest point is an explicit proxy.
    No water-table depth is converted into cbar and no static value is used.
    """
    curve = getattr(farm, "soil_water_retention_curve", None)
    tensions, vwcs = _retention_curve_points(curve)
    saturation = _nonnegative_finite(getattr(farm, "saturation", 0), "saturation")
    saturation_source = "soil_retention_curve"
    if saturation < tensions[0]:
        from soil_calibration import is_original_soil_preset
        if not is_original_soil_preset(
                list(zip(tensions.tolist(), vwcs.tolist())),
                getattr(farm, "field_capacity_lower", None),
                getattr(farm, "permanent_wilting_point", None), saturation,
                getattr(farm, "soil_calibration", {}) or {}):
            raise ValueError(
                "Rice dynamic threshold needs a soil retention curve covering "
                "saturation. Select a soil-type preset or correct the saved curve.")
        theta_sat = float(vwcs[0])
        saturation_source = "original_preset_wettest_point_proxy"
    else:
        theta_sat = _vwc_for_tension(saturation, curve)
    etc = (None if etc_daily_mm is None else
           _nonnegative_finite(etc_daily_mm, "etc_daily_mm"))
    reference, fraction, policy = _depletion_policy(farm, gdd, etc)
    unconstrained_target = theta_sat * (1.0 - fraction)
    target = float(np.clip(unconstrained_target, vwcs[-1], theta_sat))
    depletion_threshold = _tension_for_vwc(target, curve)
    # IRRI's safe-AWD guidance associates the 15-cm field-water level with
    # root-zone tension below 10 kPa. Use 10 cbar as a conservative application
    # ceiling, not a conversion between ponding depth and soil tension.
    # IRRI technical leaflet (hosted by TNAU):
    # https://www.agritech.tnau.ac.in/agriculture/pdf/csa_pdf/Smart_water_technique_for_rice.pdf
    management_limit = 10.0
    threshold = min(depletion_threshold, management_limit)
    return {
        **policy,
        "source": "rice_saturation_soil_curve",
        "threshold_cbar": threshold,
        "depletion_threshold_cbar": depletion_threshold,
        "rice_management_limit_cbar": management_limit,
        "rice_management_reference_url": "https://www.agritech.tnau.ac.in/agriculture/pdf/csa_pdf/Smart_water_technique_for_rice.pdf",
        "rice_management_limited": threshold < depletion_threshold,
        "growth_stage": STAGE_NAMES[get_growth_stage(gdd, farm.crop_type)],
        "etc_daily_mm": etc,
        "depletion_fraction_reference": reference,
        "depletion_fraction_adjusted": fraction,
        "depletion_reference_basis": "saturation",
        "saturation_vwc": theta_sat,
        "saturation_vwc_source": saturation_source,
        "saturation_depletion_target_vwc": target,
        "target_vwc": (_vwc_for_tension(threshold, curve)
                       if tensions[0] <= threshold <= tensions[-1] else None),
        "curve_limited": target != unconstrained_target,
        "estimated": True,
    }


def _get_depletion_threshold_details(
    farm,
    gdd_cumulative: float,
    etc_daily_mm: Optional[float],
) -> dict:
    """Calculate a field-specific trigger from FAO-56 TAW/RAW concepts.

    The normal UI path uses the already configured retention curve and its
    field-capacity/wilting-point tension boundaries. Optional measured VWC and
    root-depth inputs remain supported for externally managed configurations.
    Universal additive cbar offsets are never used because matric-potential
    response is nonlinear and soil-specific.
    """
    stage_threshold = _stage_calibrated_threshold(farm, gdd_cumulative)
    if stage_threshold is not None:
        return {
            "threshold_cbar": stage_threshold,
            "source": "farmer_calibrated_stage_curve",
            "growth_stage": STAGE_NAMES[get_growth_stage(gdd_cumulative, farm.crop_type)],
            "estimated": False,
        }
    if (farm.crop_type == "rice"
            and getattr(farm, "depletion_fraction", None) in (None, "")
            and not getattr(farm, "stage_depletion_fractions", None)):
        return _rice_saturation_threshold_details(farm, gdd_cumulative, etc_daily_mm)
    hydraulic_names = (
        "field_capacity_vwc", "wilting_point_vwc", "root_depth_m",
        "sensor_depth_m", "soil_water_retention_curve",
    )
    hydraulic_available = all(
        getattr(farm, name, None) not in (None, "", [])
        for name in hydraulic_names
    )
    if not hydraulic_available:
        curve = getattr(farm, "soil_water_retention_curve", None)
        fc_tension = getattr(farm, "field_capacity_lower", None)
        wp_tension = getattr(farm, "permanent_wilting_point", None)
        existing_soil_inputs = (
            curve not in (None, "", [])
            and fc_tension not in (None, "")
            and wp_tension not in (None, "")
        )
        if existing_soil_inputs:
            etc = _nonnegative_finite(etc_daily_mm, "etc_daily_mm")
            wp_tension = _positive_finite(
                wp_tension, "permanent_wilting_point")
            from soil_calibration import validate_soil_calibration
            validate_soil_calibration(curve, fc_tension, wp_tension,
                getattr(farm, "saturation", 0), getattr(farm, "soil_calibration", {}))
            p_reference, depletion_fraction, policy = _depletion_policy(farm, gdd_cumulative, etc)
            theta_fc = _vwc_for_tension(fc_tension, curve)
            theta_wp = _vwc_for_tension(wp_tension, curve)
            if theta_fc <= theta_wp:
                raise ValueError(
                    "field-capacity VWC must exceed wilting-point VWC on the "
                    "configured retention curve")
            target_vwc = theta_fc - depletion_fraction * (
                theta_fc - theta_wp)
            threshold = _tension_for_vwc(target_vwc, curve)
            stage = get_growth_stage(gdd_cumulative, farm.crop_type)
            return {
                **policy,
                "threshold_cbar": threshold,
                "source": "existing_soil_curve_crop_demand",
                "growth_stage": STAGE_NAMES[stage],
                "etc_daily_mm": etc,
                "depletion_fraction_reference": p_reference,
                "depletion_fraction_adjusted": depletion_fraction,
                "field_capacity_tension_cbar": float(fc_tension),
                "wilting_point_tension_cbar": float(wp_tension),
                "field_capacity_vwc": theta_fc,
                "wilting_point_vwc": theta_wp,
                "target_vwc": target_vwc,
            }
        raise ValueError(
            "dynamic mode requires the configured soil-retention curve and "
            "field-capacity/wilting-point boundaries, or five farmer-calibrated "
            "stage thresholds")

    theta_fc = _positive_finite(
        getattr(farm, "field_capacity_vwc"), "field_capacity_vwc")
    theta_wp = _positive_finite(
        getattr(farm, "wilting_point_vwc"), "wilting_point_vwc")
    if theta_fc >= 1 or theta_wp >= 1 or theta_fc <= theta_wp:
        raise ValueError(
            "field_capacity_vwc and wilting_point_vwc must be fractions with "
            "0 < wilting point < field capacity < 1")
    root_depth = _positive_finite(getattr(farm, "root_depth_m"), "root_depth_m")
    sensor_depth = _positive_finite(
        getattr(farm, "sensor_depth_m"), "sensor_depth_m")
    if sensor_depth > root_depth:
        raise ValueError(
            "sensor_depth_m cannot exceed the effective root_depth_m")
    etc = _nonnegative_finite(etc_daily_mm, "etc_daily_mm")

    stage = get_growth_stage(gdd_cumulative, farm.crop_type)
    p_reference, depletion_fraction, policy = _depletion_policy(farm, gdd_cumulative, etc)
    taw_mm = 1000.0 * (theta_fc - theta_wp) * root_depth
    raw_mm = depletion_fraction * taw_mm
    target_vwc = theta_fc - raw_mm / (1000.0 * root_depth)
    _, curve_vwc = _retention_curve_points(
        getattr(farm, "soil_water_retention_curve"))
    if theta_fc > curve_vwc[0] or theta_wp < curve_vwc[-1]:
        raise ValueError(
            "the field retention curve must bracket both field-capacity and "
            "wilting-point VWC")
    threshold = _tension_for_vwc(
        target_vwc, getattr(farm, "soil_water_retention_curve"))
    return {
        **policy,
        "threshold_cbar": threshold,
        "source": "fao56_root_zone_depletion",
        "growth_stage": STAGE_NAMES[stage],
        "etc_daily_mm": etc,
        "depletion_fraction_reference": p_reference,
        "depletion_fraction_adjusted": depletion_fraction,
        "field_capacity_vwc": theta_fc,
        "wilting_point_vwc": theta_wp,
        "target_vwc": target_vwc,
        "root_depth_m": root_depth,
        "sensor_depth_m": sensor_depth,
        "taw_mm": taw_mm,
        "raw_mm": raw_mm,
    }


def get_dynamic_threshold_details(farm, gdd_cumulative, etc_daily_mm):
    """Separate estimated point-sensor management from root-zone RAW.

    OSU EM 9868 Table 1 supplies approximate texture management bands, not
    stage-specific prescriptions. Mapping the FAO demand-adjusted p into that
    band is a conservative application policy, not a published FAO equation.
    Explicit site depletion/tension targets and measured curves retain their
    existing calculation. Never raise an already wetter depletion trigger.
    """
    details = _get_depletion_threshold_details(farm, gdd_cumulative, etc_daily_mm)
    calibration = getattr(farm, "soil_calibration", {}) or {}
    if (details["source"] == "farmer_calibrated_stage_curve"
            or calibration.get("source", "estimate") != "estimate"
            or details.get("depletion_policy") != "fao56_table22"):
        return details
    # Class IDs match the settings soilTextureClasses mapping, including silt=9.
    bands = {0: (20., 30.), 1: (20., 30.), 2: (20., 30.),
             3: (30., 50.), 4: (30., 50.), 9: (30., 50.),
             7: (50., 60.), 11: (50., 60.)}
    band = bands.get(getattr(farm, "soil_texture_class", None))
    if band is None:
        # Do not assign an unsupported texture-specific prescription.
        return details
    low, high = band
    p = details["depletion_fraction_adjusted"]
    management_limit = low + (high - low) * (p - 0.1) / 0.7
    candidate = details["threshold_cbar"]
    threshold = min(candidate, management_limit)
    target = _vwc_for_tension(threshold, farm.soil_water_retention_curve)
    return {
        **details,
        "threshold_cbar": threshold,
        "depletion_threshold_cbar": candidate,
        "management_policy": "estimated_texture_demand_band",
        "management_reference_url": "https://extension.oregonstate.edu/catalog/em-9868-soil-moisture-monitoring-support-irrigation-scheduling",
        "management_band_cbar": [low, high],
        "management_limit_cbar": management_limit,
        "management_target_vwc": target,
        "management_depletion_fraction": (
            (details["field_capacity_vwc"] - target)
            / (details["field_capacity_vwc"] - details["wilting_point_vwc"])),
        "estimated": True,
    }


def _configured_threshold(
    farm,
    gdd_cumulative: float,
    etc_daily_mm: Optional[float] = None,
) -> tuple[float, str, str, dict]:
    """Resolve the selected tension-trigger mode from a calibrated baseline."""
    mode = str(getattr(farm, "threshold_mode", "") or "").strip().lower()
    if not mode:
        mode = "dynamic" if bool(getattr(farm, "use_dynamic_threshold", False)) else "static"
    raw = getattr(farm, "static_threshold_cbar", None)
    if raw is None:
        raw = getattr(farm, "threshold_static", None)
    if mode != "dynamic":
        try:
            baseline = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("A field-calibrated static_threshold_cbar is required") from exc
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("static_threshold_cbar must be a positive finite value")

    if mode not in {"static", "dynamic"}:
        raise ValueError("threshold_mode must be 'static' or 'dynamic'")
    if mode == "static":
        return (baseline, "static",
                "Using the field-calibrated threshold unchanged.",
                {"source": "field_static", "threshold_cbar": baseline})

    details = get_dynamic_threshold_details(farm, gdd_cumulative, etc_daily_mm)
    hysteresis = _nonnegative_finite(
        getattr(farm, "threshold_hysteresis_cbar", 0.0) or 0.0,
        "threshold_hysteresis_cbar",
    )
    if hysteresis > 0:
        details["recovery_hysteresis_cbar"] = hysteresis
    threshold = float(details["threshold_cbar"])
    stage = details["growth_stage"]
    if details["source"] == "rice_saturation_soil_curve":
        reason = f"{stage}: estimated rice trigger from saturation-relative depletion and the soil retention curve."
    elif details["source"] in {
        "fao56_root_zone_depletion", "existing_soil_curve_crop_demand"
    }:
        reason = (
            f"{stage}: {'Estimated management' if details.get('estimated') else 'Calibrated'} depletion target using p="
            f"{details['depletion_fraction_adjusted']:.2f}, target VWC "
            f"{details['target_vwc']:.3f}, and the field retention curve."
        )
    else:
        reason = f"{stage}: interpolated farmer-calibrated stage trigger."
    if details.get("management_policy"):
        reason = (
            f"{stage}: estimated soil-texture management trigger adjusted for "
            "crop demand; local validation required. Root-zone depletion "
            f"alone would give {details['depletion_threshold_cbar']:.1f} cbar."
        )
    return (
        threshold,
        "dynamic",
        reason,
        details,
    )


def get_crop_season_status(
    farm,
    gdd_cumulative: float,
    as_of=None,
) -> dict:
    """Resolve whether irrigation advice belongs to an active crop season."""
    params = get_crop_params(farm.crop_type)
    timestamp = pd.Timestamp(as_of) if as_of is not None else None
    planting_raw = getattr(farm, "planting_date", None)
    planting = pd.Timestamp(planting_raw) if planting_raw else None
    if timestamp is not None and planting is not None:
        if timestamp.tzinfo is not None and planting.tzinfo is None:
            planting = planting.tz_localize(timestamp.tzinfo)
        elif timestamp.tzinfo is None and planting.tzinfo is not None:
            planting = planting.tz_localize(None)
        elif timestamp.tzinfo is not None and planting.tzinfo is not None:
            planting = planting.tz_convert(timestamp.tzinfo)
        if timestamp < planting:
            return {
                "active": False,
                "status": "not_started",
                "reason": "before_planting_date",
            }

    harvest_raw = getattr(farm, "harvest_date", None)
    if timestamp is not None and harvest_raw:
        harvest = pd.Timestamp(harvest_raw)
        if timestamp.tzinfo is not None and harvest.tzinfo is None:
            harvest = harvest.tz_localize(timestamp.tzinfo)
        elif timestamp.tzinfo is None and harvest.tzinfo is not None:
            harvest = harvest.tz_localize(None)
        elif timestamp.tzinfo is not None and harvest.tzinfo is not None:
            harvest = harvest.tz_convert(timestamp.tzinfo)
        if params.is_perennial and planting is not None:
            season_start = phenology_season_start(
                timestamp, planting, farm.crop_type)
            try:
                harvest = harvest.replace(year=season_start.year)
            except ValueError:
                harvest = harvest.replace(year=season_start.year, day=28)
            if harvest.normalize() <= season_start:
                try:
                    harvest = harvest.replace(year=season_start.year + 1)
                except ValueError:
                    harvest = harvest.replace(
                        year=season_start.year + 1, day=28)
        if timestamp.normalize() >= harvest.normalize():
            return {
                "active": False,
                "status": "harvested",
                "reason": "harvest_date_reached",
            }

    if (not params.is_perennial and not params.continuous_leaf_harvest
            and float(gdd_cumulative) >= params.gdd_maturity):
        return {
            "active": False,
            "status": "mature",
            "reason": "physiological_maturity_reached",
        }
    return {"active": True, "status": "active", "reason": None}


def get_crop_state(
    farm: FarmConfig,
    gdd_cumulative: float,
    ndvi: float = float("nan"),
    ndre: float = float("nan"),
    ndvi_age_hours: float = float("inf"),
    ndre_age_hours: float = float("inf"),
    ndvi_quality: Optional[float] = None,
    ndre_quality: Optional[float] = None,
    allow_experimental_ndre: bool = False,
    etc_daily_mm: Optional[float] = None,
    et0_daily_mm: Optional[float] = None,
    as_of=None,
) -> CropState:
    crop_type = farm.crop_type
    stage = get_growth_stage(gdd_cumulative, crop_type)
    kc_diagnostics = compute_kc_dynamic_diagnostics(
        cumulative_gdd=gdd_cumulative,
        ndvi=ndvi,
        ndvi_age_hours=ndvi_age_hours,
        crop_type=crop_type,
        ndre=ndre,
        ndre_age_hours=ndre_age_hours,
        ndvi_quality=ndvi_quality,
        ndre_quality=ndre_quality,
        allow_experimental_ndre=allow_experimental_ndre,
    )
    kc = float(kc_diagnostics["kc_dynamic"])
    resolved_etc = (
        float(etc_daily_mm)
        if etc_daily_mm is not None
        else (float(et0_daily_mm) * kc if et0_daily_mm is not None else None)
    )
    threshold, threshold_mode, threshold_reason, threshold_details = (
        _configured_threshold(farm, gdd_cumulative, resolved_etc))
    season = get_crop_season_status(farm, gdd_cumulative, as_of)
    return CropState(
        growth_stage=stage,
        growth_stage_name=STAGE_NAMES[stage],
        stress_threshold_cbar=threshold,
        kc=kc,
        gdd_cumulative=gdd_cumulative,
        crop_type=crop_type,
        etc_daily_mm=0.0 if resolved_etc is None else resolved_etc,
        et0_today_mm=(float(et0_daily_mm) if et0_daily_mm is not None else None),
        sat_ndvi_quality=(float(ndvi_quality)
                          if ndvi_quality is not None and np.isfinite(ndvi_quality) else 0.0),
        sat_ndre_quality=(float(ndre_quality)
                          if ndre_quality is not None and np.isfinite(ndre_quality) else 0.0),
        sat_ndre_age_hours=float(ndre_age_hours),
        kc_gdd=float(kc_diagnostics["kc_gdd"]),
        kc_eo=kc_diagnostics["kc_eo"],
        eo_source=str(kc_diagnostics["eo_source"]),
        eo_weight=float(kc_diagnostics["eo_weight"]),
        eo_quality_factor=float(kc_diagnostics["quality_factor"]),
        eo_plausibility_factor=float(kc_diagnostics["plausibility_factor"]),
        eo_kc_delta=float(kc_diagnostics["eo_kc_delta"]),
        eo_reason=str(kc_diagnostics["reason"]),
        eo_indices=kc_diagnostics["indices"],
        threshold_mode=threshold_mode,
        threshold_reason=threshold_reason,
        threshold_details=threshold_details,
        season_active=bool(season["active"]),
        season_status=str(season["status"]),
        season_end_reason=season["reason"],
    )


def get_stress_threshold(
    farm: FarmConfig,
    gdd_cumulative: float,
    etc_daily_mm: Optional[float] = None,
) -> float:
    return _configured_threshold(farm, gdd_cumulative, etc_daily_mm)[0]


def get_stage_threshold_range(farm, gdd_start, gdd_end, et0_daily_mm, kc_adjustment=0.0):
    """Bounds across an active stage under the same current daily weather.

    Evaluate both ends of the continuous Kc/calibration segment. Stop just
    before the next stage so maturity's Kc reset cannot contaminate late season.
    These are reference ranges, not a forecast of weather months ahead.
    """
    endpoints = [float(gdd_start)]
    if gdd_end is not None and float(gdd_end) > float(gdd_start):
        endpoints.append(float(np.nextafter(float(gdd_end), float(gdd_start))))
    values = []
    for gdd in endpoints:
        etc = (float(et0_daily_mm) * project_stage_kc(gdd, farm.crop_type, kc_adjustment)
               if et0_daily_mm is not None else None)
        values.append(get_stress_threshold(farm, gdd, etc))
    return min(values), max(values)


def project_stage_kc(gdd, crop_type, kc_adjustment=0.0):
    """Hold the current quality-weighted EO delta within crop Kc bounds."""
    delta = float(kc_adjustment)
    if not np.isfinite(delta):
        raise ValueError("Forecast Kc adjustment must be finite")
    return float(np.clip(compute_kc_gdd(gdd, crop_type) + delta, 0.0,
                         get_crop_params(crop_type).kc_mid * KC_UPPER_TOLERANCE))


def get_stress_threshold_series(
    gdd_series: pd.Series,
    farm: FarmConfig,
    etc_daily_series: Optional[pd.Series] = None,
) -> pd.Series:
    if gdd_series.empty:
        return pd.Series(dtype=float, index=gdd_series.index)
    baseline_raw = getattr(farm, "static_threshold_cbar", None)
    if baseline_raw is None:
        baseline_raw = getattr(farm, "threshold_static", None)
    mode = str(getattr(farm, "threshold_mode", "") or "").strip().lower()
    if not mode:
        mode = "dynamic" if bool(getattr(
            farm, "use_dynamic_threshold", False)) else "static"
    if mode == "dynamic":
        if etc_daily_series is None:
            etc_daily_series = pd.Series(None, index=gdd_series.index)
        else:
            etc_daily_series = etc_daily_series.reindex(gdd_series.index)
        return pd.Series(
            [get_stress_threshold(farm, gdd, etc)
             for gdd, etc in zip(gdd_series, etc_daily_series)],
            index=gdd_series.index,
            dtype=float,
        )
    if mode != "static":
        raise ValueError("threshold_mode must be 'static' or 'dynamic'")
    baseline = _positive_finite(baseline_raw, "static_threshold_cbar")
    return pd.Series(baseline, index=gdd_series.index, dtype=float)


def build_forecast_thresholds(
    farm,
    current_gdd: float,
    tension_forecast: dict,
    forecast_weather: pd.DataFrame,
    current_timestamp,
    *,
    horizon_timestamps=None,
    kc_adjustment=0.0,
) -> dict:
    """Build a threshold for every tension-forecast horizon.

    All modes project crop-season state from forecast temperature so horizons
    after harvest or annual-crop maturity can be disabled. Dynamic mode also
    adjusts the depletion fraction using forecast daily ETc.
    """
    if not tension_forecast:
        return {}
    mode = str(getattr(farm, "threshold_mode", "static") or "static").lower()
    if mode not in {"static", "dynamic"}:
        raise ValueError("threshold_mode must be 'static' or 'dynamic'")
    if not isinstance(forecast_weather, pd.DataFrame) or forecast_weather.empty:
        raise ValueError("forecast crop-season thresholds require forecast weather")
    demand_threshold = mode == "dynamic" and not getattr(farm, "stage_thresholds_cbar", None)
    required = {"Temperature"}
    if demand_threshold:
        required.add("Et0_evapotranspiration")
    missing = sorted(required.difference(forecast_weather.columns))
    if missing:
        raise ValueError(
            f"forecast crop-season thresholds require weather fields: {missing}")

    now = pd.Timestamp(current_timestamp)
    weather = forecast_weather.copy()
    weather.index = pd.to_datetime(weather.index, utc=True, errors="coerce")
    weather = weather[~weather.index.isna()].sort_index()
    weather = deduplicate_weather_frame(weather)
    timezone_name = getattr(farm, "timezone", None) or "UTC"
    weather.index = weather.index.tz_convert(timezone_name)
    if now.tzinfo is None:
        now = now.tz_localize(timezone_name)
    else:
        now = now.tz_convert(timezone_name)
    horizons = {}
    for label in tension_forecast:
        match = re.search(r"(\d+(?:\.\d+)?)h$", str(label).strip().lower())
        if not match:
            raise ValueError(f"Cannot build threshold for horizon {label!r}")
        horizons[label] = float(match.group(1))
    # Rounded display labels must not move exact model timestamps.
    targets = {}
    for label, hours in horizons.items():
        target = (pd.Timestamp(horizon_timestamps[label])
                  if horizon_timestamps is not None
                  else now + pd.Timedelta(hours=hours))
        target = (target.tz_localize(timezone_name) if target.tzinfo is None
                  else target.tz_convert(timezone_name))
        if target <= now:
            raise ValueError("Forecast targets must be after the current timestamp")
        targets[label] = target
    last_target = max(targets.values())
    if weather.empty or weather.index.max() < last_target:
        raise ValueError(
            "forecast weather does not cover the complete tension-forecast horizon")

    params = get_crop_params(farm.crop_type)
    temperature = pd.to_numeric(weather["Temperature"], errors="coerce")
    et0 = (
        pd.to_numeric(weather["Et0_evapotranspiration"], errors="coerce")
        if demand_threshold else None
    )
    daily_et0 = complete_daily_et0(weather, timezone_name) if demand_threshold else None
    kc_adjustment = float(kc_adjustment)
    if not np.isfinite(kc_adjustment):
        raise ValueError("Forecast Kc adjustment must be finite")
    if not np.isfinite(temperature.to_numpy(dtype=float)).all() or (
            et0 is not None and not np.isfinite(et0.to_numpy(dtype=float)).all()):
        raise ValueError("forecast temperature and ET0 must be complete and finite")

    thresholds = {}
    current_season_start = (
        phenology_season_start(now, farm.planting_date, farm.crop_type)
        if params.is_perennial else None
    )
    for label in sorted(horizons, key=horizons.get):
        target = targets[label]
        interval_start = now
        projected_base_gdd = float(current_gdd)
        if params.is_perennial:
            target_season_start = phenology_season_start(
                target, farm.planting_date, farm.crop_type)
            if target_season_start != current_season_start:
                interval_start = max(now, target_season_start)
                projected_base_gdd = 0.0
        interval = temperature[(temperature.index > interval_start)
                               & (temperature.index <= target)]
        if target > interval_start and (
                interval.empty or interval.index.max() < target):
            raise ValueError(
                f"forecast temperature is unavailable through {label}")
        previous = interval_start
        gdd_increment = 0.0
        for timestamp, value in interval.items():
            hours = max(0.0, (timestamp - previous).total_seconds() / 3600.0)
            degree_temperature = float(np.clip(
                value, params.t_base, params.t_ceiling))
            gdd_increment += max(0.0, degree_temperature - params.t_base) * hours / 24.0
            previous = timestamp
        projected_gdd = projected_base_gdd + gdd_increment

        season = get_crop_season_status(farm, projected_gdd, target)
        if not season["active"]:
            thresholds[label] = None
            continue

        if mode == "static" or not demand_threshold:
            thresholds[label] = get_stress_threshold(
                farm, projected_gdd)
            continue

        target_day = target.normalize()
        total_et0 = daily_et0.get(target_day, np.nan)
        if not np.isfinite(total_et0):
            raise ValueError(f"Complete local-day forecast ET0 is required for {target_day.date()}")
        projected_kc = project_stage_kc(projected_gdd, farm.crop_type, kc_adjustment)
        etc_daily = float(total_et0) * projected_kc
        thresholds[label] = get_stress_threshold(
            farm, projected_gdd, etc_daily)
    return thresholds


def compute_gdd_from_weather(
    tmax_series: pd.Series,
    tmin_series: pd.Series,
    farm: FarmConfig,
) -> pd.Series:
    return compute_gdd_series(
        tmax_series,
        tmin_series,
        farm.planting_date,
        farm.crop_type,
        farm.initial_gdd,
    )


def compute_phenology_features(
    df: pd.DataFrame,
    farm: FarmConfig,
) -> pd.DataFrame:
    df = df.copy()
    crop_type = farm.crop_type
    params = get_crop_params(crop_type)

    planting_ts = pd.Timestamp(farm.planting_date)
    if df.index.tz is not None and planting_ts.tz is None:
        planting_ts = planting_ts.tz_localize(df.index.tz)
    elif df.index.tz is None and planting_ts.tz is not None:
        planting_ts = planting_ts.tz_localize(None)

    if "Temperature" in df.columns:
        daily_tmax, daily_tmin = daily_temperature_extrema(
            df["Temperature"], getattr(farm, "timezone", None))
    elif "Tmax_daily" in df.columns and "Tmin_daily" in df.columns:
        daily_tmax = df["Tmax_daily"].resample("D").first()
        daily_tmin = df["Tmin_daily"].resample("D").first()
        log.info("  Using Tmax_daily/Tmin_daily for GDD")
    else:
        log.error(
            "  Farm '%s': no temperature data found — need 'Temperature' or 'Tmax_daily'/'Tmin_daily'. "
            "Phenology columns set to zero/NaN defaults.",
            farm.name,
        )
        df["gdd_cumulative"] = 0.0
        df["growth_stage"] = STAGE_PRE_EMERGENCE
        df["kc_gdd"] = params.kc_ini
        df["kc_ndvi"] = np.nan
        df["kc_ndre"] = np.nan
        df["kc_dynamic"] = params.kc_ini
        df["etc_daily"] = 0.0
        return df

    log.info(
        "  [%s] GDD: crop=%s, T_base=%.1f°C, T_ceil=%.1f°C, planting=%s",
        farm.name,
        crop_type,
        params.t_base,
        params.t_ceiling,
        planting_ts.date(),
    )
    daily_gdd = compute_gdd_series(
        daily_tmax, daily_tmin, planting_ts, crop_type,
        initial_gdd=farm.initial_gdd,
    )
    gdd_hourly = daily_gdd.reindex(df.index, method="ffill")
    n_unfilled = int(gdd_hourly.isna().sum())
    if n_unfilled > 0:
        log.warning(
            "  %s hourly rows before first daily GDD — filling with 0",
            n_unfilled,
        )
    df["gdd_cumulative"] = gdd_hourly.fillna(0.0)

    df["growth_stage"] = get_growth_stage_series(
        df["gdd_cumulative"], crop_type)
    gdd_final = df["gdd_cumulative"].iloc[-1]
    log.info("  GDD range: 0 → %.0f", gdd_final)
    for s in sorted(df["growth_stage"].unique()):
        first_idx = df[df["growth_stage"] == s].index[0]
        log.info(
            "    %s entered %s (GDD=%.0f)",
            f"{STAGE_NAMES[s]:20s}",
            first_idx.date(),
            df.loc[first_idx, 'gdd_cumulative'],
        )

    df["kc_gdd"] = compute_kc_gdd_series(df["gdd_cumulative"], crop_type)

    # The provider supplies prepared EO indices. Freshness, cadence quality and
    # value plausibility bound their contribution in phenology_engine.
    ndvi_source = df["sat_ndvi"] if "sat_ndvi" in df.columns else None
    ndre_source = df["sat_ndre"] if "sat_ndre" in df.columns else None
    df["kc_ndvi"] = (
        compute_kc_ndvi(ndvi_source.values.astype(float), crop_type)
        if ndvi_source is not None else np.nan
    )
    df["kc_ndre"] = (
        compute_kc_ndre(ndre_source.values.astype(float), crop_type)
        if ndre_source is not None else np.nan
    )

    if "sat_ndvi_age" in df.columns:
        ndvi_age = df["sat_ndvi_age"].fillna(np.inf)
        log.info(
            "  sat_ndvi_age: mean=%.1fh",
            ndvi_age[ndvi_age < np.inf].mean(),
        )
    elif "satellite_data_age" in df.columns:
        ndvi_age = df["satellite_data_age"].fillna(np.inf)
        log.info("  Using satellite_data_age as NDVI age proxy")
    else:
        ndvi_age = pd.Series(np.inf, index=df.index)
        log.warning(
            "  No satellite age — Kc_dynamic will use pure GDD-based Kc")

    # SAR correction: when VV backscatter confirms canopy, cap NDVI staleness
    # so cloud-cover gaps don't decay the satellite quality weight to zero.
    if "sat_vv_db" in df.columns:
        ndvi_age = sar_adjusted_ndvi_age(ndvi_age, df["sat_vv_db"])

    ndvi_col = (ndvi_source if ndvi_source is not None
                else pd.Series(np.nan, index=df.index))
    ndre_col = ndre_source
    df["kc_dynamic"] = compute_kc_dynamic_series(
        df["gdd_cumulative"], ndvi_col, ndvi_age, crop_type, ndre_col,
        df.get("sat_ndvi_quality"), df.get("sat_ndre_age"),
        df.get("sat_ndre_quality"),
        allow_experimental_ndre=bool(
            getattr(farm, "enable_experimental_ndre_kc", False)),
    )

    if "Et0_evapotranspiration" in df.columns:
        df["etc_daily"] = (
            (df["Et0_evapotranspiration"] * df["kc_dynamic"])
            .rolling(24, min_periods=1).sum()
        )
    else:
        df["etc_daily"] = 0.0

    log.info(
        "  Kc_dynamic [%.3f, %.3f]  ETc [%.2f, %.2f] mm/day",
        df["kc_dynamic"].min(),
        df["kc_dynamic"].max(),
        df["etc_daily"].min(),
        df["etc_daily"].max(),
    )
    return df
