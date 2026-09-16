"""Stable, explainable irrigation recommendation composition.

The helpers are deliberately independent of HTTP and hardware so agronomic
calculations can be tested deterministically and reused by every UI surface.
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from crops import STAGE_NAMES, get_crop_params
from phenology_engine import (
    compute_daily_gdd,
    compute_kc_gdd,
    daily_temperature_extrema,
)
from water_demand import calculate_irrigation_requirement_from_etc
from weather_quality import deduplicate_weather_frame


SCHEMA_VERSION = "2.0"


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _iso_or_value(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def threshold_condition(current_tension, threshold_cbar):
    current = _finite(current_tension)
    threshold = _finite(threshold_cbar)
    if current is None or threshold is None:
        return {"current_tension_cbar": current, "threshold_cbar": threshold,
                "margin_cbar": None, "status": "unknown",
                "message": "Current soil tension or threshold is unavailable."}
    margin = threshold - current
    if margin < 0:
        status = "threshold_exceeded"
        message = f"Soil tension is {abs(margin):.1f} cbar beyond the irrigation threshold."
    elif margin == 0:
        status = "at_threshold"
        message = "Soil tension is at the irrigation threshold."
    else:
        status = "within_threshold"
        message = f"Soil tension can rise {margin:.1f} cbar before reaching the threshold."
    return {"current_tension_cbar": round(current, 2),
            "threshold_cbar": round(threshold, 2), "margin_cbar": round(margin, 2),
            "status": status, "message": message}


def _next_stage_target(crop_type, growth_stage):
    params = get_crop_params(crop_type)
    targets = {
        0: (STAGE_NAMES[1], params.gdd_emergence),
        1: (STAGE_NAMES[2], params.gdd_dev_end),
        2: (STAGE_NAMES[3], params.gdd_mid_end),
        3: (STAGE_NAMES[4], params.gdd_maturity),
    }
    return targets.get(int(growth_stage))


def _daily_gdd(
    weather: Optional[pd.DataFrame],
    crop_type: str,
    timezone_name: Optional[str] = None,
    after=None,
) -> pd.Series:
    if not isinstance(weather, pd.DataFrame) or weather.empty or "Temperature" not in weather:
        return pd.Series(dtype=float)
    temperatures = pd.to_numeric(weather["Temperature"], errors="coerce").dropna()
    if temperatures.empty:
        return pd.Series(dtype=float)
    params = get_crop_params(crop_type)
    if not isinstance(temperatures.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    daily_max, daily_min = daily_temperature_extrema(
        temperatures, timezone_name)
    result = pd.Series(
        [compute_daily_gdd(tmax, tmin, params.t_base, params.t_ceiling)
         for tmax, tmin in zip(daily_max, daily_min)],
        index=daily_max.index,
        dtype=float,
    )
    if after is not None and not result.empty:
        cutoff = pd.Timestamp(after)
        if result.index.tz is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        elif result.index.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(result.index.tz)
        elif result.index.tz is not None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(result.index.tz)
        # Current GDD already includes observations through `after`; do not add
        # a full forecast value for that same partially observed local day.
        result = result[result.index > cutoff.normalize()]
    return result


def estimate_next_stage(crop_type, growth_stage, cumulative_gdd,
                        forecast_weather=None, as_of=None, timezone_name=None):
    current_gdd = _finite(cumulative_gdd) or 0.0
    target = _next_stage_target(crop_type, growth_stage)
    if target is None:
        return {"name": None, "target_gdd": None, "remaining_gdd": 0.0,
                "estimated_date": None, "estimated_range": None,
                "estimate_source": "complete"}
    name, target_gdd = target
    remaining = max(0.0, float(target_gdd) - current_gdd)
    daily = _daily_gdd(
        forecast_weather, crop_type, timezone_name, after=as_of)
    estimate = None
    source = "unavailable"
    if remaining == 0:
        estimate = pd.Timestamp(as_of or datetime.now(timezone.utc))
        source = "already_reached"
    elif not daily.empty and daily.sum() > 0:
        cumulative = daily.cumsum()
        reached = cumulative[cumulative >= remaining]
        if not reached.empty:
            estimate = reached.index[0]
            source = "forecast"
        else:
            average = float(daily.mean())
            extra_days = int(np.ceil((remaining - float(cumulative.iloc[-1])) / average))
            estimate = daily.index[-1] + pd.Timedelta(days=max(1, extra_days))
            source = "forecast_rate_extrapolation"
    if estimate is None:
        date_value = None
        date_range = None
    else:
        estimate = pd.Timestamp(estimate)
        date_value = estimate.date().isoformat()
        uncertainty_days = 1 if source in {"forecast", "already_reached"} else max(2, int(np.ceil(remaining * 0.1 / max(float(daily.mean()), 1.0))))
        date_range = {
            "start": (estimate - pd.Timedelta(days=uncertainty_days)).date().isoformat(),
            "end": (estimate + pd.Timedelta(days=uncertainty_days)).date().isoformat(),
        }
    return {"name": name, "target_gdd": float(target_gdd),
            "remaining_gdd": round(remaining, 1), "estimated_date": date_value,
            "estimated_range": date_range, "estimate_source": source}


def satellite_verdict(validation, age_hours=None):
    age = _finite(age_hours)
    if not isinstance(validation, dict) or not validation.get("available", False):
        return {"verdict": "insufficient_data", "label": "Insufficient canopy data",
                "reason": "No usable canopy assessment is available.", "source_age_hours": age,
                "assessment_type": "canopy_consistency"}
    quality = _finite(validation.get("data_quality"))
    if quality is not None and quality <= 0:
        verdict, label = "stale_data", "Canopy observation is stale"
    elif validation.get("insufficient_data", False):
        verdict, label = "insufficient_data", "Insufficient canopy history"
    else:
        agreement = _finite(validation.get("direction_agreement"))
        if agreement is not None and agreement >= 0.7:
            verdict, label = "supports", "Canopy development matches the expected stage"
        elif agreement is not None and agreement < 0.4:
            verdict, label = "contradicts", "Canopy development conflicts with the expected stage"
        else:
            verdict, label = "insufficient_data", "Canopy evidence is mixed"
    return {"verdict": verdict, "label": label,
            "reason": validation.get("reason") or label,
            "source": (validation.get("factors") or {}).get("satellite_source"),
            "source_age_hours": age,
            "confidence": _finite(validation.get("overall_confidence")),
            "assessment_type": validation.get("assessment_type", "canopy_consistency"),
            "canopy_status": validation.get("canopy_status")}


def _timestamps(forecast_timestamps: Iterable, tension_forecast: dict, now):
    values = []
    for item in ([] if forecast_timestamps is None else forecast_timestamps):
        timestamp = pd.to_datetime(item, utc=True, errors="coerce")
        if not pd.isna(timestamp):
            values.append(timestamp)
    if values:
        return sorted(set(values))
    for label in tension_forecast or {}:
        text = str(label).strip().lower().removesuffix("h")
        try:
            values.append(pd.Timestamp(now) + pd.Timedelta(hours=float(text)))
        except (TypeError, ValueError):
            continue
    return sorted(set(values))


def _interval_total(frame, column):
    """Return a non-negative interval total, preserving missing data as unknown."""
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.empty or values.isna().any():
        return None
    return float(values.clip(lower=0).sum())


def _complete_point_sum(points, key):
    """Sum a point field only when every interval has a known value."""
    if not points or any(point.get(key) is None for point in points):
        return None
    return round(sum(point[key] for point in points), 3)


def build_water_outlook(forecast_timestamps, tension_forecast, forecast_weather,
                        crop_type, current_gdd, now=None, *, plot_area_m2=None,
                        application_efficiency=0.85,
                        effective_rainfall_fraction=0.80,
                        demand_horizon_hours=24.0,
                        timezone_name=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    timestamps = _timestamps(forecast_timestamps, tension_forecast, now)
    weather = forecast_weather.copy() if isinstance(forecast_weather, pd.DataFrame) else pd.DataFrame()
    weather_error = None
    if not weather.empty:
        weather.index = pd.to_datetime(weather.index, utc=True, errors="coerce")
        weather = weather[~weather.index.isna()].sort_index()
        try:
            weather = deduplicate_weather_frame(weather)
        except ValueError as exc:
            weather_error = str(exc)
            weather = weather.iloc[:0]
    daily_gdd = _daily_gdd(
        weather, crop_type, timezone_name, after=now)
    ordered_tensions = []
    for label, value in (tension_forecast or {}).items():
        try:
            hours = float(str(label).strip().lower().removesuffix("h"))
            tension = _finite(value)
        except (TypeError, ValueError):
            continue
        ordered_tensions.append((hours, tension))
    ordered_tensions.sort(key=lambda item: item[0])
    points, previous = [], now
    for index, timestamp in enumerate(timestamps):
        timestamp = pd.Timestamp(timestamp)
        interval = weather[(weather.index > previous) & (weather.index <= timestamp)] if not weather.empty else pd.DataFrame()
        et0 = _interval_total(interval, "Et0_evapotranspiration")
        rain = _interval_total(interval, "Rain")
        projected_gdd = float(current_gdd or 0.0)
        if not daily_gdd.empty:
            projected_gdd += float(daily_gdd[daily_gdd.index <= timestamp].sum())
        kc = float(compute_kc_gdd(projected_gdd, crop_type))
        etc = None if et0 is None else max(0.0, et0 * kc)
        effective_rain = (
            None if rain is None
            else max(0.0, rain * float(effective_rainfall_fraction))
        )
        net = (
            None if etc is None or effective_rain is None
            else max(0.0, etc - effective_rain)
        )
        tension = ordered_tensions[index][1] if index < len(ordered_tensions) else None
        points.append({"timestamp": timestamp.isoformat(),
                       "tension_cbar": None if tension is None else round(tension, 3),
                       "et0_mm": None if et0 is None else round(et0, 3),
                       "kc": round(kc, 3), "etc_mm": None if etc is None else round(etc, 3),
                       "rainfall_mm": None if rain is None else round(rain, 3),
                       "effective_rainfall_mm": None if effective_rain is None else round(effective_rain, 3),
                       "rain_credit_mm": None if effective_rain is None else round(effective_rain, 3),
                       "net_demand_mm": None if net is None else round(net, 3)})
        previous = timestamp
    totals = {}
    for key in ("et0_mm", "etc_mm", "rainfall_mm", "effective_rainfall_mm",
                "rain_credit_mm", "net_demand_mm"):
        totals[key] = _complete_point_sum(points, key)
    days = []
    for date_value in sorted({pd.Timestamp(point["timestamp"]).date() for point in points}):
        day_points = [point for point in points if pd.Timestamp(point["timestamp"]).date() == date_value]
        offset = (date_value - now.date()).days
        label = "Today" if offset == 0 else ("Tomorrow" if offset == 1 else f"+{offset} days")
        summary = {"date": date_value.isoformat(), "label": label,
                   "timestamps": [point["timestamp"] for point in day_points]}
        for key in ("et0_mm", "etc_mm", "rainfall_mm", "effective_rainfall_mm",
                    "rain_credit_mm", "net_demand_mm"):
            summary[key] = _complete_point_sum(day_points, key)
        days.append(summary)
    interval_hours = None
    if len(timestamps) > 1:
        interval_hours = round(float(np.median(np.diff(pd.DatetimeIndex(timestamps).asi8)) / 3.6e12), 3)
    horizon_hours = round((timestamps[-1] - now).total_seconds() / 3600.0, 3) if timestamps else None
    try:
        demand_horizon = max(0.0, float(demand_horizon_hours))
        cutoff = now + pd.Timedelta(hours=demand_horizon)
        demand_points = [
            point for point in points
            if pd.Timestamp(point["timestamp"]) <= cutoff
        ]
        coverage_endpoint = (
            pd.Timestamp(demand_points[-1]["timestamp"])
            if demand_points else None
        )
        covered_hours = (
            max(0.0, (coverage_endpoint - now).total_seconds() / 3600.0)
            if coverage_endpoint is not None else 0.0
        )
        coverage_complete = (
            coverage_endpoint is not None and coverage_endpoint >= cutoff
        )
        complete = coverage_complete and all(
            point["etc_mm"] is not None and point["rainfall_mm"] is not None
            for point in demand_points
        )
        if complete:
            requirement = calculate_irrigation_requirement_from_etc(
                etc_mm=sum(point["etc_mm"] for point in demand_points),
                rainfall_mm=sum(point["rainfall_mm"] for point in demand_points),
                application_efficiency=application_efficiency,
                effective_rainfall_fraction=effective_rainfall_fraction,
                area_m2=plot_area_m2,
            )
            requirement = {
                key: (round(value, 3) if isinstance(value, float) else value)
                for key, value in requirement.items()
            }
            requirement["period_hours"] = demand_horizon
            requirement["forecast_coverage_hours"] = round(covered_hours, 3)
            requirement["complete"] = True
        else:
            if not coverage_complete:
                reason = (
                    f"The forecast covers {covered_hours:g} hours, short of the "
                    f"required {demand_horizon:g}-hour demand period."
                )
            else:
                reason = (
                    "Forecast ET0 and rainfall are required for the complete "
                    "demand period."
                )
            requirement = {
                "period_hours": demand_horizon,
                "forecast_coverage_hours": round(covered_hours, 3),
                "complete": False,
                "reason": reason,
                "etc_mm": None,
                "rainfall_mm": None,
                "effective_rainfall_mm": None,
                "net_irrigation_mm": None,
                "gross_irrigation_mm": None,
                "volume_m3": None,
                "application_efficiency": float(application_efficiency),
                "effective_rainfall_fraction": float(effective_rainfall_fraction),
            }
    except (TypeError, ValueError) as exc:
        requirement = {
            "period_hours": demand_horizon_hours,
            "complete": False,
            "reason": str(exc),
            "volume_m3": None,
        }
    if weather_error:
        requirement["reason"] = weather_error
    return {"timestamps": [point["timestamp"] for point in points], "points": points,
            "days": days, "totals": totals, "interval_hours": interval_hours,
            "horizon_hours": horizon_hours, "aligned_to_tension_forecast": True,
            "requirement": requirement}


def compose_recommendation(*, plot, base, decision, pipeline, forecast_weather=None, generated_at=None):
    generated_at = pd.Timestamp(generated_at or datetime.now(timezone.utc))
    current = (decision or {}).get("current_tension", getattr(pipeline, "current_tension", None))
    threshold = (decision or {}).get("stress_threshold", base.get("threshold_cbar"))
    condition = threshold_condition(current, threshold)
    condition["mode"] = base.get("threshold_mode", "static")
    condition["baseline_cbar"] = _finite(base.get("threshold_static_cbar"))
    condition["reason"] = base.get("threshold_reason", "")
    condition["comparison_operator"] = "greater_than_or_equal"
    condition["forecast_thresholds_cbar"] = dict(getattr(
        pipeline, "stress_threshold_forecast", {}) or {})
    condition["threshold_details"] = base.get("threshold_details")
    state = getattr(pipeline, "crop_state", None)
    stage_code = getattr(state, "growth_stage", 0)
    crop_type = base.get("crop_type", getattr(plot, "crop_type", ""))
    timezone_name = getattr(plot, "timezone", None)
    weather = forecast_weather if forecast_weather is not None else getattr(state, "weather_forecast_frame", None)
    next_stage = estimate_next_stage(
        crop_type, stage_code, base.get("gdd_cumulative", 0), weather,
        generated_at, timezone_name)
    outlook = build_water_outlook(
        getattr(pipeline, "forecast_timestamps", []), getattr(pipeline, "tension_forecast", {}),
        weather, crop_type, base.get("gdd_cumulative", 0), generated_at,
        plot_area_m2=base.get("plot_area_m2"),
        application_efficiency=base.get("application_efficiency", 0.85),
        effective_rainfall_fraction=base.get("effective_rainfall_fraction", 0.80),
        demand_horizon_hours=base.get("demand_horizon_hours", 24.0),
        timezone_name=timezone_name)
    requirement = outlook["requirement"]
    calculated_volume_m3 = _finite(requirement.get("volume_m3"))
    credit = base.get("applied_irrigation_credit") or {}
    applied_volume_m3 = max(0.0, _finite(credit.get("volume_m3")) or 0.0)
    credited_volume_m3 = (
        min(calculated_volume_m3, applied_volume_m3)
        if calculated_volume_m3 is not None else 0.0
    )
    remaining_volume_m3 = (
        max(0.0, calculated_volume_m3 - credited_volume_m3)
        if calculated_volume_m3 is not None else None
    )
    area_m2 = _finite(base.get("plot_area_m2"))
    applied_depth_mm = (
        applied_volume_m3 * 1000.0 / area_m2
        if area_m2 is not None and area_m2 > 0 else None
    )
    gross_depth_mm = _finite(requirement.get("gross_irrigation_mm"))
    net_depth_mm = _finite(requirement.get("net_irrigation_mm"))
    application_efficiency = _finite(requirement.get("application_efficiency"))
    credited_gross_depth_mm = (
        credited_volume_m3 * 1000.0 / area_m2
        if area_m2 is not None and area_m2 > 0 else None
    )
    remaining_gross_depth_mm = (
        max(0.0, gross_depth_mm - credited_gross_depth_mm)
        if gross_depth_mm is not None and credited_gross_depth_mm is not None
        else None
    )
    remaining_net_depth_mm = (
        max(0.0, net_depth_mm - credited_gross_depth_mm * application_efficiency)
        if (net_depth_mm is not None and credited_gross_depth_mm is not None
            and application_efficiency is not None)
        else None
    )
    requirement.update({
        "recorded_applied_irrigation_m3": round(applied_volume_m3, 3),
        "credited_applied_irrigation_m3": round(credited_volume_m3, 3),
        "recorded_applied_irrigation_depth_mm": (
            round(applied_depth_mm, 3) if applied_depth_mm is not None else None),
        "remaining_gross_irrigation_mm": (
            round(remaining_gross_depth_mm, 3)
            if remaining_gross_depth_mm is not None else None),
        "remaining_net_irrigation_mm": (
            round(remaining_net_depth_mm, 3)
            if remaining_net_depth_mm is not None else None),
        "remaining_volume_m3": (
            round(remaining_volume_m3, 3)
            if remaining_volume_m3 is not None else None),
        "irrigation_credit_since": credit.get("since"),
        "irrigation_credit_until": credit.get("until"),
        "credited_operation_count": int(credit.get("operation_count") or 0),
    })
    urgency = (decision or {}).get("urgency", "unknown")
    should_irrigate = bool((decision or {}).get("should_irrigate", False))
    season_active = bool(base.get("season_active", True))
    if not season_active:
        should_irrigate = False
        urgency = "none"
    available = bool(base.get("available", False))
    action_labels = {"critical": "Irrigate now", "advise": "Prepare to irrigate",
                     "watch": "Monitor closely", "none": "No irrigation needed",
                     "error": "Recommendation unavailable", "unknown": "Awaiting forecast"}
    satellite = satellite_verdict(base.get("satellite_validation"), base.get("sat_data_age_hours"))
    if (requirement.get("complete")
            and remaining_volume_m3 is not None
            and remaining_volume_m3 <= 0):
        should_irrigate = False
        urgency = "none"
    recommended_volume_m3 = (
        remaining_volume_m3 if should_irrigate and available else
        (0.0 if remaining_volume_m3 is not None else None)
    )
    limitations = []
    if base.get("data_stale"):
        limitations.append("Stale inputs: " + ", ".join(base.get("data_stale_sources", [])))
    if not outlook["points"]:
        limitations.append("No timestamped tension forecast is available for the water outlook.")
    elif outlook["totals"]["et0_mm"] is None:
        limitations.append("Forecast ET0 is unavailable; water demand cannot be quantified for each interval.")
    if satellite["verdict"] in {"insufficient_data", "stale_data"}:
        limitations.append(satellite["label"] + ".")
    if calculated_volume_m3 is None:
        limitations.append(
            "Calculated irrigation volume is unavailable; complete forecast data and plot area are required.")
    return {
        "schema_version": SCHEMA_VERSION, "generated_at": generated_at.isoformat(),
        "available": available,
        "plot": {"plot_id": getattr(plot, "stable_id", getattr(plot, "id", None)),
                 "farm_id": getattr(plot, "farm_id", None), "name": getattr(plot, "user_given_name", "")},
        "action": {"should_irrigate": should_irrigate if available else False,
                   "urgency": urgency if available else "unknown",
                   "label": (action_labels.get(urgency, action_labels["unknown"])
                             if available else action_labels["error"]),
                   "mode": base.get("irrigation_mode"),
                   "advisory_only": not bool(base.get("actuator_based", False)),
                   "reason": (decision or {}).get("status_message")},
        "condition": condition,
        "crop": {"type": crop_type, "current_stage": base.get("growth_stage"),
                 "gdd_cumulative": base.get("gdd_cumulative"), "kc": base.get("kc"),
                 "kc_gdd": base.get("kc_gdd"), "kc_eo": base.get("kc_eo"),
                 "next_stage": next_stage,
                 "season_active": season_active,
                 "season_status": base.get("season_status"),
                 "season_end_reason": base.get("season_end_reason")},
        "timing": {"first_breach_horizon": (decision or {}).get("first_breach_horizon"),
                   "first_breach_timestamp": _iso_or_value(
                       (decision or {}).get("first_breach_timestamp")),
                   "forecast_interval_hours": getattr(pipeline, "forecast_interval_hours", None),
                   "forecast_horizon_hours": getattr(pipeline, "forecast_horizon_hours", None)},
        "water": {"recommended_volume_m3": recommended_volume_m3,
                  "calculated_requirement_volume_m3": calculated_volume_m3,
                  "applied_since_previous_calculation_m3": round(
                      applied_volume_m3, 3),
                  "credited_applied_irrigation_m3": round(
                      credited_volume_m3, 3),
                  "remaining_requirement_volume_m3": remaining_volume_m3,
                  "credited_operation_count": int(
                      credit.get("operation_count") or 0),
                  "irrigation_credit_since": credit.get("since"),
                  "irrigation_credit_until": credit.get("until"),
                  "volume_source": "remaining_crop_water_requirement_after_recorded_irrigation",
                  "plot_area_m2": _finite(base.get("plot_area_m2")),
                  "irrigation_type": base.get("irrigation_type"),
                  "application_efficiency": _finite(base.get("application_efficiency")),
                  "effective_rainfall_fraction": _finite(
                      base.get("effective_rainfall_fraction")),
                  "outlook": outlook},
        "weather": {"etc_daily_mm": base.get("etc_daily_mm"),
                    "rain_since_planting_mm": base.get("rain_since_planting_mm"),
                    "rain_last_24h_mm": base.get("rain_last_24h_mm"),
                    "rain_forecast_mm": base.get("rain_forecast_mm"),
                    "historical_rain_credit_mm": base.get("historical_rain_credit_mm"),
                    "cumulative_etc_mm": base.get("cumulative_etc_mm"),
                    "age_hours": base.get("weather_data_age_hours")},
        "satellite": {
            **satellite,
            "ndvi": base.get("sat_ndvi"),
            "ndre": base.get("sat_ndre"),
            "kc_contribution": {
                "source": base.get("eo_source"),
                "effective_weight": base.get("eo_weight"),
                "quality_factor": base.get("eo_quality_factor"),
                "plausibility_factor": base.get("eo_plausibility_factor"),
                "kc_delta": base.get("eo_kc_delta"),
                "etc_delta_mm": base.get("eo_etc_delta_mm"),
                "reason": base.get("eo_reason"),
            },
        },
        "freshness": {"stale": bool(base.get("data_stale")),
                      "stale_sources": list(base.get("data_stale_sources", []))},
        "capability_readiness": base.get("capability_readiness", {}),
        "limitations": limitations,
        "reason": (base.get("availability_reason")
                   if not available else condition["message"]),
    }
