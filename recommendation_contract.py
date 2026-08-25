"""Stable, explainable irrigation recommendation composition.

The helpers are deliberately independent of HTTP and hardware so agronomic
calculations can be tested deterministically and reused by every UI surface.
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from crops import STAGE_NAMES, get_crop_params
from phenology_engine import compute_kc_gdd


SCHEMA_VERSION = "1.0"


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


def _daily_gdd(weather: Optional[pd.DataFrame], crop_type: str) -> pd.Series:
    if not isinstance(weather, pd.DataFrame) or weather.empty or "Temperature" not in weather:
        return pd.Series(dtype=float)
    temperatures = pd.to_numeric(weather["Temperature"], errors="coerce").dropna()
    if temperatures.empty:
        return pd.Series(dtype=float)
    params = get_crop_params(crop_type)
    if not isinstance(temperatures.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    daily_max = temperatures.resample("D").max().clip(upper=params.t_ceiling)
    daily_min = temperatures.resample("D").min().clip(lower=params.t_base)
    return (((daily_max + daily_min) / 2.0) - params.t_base).clip(lower=0.0)


def estimate_next_stage(crop_type, growth_stage, cumulative_gdd, forecast_weather=None, as_of=None):
    current_gdd = _finite(cumulative_gdd) or 0.0
    target = _next_stage_target(crop_type, growth_stage)
    if target is None:
        return {"name": None, "target_gdd": None, "remaining_gdd": 0.0,
                "estimated_date": None, "estimated_range": None,
                "estimate_source": "complete"}
    name, target_gdd = target
    remaining = max(0.0, float(target_gdd) - current_gdd)
    daily = _daily_gdd(forecast_weather, crop_type)
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
        return {"verdict": "insufficient_data", "label": "Insufficient satellite data",
                "reason": "No usable satellite validation is available.", "source_age_hours": age}
    quality = _finite(validation.get("data_quality"))
    if quality is not None and quality <= 0:
        verdict, label = "stale_data", "Satellite data is stale"
    elif validation.get("insufficient_data", False):
        verdict, label = "insufficient_data", "Insufficient satellite history"
    else:
        agreement = _finite(validation.get("direction_agreement"))
        if agreement is not None and agreement >= 0.7:
            verdict, label = "supports", "Satellite supports the assessment"
        elif agreement is not None and agreement < 0.4:
            verdict, label = "contradicts", "Satellite contradicts the assessment"
        else:
            verdict, label = "insufficient_data", "Satellite evidence is mixed"
    return {"verdict": verdict, "label": label,
            "reason": validation.get("reason") or label,
            "source": (validation.get("factors") or {}).get("satellite_source"),
            "source_age_hours": age,
            "confidence": _finite(validation.get("overall_confidence"))}


def _timestamps(forecast_timestamps: Iterable, tension_forecast: dict, now):
    values = []
    for item in forecast_timestamps or []:
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


def build_water_outlook(forecast_timestamps, tension_forecast, forecast_weather,
                        crop_type, current_gdd, area_m2=None, efficiency=0.7,
                        now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    timestamps = _timestamps(forecast_timestamps, tension_forecast, now)
    weather = forecast_weather.copy() if isinstance(forecast_weather, pd.DataFrame) else pd.DataFrame()
    if not weather.empty:
        weather.index = pd.to_datetime(weather.index, utc=True, errors="coerce")
        weather = weather[~weather.index.isna()].sort_index()
    daily_gdd = _daily_gdd(weather, crop_type)
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
        et0 = None
        rain = None
        if not interval.empty and "Et0_evapotranspiration" in interval:
            et0 = float(pd.to_numeric(interval["Et0_evapotranspiration"], errors="coerce").fillna(0).clip(lower=0).sum())
        if not interval.empty and "Rain" in interval:
            rain = float(pd.to_numeric(interval["Rain"], errors="coerce").fillna(0).clip(lower=0).sum())
        projected_gdd = float(current_gdd or 0.0)
        if not daily_gdd.empty:
            projected_gdd += float(daily_gdd[daily_gdd.index <= timestamp].sum())
        kc = float(compute_kc_gdd(projected_gdd, crop_type))
        etc = None if et0 is None else max(0.0, et0 * kc)
        credit = None if rain is None or etc is None else min(rain, etc)
        net = None if etc is None else max(0.0, etc - (credit or 0.0))
        volume = None
        if net is not None and _finite(area_m2) and _finite(efficiency) and float(efficiency) > 0:
            volume = net * float(area_m2) / (float(efficiency) * 1000.0)
        tension = ordered_tensions[index][1] if index < len(ordered_tensions) else None
        points.append({"timestamp": timestamp.isoformat(),
                       "tension_cbar": None if tension is None else round(tension, 3),
                       "et0_mm": None if et0 is None else round(et0, 3),
                       "kc": round(kc, 3), "etc_mm": None if etc is None else round(etc, 3),
                       "rain_credit_mm": None if credit is None else round(credit, 3),
                       "net_demand_mm": None if net is None else round(net, 3),
                       "recommended_volume_m3": None if volume is None else round(volume, 4)})
        previous = timestamp
    totals = {}
    for key in ("et0_mm", "etc_mm", "rain_credit_mm", "net_demand_mm", "recommended_volume_m3"):
        present = [point[key] for point in points if point[key] is not None]
        totals[key] = round(sum(present), 3) if present else None
    days = []
    for date_value in sorted({pd.Timestamp(point["timestamp"]).date() for point in points}):
        day_points = [point for point in points if pd.Timestamp(point["timestamp"]).date() == date_value]
        offset = (date_value - now.date()).days
        label = "Today" if offset == 0 else ("Tomorrow" if offset == 1 else f"+{offset} days")
        summary = {"date": date_value.isoformat(), "label": label,
                   "timestamps": [point["timestamp"] for point in day_points]}
        for key in ("et0_mm", "etc_mm", "rain_credit_mm", "net_demand_mm", "recommended_volume_m3"):
            values = [point[key] for point in day_points if point[key] is not None]
            summary[key] = round(sum(values), 3) if values else None
        days.append(summary)
    interval_hours = None
    if len(timestamps) > 1:
        interval_hours = round(float(np.median(np.diff(pd.DatetimeIndex(timestamps).asi8)) / 3.6e12), 3)
    horizon_hours = round((timestamps[-1] - now).total_seconds() / 3600.0, 3) if timestamps else None
    return {"timestamps": [point["timestamp"] for point in points], "points": points,
            "days": days, "totals": totals, "interval_hours": interval_hours,
            "horizon_hours": horizon_hours, "aligned_to_tension_forecast": True}


def compose_recommendation(*, plot, base, decision, pipeline, forecast_weather=None, generated_at=None):
    generated_at = pd.Timestamp(generated_at or datetime.now(timezone.utc))
    current = (decision or {}).get("current_tension", getattr(pipeline, "current_tension", None))
    threshold = (decision or {}).get("stress_threshold", base.get("threshold_cbar"))
    condition = threshold_condition(current, threshold)
    state = getattr(pipeline, "crop_state", None)
    stage_code = getattr(state, "growth_stage", 0)
    crop_type = base.get("crop_type", getattr(plot, "crop_type", "generic"))
    weather = forecast_weather if forecast_weather is not None else getattr(state, "weather_forecast_frame", None)
    next_stage = estimate_next_stage(crop_type, stage_code, base.get("gdd_cumulative", 0), weather, generated_at)
    outlook = build_water_outlook(
        getattr(pipeline, "forecast_timestamps", []), getattr(pipeline, "tension_forecast", {}),
        weather, crop_type, base.get("gdd_cumulative", 0), base.get("plot_area_m2"),
        base.get("efficiency", 0.7), generated_at)
    urgency = (decision or {}).get("urgency", "unknown")
    should_irrigate = bool((decision or {}).get("should_irrigate", False))
    action_labels = {"critical": "Irrigate now", "advise": "Prepare to irrigate",
                     "watch": "Monitor closely", "none": "No irrigation needed",
                     "error": "Recommendation unavailable", "unknown": "Awaiting forecast"}
    satellite = satellite_verdict(base.get("satellite_validation"), base.get("sat_data_age_hours"))
    net_depth_mm = _finite(base.get("recommended_depth_mm"))
    efficiency = _finite(base.get("efficiency"))
    gross_depth_mm = (
        net_depth_mm / efficiency
        if net_depth_mm is not None and efficiency is not None and efficiency > 0
        else None
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
    return {
        "schema_version": SCHEMA_VERSION, "generated_at": generated_at.isoformat(),
        "available": bool(base.get("available", False)),
        "plot": {"plot_id": getattr(plot, "stable_id", getattr(plot, "id", None)),
                 "farm_id": getattr(plot, "farm_id", None), "name": getattr(plot, "user_given_name", "")},
        "action": {"should_irrigate": should_irrigate, "urgency": urgency,
                   "label": action_labels.get(urgency, action_labels["unknown"]),
                   "mode": base.get("irrigation_mode"),
                   "advisory_only": not bool(base.get("actuator_based", False))},
        "condition": condition,
        "crop": {"type": crop_type, "current_stage": base.get("growth_stage"),
                 "gdd_cumulative": base.get("gdd_cumulative"), "kc": base.get("kc"),
                 "next_stage": next_stage},
        "timing": {"first_breach_horizon": (decision or {}).get("first_breach_horizon"),
                   "first_breach_timestamp": _iso_or_value(
                       (decision or {}).get("first_breach_timestamp")),
                   "forecast_interval_hours": getattr(pipeline, "forecast_interval_hours", None),
                   "forecast_horizon_hours": getattr(pipeline, "forecast_horizon_hours", None)},
        "water": {"recommended_depth_mm": net_depth_mm,
                  "net_depth_mm": net_depth_mm,
                  "gross_application_depth_mm": (
                      None if gross_depth_mm is None else round(gross_depth_mm, 3)),
                  "recommended_volume_m3": base.get("recommended_volume_m3"),
                  "plot_area_m2": _finite(base.get("plot_area_m2")),
                  "irrigation_type": base.get("irrigation_type"), "efficiency": efficiency,
                  "outlook": outlook},
        "weather": {"etc_daily_mm": base.get("etc_daily_mm"),
                    "rain_since_planting_mm": base.get("rain_since_planting_mm"),
                    "rain_last_24h_mm": base.get("rain_last_24h_mm"),
                    "rain_forecast_mm": base.get("rain_forecast_mm"),
                    "historical_rain_credit_mm": base.get("historical_rain_credit_mm"),
                    "cumulative_etc_mm": base.get("cumulative_etc_mm"),
                    "net_irrigation_need_mm": base.get("net_irrigation_need_mm"),
                    "age_hours": base.get("weather_data_age_hours")},
        "satellite": {**satellite, "ndvi": base.get("sat_ndvi"),
                      "ndre": base.get("sat_ndre")},
        "freshness": {"stale": bool(base.get("data_stale")),
                      "stale_sources": list(base.get("data_stale_sources", []))},
        "limitations": limitations,
        "reason": condition["message"],
    }
