import json
import hashlib
import logging
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
import pytz

import numpy as np
import pandas as pd
import requests
import runtime_config

# import create_model
from utils import NetworkUtils, TimeUtils
import threading
from recommendation_contract import compose_recommendation
from operations_store import get_operations_store, MODES

if __package__:
    from importlib import import_module

    _spaceiotbox_client = import_module(
        ".spaceiotbox_client", package=__package__)
    _spaceiotbox_satellite = import_module(
        ".spaceiotbox_satellite", package=__package__)
else:
    from importlib import import_module

    _spaceiotbox_client = import_module("spaceiotbox_client")
    _spaceiotbox_satellite = import_module("spaceiotbox_satellite")

fetch_weather_frame = _spaceiotbox_client.fetch_weather_frame
fetch_satellite_snapshot = _spaceiotbox_satellite.fetch_satellite_snapshot
fetch_satellite_history = _spaceiotbox_satellite.fetch_satellite_history
WEATHER_FIELDS = tuple(_spaceiotbox_client.LEGACY_WEATHER_COLUMNS)
SATELLITE_FIELDS = tuple(_spaceiotbox_satellite.SATELLITE_COLUMNS)

# Crop model — computes stress threshold from phenology
try:
    from crop_model import compute_gdd_from_weather, get_crop_state
    from phenology_engine import compute_kc_gdd_series
    from phenology_engine import get_growth_stage
    from phenology_engine import check_satellite_tension_consistency
    from crops import STAGE_NAMES
    HAS_CROP_MODEL = True
except ImportError:
    HAS_CROP_MODEL = False

# Decision engine — compares forecast vs threshold
try:
    from decision_engine import evaluate_forecast
    HAS_DECISION_ENGINE = True
except ImportError:
    HAS_DECISION_ENGINE = False


# Globals
WEATHER_STALE_HOURS = 6.0
# Per-plot irrigation retry counts (thread-safe: keyed by plot.id)
# Prevents cross-plot interference when multiple plots verify concurrently.
_irrigation_retries: dict = {}
# Tracks in-flight requests per plot so repeated recommendations cannot send
# duplicate actuator commands before confirmation completes.
_active_irrigations = set()
_irrigation_lock = threading.Lock()


def _release_active_irrigation(plot_id):
    """Release a plot's in-flight reservation under the shared lock."""
    with _irrigation_lock:
        _active_irrigations.discard(plot_id)

# Cache for dynamic threshold (recomputed at most once per day)
# key: (crop_type, planting_date, lat, lon) -> (date, cumulative_gdd)
_runtime_crop_state_cache = {}


IRRIGATION_EFFICIENCY = {
    "drip": 0.90,
    "sprinkler": 0.75,
    "flood": 0.55,
    "unknown": 0.70,
}

NON_ACTUATOR_IRRIGATION_TYPES = {
    "furrow",
    "gravity",
    "manual",
    "natural_rain",
    "none",
    "rainfed",
    "rain-fed",
}

log = logging.getLogger(__name__)


def _runtime_state_fail(plot, reason: str, exc: Exception = None):
    try:
        setattr(plot, 'runtime_crop_state_error', reason)
    except (AttributeError, TypeError):
        pass
    if exc is None:
        log.warning("Runtime crop-state unavailable: %s", reason)
    else:
        log.warning("Runtime crop-state unavailable: %s",
                    reason, exc_info=True)
    return None


def _build_runtime_farm_config(plot):
    """Build a minimal farm-like config object for crop_model helpers."""
    initial_gdd = getattr(plot, 'initial_gdd', 0.0)
    return SimpleNamespace(
        crop_type=getattr(plot, 'crop_type', 'generic'),
        soil_texture_class=getattr(plot, 'soil_texture_class', None),
        planting_date=getattr(plot, 'planting_date', ''),
        initial_gdd=float(initial_gdd or 0.0),
    )


def _resolve_plot_area_m2(plot):
    area_m2 = getattr(plot, 'plot_area_m2', None)
    if area_m2 is None:
        field_area_ha = getattr(plot, 'field_area_ha', None)
        try:
            field_area_ha = float(
                field_area_ha) if field_area_ha is not None else None
        except (TypeError, ValueError):
            field_area_ha = None
        if field_area_ha is not None and field_area_ha > 0:
            area_m2 = field_area_ha * 10000.0

    try:
        area_m2 = float(area_m2) if area_m2 is not None else None
    except (TypeError, ValueError):
        area_m2 = None

    if area_m2 is not None and area_m2 > 0:
        return area_m2
    return None


def _resolve_irrigation_type(plot):
    irrigation_type = getattr(plot, 'irrigation_type', None)
    if irrigation_type is None:
        irrigation_type = getattr(plot, 'Irrigation_type', None)
    irrigation_type = str(irrigation_type or '').strip().lower()
    return irrigation_type or 'unknown'


def _resolve_irrigation_efficiency(plot):
    return IRRIGATION_EFFICIENCY.get(
        _resolve_irrigation_type(plot), IRRIGATION_EFFICIENCY['unknown'])


def _has_actuator_support(plot):
    irrigation_type = _resolve_irrigation_type(plot)
    if irrigation_type in NON_ACTUATOR_IRRIGATION_TYPES:
        return False

    flow_ids = getattr(plot, 'device_and_sensor_ids_flow', []) or []
    return len(flow_ids) > 0


def resolve_irrigation_mode(plot):
    """Return an explicit mode, preserving legacy actuator behavior once."""
    configured = str(getattr(plot, "irrigation_mode", "") or "").strip().lower()
    if configured in MODES:
        return configured
    # Legacy JSON did not distinguish method from operating mode. Existing
    # actuator-backed installations behaved automatically; non-actuator plots
    # were advisory-only.
    return "automatic" if _has_actuator_support(plot) else "advisory_only"


def _recommendation_operation(plot, pipeline_result, recommendation, runtime_state,
                              current_tension, threshold, satellite_validation):
    """Persist one idempotent operational record per model forecast cycle."""
    stable_id = getattr(plot, "stable_id", None)
    if not stable_id:
        return None, True
    forecast_timestamps = list(getattr(pipeline_result, "forecast_timestamps", []) or [])
    anchor = forecast_timestamps[0] if forecast_timestamps else getattr(
        recommendation, "first_breach_timestamp", None)
    if hasattr(anchor, "isoformat"):
        anchor = anchor.isoformat()
    if not anchor:
        cadence = max(1, int(float(getattr(plot, "predict_period_hours", 3) or 3)))
        anchor = pd.Timestamp.now(tz="UTC").floor(f"{cadence}h").isoformat()
    amount = _resolve_runtime_irrigation_volume(plot, runtime_state)
    mode = resolve_irrigation_mode(plot)
    should_irrigate = bool(getattr(recommendation, "should_irrigate", False))
    status = "pending_approval" if should_irrigate and mode == "approval_required" else "planned"
    decision = {
        "urgency": getattr(recommendation, "urgency", "unknown"),
        "should_irrigate": should_irrigate,
        "first_breach_horizon": getattr(recommendation, "first_breach_horizon", None),
        "first_breach_timestamp": getattr(recommendation, "first_breach_timestamp", None),
        "current_tension_cbar": current_tension,
        "threshold_cbar": threshold,
        "satellite_validation": satellite_validation,
    }
    key_material = _json_for_key({"plot": stable_id, "anchor": anchor,
                                  "urgency": decision["urgency"], "amount": amount})
    key = "recommendation:" + hashlib.sha256(key_material.encode()).hexdigest()[:32]
    store = get_operations_store()
    operation, created = store.create_operation(
        idempotency_key=key, plot_id=stable_id, farm_id=getattr(plot, "farm_id", None),
        plot_name=getattr(plot, "user_given_name", ""), source="recommendation",
        mode=mode, status=status, amount_m3=amount, planned_start=(
            getattr(recommendation, "first_breach_timestamp", None).isoformat()
            if hasattr(getattr(recommendation, "first_breach_timestamp", None), "isoformat") else None),
        recommendation=decision)
    urgency = decision["urgency"]
    alert_key = f"alert:{key}"
    store.record_alert(
        idempotency_key=alert_key, operation_id=operation["operation_id"],
        farm_id=getattr(plot, "farm_id", None), plot_id=stable_id,
        urgency=urgency, payload=decision)
    return operation, created


def _json_for_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _transition_operation(operation, status, detail=None):
    if operation is None:
        return None
    try:
        return get_operations_store().transition(
            operation["operation_id"], status, detail)[0]
    except (KeyError, ValueError) as exc:
        log.warning("Operation transition failed for %s: %s",
                    operation.get("operation_id"), exc)
        return operation


def execute_operation_command(plot, operation):
    """Issue one command for a persisted operation and record the outcome."""
    if operation is None:
        return None
    if operation.get("status") not in {"planned", "approved"}:
        return None
    amount = operation.get("amount_m3")
    if amount is None:
        _transition_operation(operation, "failed", {"error": "missing_irrigation_amount"})
        return None
    response = irrigate_amount(plot, float(amount), authorized=True)
    if response is True:
        _transition_operation(operation, "active", {"command": "accepted"})
    else:
        _transition_operation(operation, "failed", {"error": "actuator_command_failed"})
    return response


def _sum_rainfall_mm(frame):
    if frame is None or frame.empty or "Rain" not in frame.columns:
        return 0.0

    rain = pd.to_numeric(frame["Rain"], errors="coerce").fillna(0.0)
    rain = rain.clip(lower=0.0)
    return float(rain.sum())


def _build_rainfall_summary(plot, weather=None, forecast_weather=None):
    rain_since_planting_mm = 0.0
    rain_last_24h_mm = 0.0
    rain_forecast_mm = 0.0

    if weather is not None and not weather.empty:
        rain_since_planting_mm = _sum_rainfall_mm(weather)
        latest_index = weather.index.max()
        if isinstance(latest_index, pd.Timestamp):
            recent_weather = weather[weather.index >=
                                     latest_index - timedelta(days=1)]
            rain_last_24h_mm = _sum_rainfall_mm(recent_weather)

    if forecast_weather is not None:
        look_ahead_hours = float(getattr(plot, 'look_ahead_time', 24) or 24)
        forecast_window = forecast_weather
        if isinstance(forecast_weather, pd.DataFrame) and isinstance(
                forecast_weather.index, pd.DatetimeIndex):
            now = pd.Timestamp.now(tz=forecast_weather.index.tz) if forecast_weather.index.tz else pd.Timestamp.now()
            forecast_window = forecast_weather[
                (forecast_weather.index > now)
                & (forecast_weather.index <= now + pd.Timedelta(hours=look_ahead_hours))]
        rain_forecast_mm = _sum_rainfall_mm(forecast_window)
    else:
        try:
            lat, lon = _parse_plot_coordinates(plot)
            today = datetime.now().date()
            look_ahead_hours = float(getattr(plot, 'look_ahead_time', 24) or 24)
            forecast_days = max(1, int(np.ceil(look_ahead_hours / 24.0)))
            start_date = today.strftime("%Y-%m-%d")
            end_date = (today + timedelta(days=forecast_days - 1)).strftime("%Y-%m-%d")
            forecast_weather = fetch_weather_frame(
                lat, lon, start_date=start_date, end_date=end_date)
            rain_forecast_mm = _sum_rainfall_mm(forecast_weather)
        except (ValueError, TypeError, AttributeError, requests.RequestException):
            rain_forecast_mm = 0.0

    return {
        "rain_since_planting_mm": float(rain_since_planting_mm),
        "rain_last_24h_mm": float(rain_last_24h_mm),
        "rain_forecast_mm": float(rain_forecast_mm),
    }


def _summarize_data_frame(frame: pd.DataFrame, expected_fields):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {
            "available_fields": [],
            "missing_fields": list(expected_fields),
            "rows": 0,
            "start_timestamp": None,
            "end_timestamp": None,
            "latest_timestamp": None,
            "cadence_hours": None,
            "cadence_label": "unknown",
            "field_values": {},
        }

    available_fields = []
    missing_fields = []
    field_values = {}
    for field in expected_fields:
        if field in frame.columns and pd.to_numeric(frame[field], errors="coerce").notna().any():
            available_fields.append(field)
            series = pd.to_numeric(frame[field], errors="coerce")
            last_valid = series.dropna(
            ).iloc[-1] if not series.dropna().empty else None
            if last_valid is None:
                field_values[field] = None
            else:
                field_values[field] = round(float(last_valid), 4)
        else:
            missing_fields.append(field)
            field_values[field] = None

    start_timestamp = frame.index.min()
    end_timestamp = frame.index.max()
    latest_timestamp = frame.index.max()

    cadence_hours = None
    cadence_label = "unknown"
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index) > 1:
        ordered_index = pd.DatetimeIndex(frame.index).sort_values().unique()
        if len(ordered_index) > 1:
            deltas = pd.Series(ordered_index[1:] - ordered_index[:-1])
            if not deltas.empty:
                cadence_hours = round(
                    float(deltas.dt.total_seconds().median() / 3600.0), 2)
                if cadence_hours <= 2:
                    cadence_label = "hourly"
                elif 20 <= cadence_hours <= 28:
                    cadence_label = "daily"
                else:
                    cadence_label = "scene-based"

    def _iso_or_none(value):
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return None

    return {
        "available_fields": available_fields,
        "missing_fields": missing_fields,
        "rows": int(len(frame)),
        "start_timestamp": _iso_or_none(start_timestamp),
        "end_timestamp": _iso_or_none(end_timestamp),
        "latest_timestamp": _iso_or_none(latest_timestamp),
        "cadence_hours": cadence_hours,
        "cadence_label": cadence_label,
        "field_values": field_values,
    }


def _parse_plot_coordinates(plot):
    """Parse GPS values from dict."""
    gps_info = getattr(plot, 'gps_info', {})
    if isinstance(gps_info, dict):
        lat = gps_info.get('latitude', gps_info.get('lattitude'))
        lon = gps_info.get('longitude')
        if lat is None or lon is None:
            raise ValueError('Missing plot coordinates')
        return float(lat), float(lon)

    raise ValueError('Cannot parse plot coordinates')


def _compute_satellite_validation(plot, runtime_state, current_tension, tension_forecast):
    if runtime_state is None or not HAS_CROP_MODEL:
        return None

    try:
        lat, lon = _parse_plot_coordinates(plot)
    except (ValueError, AttributeError, TypeError):
        return None

    now_ts = pd.Timestamp.now(tz="UTC")
    try:
        # Request the available catalog; source cadence, not a crop-day window,
        # determines whether each observation is fresh enough to influence Kc.
        satellite_history = fetch_satellite_history(
            lat,
            lon,
            as_of=now_ts,
        )
    except Exception as exc:
        log.warning("Satellite validation history fetch failed: %s", exc)
        satellite_history = pd.DataFrame()

    return check_satellite_tension_consistency(
        crop_type=runtime_state.crop_type,
        growth_stage=getattr(runtime_state, "growth_stage", get_growth_stage(
            runtime_state.gdd_cumulative, runtime_state.crop_type)),
        gdd_cumulative=runtime_state.gdd_cumulative,
        current_tension=float(current_tension),
        stress_threshold_cbar=float(runtime_state.stress_threshold_cbar),
        tension_forecast=dict(tension_forecast or {}),
        satellite_history=satellite_history,
        satellite_ndvi=getattr(runtime_state, "sat_ndvi", float("nan")),
        satellite_ndre=getattr(runtime_state, "sat_ndre", float("nan")),
        satellite_ndvi_age_hours=getattr(
            runtime_state, "sat_ndvi_age_hours", float("inf")),
        satellite_ndre_age_hours=getattr(
            runtime_state, "sat_ndre_age_hours", float("inf")),
        satellite_ndvi_quality=getattr(
            runtime_state, "sat_ndvi_quality", None),
        satellite_ndre_quality=getattr(
            runtime_state, "sat_ndre_quality", None),
        satellite_data_age_hours=getattr(
            runtime_state, "sat_data_age_hours", float("inf")),
        satellite_vv_db=getattr(runtime_state, "sat_vv_db", float("nan")),
        et0_today_mm=getattr(runtime_state, "et0_today_mm", None),
        et0_baseline_mm=getattr(runtime_state, "et0_baseline_mm", None),
        et0_std_mm=getattr(runtime_state, "et0_std_mm", None),
    )


def _compute_runtime_crop_state(plot):
    """Fetch weather once and build the runtime crop state for actuation."""
    if not HAS_CROP_MODEL:
        return _runtime_state_fail(plot, 'crop_model_not_available')

    planting_date_str = getattr(plot, 'planting_date', '')
    crop_type = getattr(plot, 'crop_type', 'generic')
    if not planting_date_str:
        return _runtime_state_fail(plot, 'missing_planting_date')

    try:
        farm_cfg = _build_runtime_farm_config(plot)
        planting_date = pd.Timestamp(planting_date_str)
        lat, lon = _parse_plot_coordinates(plot)
    except (ValueError, AttributeError, TypeError):
        return _runtime_state_fail(plot, 'invalid_crop_or_coordinates')

    cache_key = (crop_type, planting_date_str, lat, lon)
    today = datetime.now().date()
    if cache_key in _runtime_crop_state_cache:
        cached_date, cached_state = _runtime_crop_state_cache[cache_key]
        if cached_date == today:
            try:
                computed_at = getattr(cached_state, "_computed_at_utc", None)
                if computed_at is not None:
                    now_ts = pd.Timestamp(pd.Timestamp.now(
                        tz="UTC").to_pydatetime().replace(tzinfo=None))
                    delta_hours = (now_ts - pd.Timestamp(computed_at)
                                   ).total_seconds() / 3600.0
                    if delta_hours > 0:
                        for attr in ("weather_data_age_hours", "sat_data_age_hours", "sat_ndvi_age_hours"):
                            current_age = getattr(cached_state, attr, None)
                            try:
                                current_age = float(current_age)
                            except (TypeError, ValueError):
                                continue
                            if np.isfinite(current_age):
                                setattr(cached_state, attr,
                                        current_age + delta_hours)
            except Exception:
                pass
            return cached_state

    try:
        start_str = planting_date.strftime("%Y-%m-%d")
        end_str = today.strftime("%Y-%m-%d")
        days_ago = (today - planting_date.date()).days

        weather = fetch_weather_frame(
            lat, lon, start_date=start_str, end_date=end_str)
        if weather.empty:
            return _runtime_state_fail(
                plot,
                f'no_weather_for_range:{start_str}:{end_str}'
            )

        weather_summary = _summarize_data_frame(weather, WEATHER_FIELDS)

        weather.index = pd.DatetimeIndex(
            pd.to_datetime(weather.index)).tz_localize(None)
        weather = weather.sort_index()

        now_ts = pd.Timestamp(pd.Timestamp.now(
            tz="UTC").to_pydatetime().replace(tzinfo=None))
        if len(weather.index) > 0:
            past_weather = weather.index[weather.index <= now_ts]
            latest_weather_ts = past_weather.max() if len(
                past_weather) > 0 else weather.index.max()
        else:
            latest_weather_ts = None
        weather_age_hours = float("inf")
        if latest_weather_ts is not None:
            weather_age_hours = (
                now_ts - latest_weather_ts
            ).total_seconds() / 3600.0
            if weather_age_hours < 0:
                weather_age_hours = 0.0

        if "Temperature" in weather.columns:
            temp_series = pd.to_numeric(
                weather["Temperature"], errors="coerce")
        else:
            temp_series = pd.Series(dtype=float)

        if temp_series.empty or temp_series.dropna().empty:
            raise ValueError(
                "SpaceIoTBox weather response did not contain temperature data")

        # SpaceIoTBox returns hourly forecast data; GDD expects daily Tmax/Tmin.
        daily_tmax = temp_series.resample(
            "D").max().interpolate(limit_direction="both")
        daily_tmin = temp_series.resample(
            "D").min().interpolate(limit_direction="both")

        if "Et0_evapotranspiration" in weather.columns:
            et0 = pd.to_numeric(
                weather["Et0_evapotranspiration"], errors="coerce")
        else:
            et0 = pd.Series(dtype=float)

        if et0.empty or et0.dropna().empty:
            et0_daily = pd.Series(0.0, index=daily_tmax.index, dtype=float)
        else:
            et0_daily = et0.resample("D").sum(min_count=1).fillna(0.0)

        gdd_series = compute_gdd_from_weather(daily_tmax, daily_tmin, farm_cfg)
        current_gdd = gdd_series.iloc[-1] if len(gdd_series) > 0 else 0.0

        satellite = fetch_satellite_snapshot(
            lat,
            lon,
            as_of=pd.Timestamp.now(tz="UTC"),
        )
        ndvi = float("nan")
        ndre = float("nan")
        ndvi_age_hours = float("inf")
        vv_db = float("nan")
        sat_age_hours = float("inf")
        ndre_age_hours = float("inf")
        ndvi_quality = 0.0
        ndre_quality = 0.0
        if isinstance(satellite, pd.DataFrame) and not satellite.empty:
            latest_sat = satellite.iloc[-1]
            ndvi = float(latest_sat.get("sat_ndvi", float("nan")))
            ndre = float(latest_sat.get("sat_ndre", float("nan")))
            ndvi_age_hours = float(latest_sat.get(
                "sat_ndvi_age", latest_sat.get("satellite_data_age", float("inf"))))
            vv_db = float(latest_sat.get("sat_vv_db", float("nan")))
            sat_age_hours = float(latest_sat.get(
                "satellite_data_age", float("inf")))
            ndre_age_hours = float(latest_sat.get("sat_ndre_age", float("inf")))
            ndvi_quality = float(latest_sat.get("sat_ndvi_quality", 0.0))
            ndre_quality = float(latest_sat.get("sat_ndre_quality", 0.0))

        satellite_summary = _summarize_data_frame(satellite, SATELLITE_FIELDS)

        state = get_crop_state(
            farm_cfg,
            current_gdd,
            ndvi=ndvi,
            ndre=ndre,
            ndvi_age_hours=ndvi_age_hours,
            ndre_age_hours=ndre_age_hours,
            ndvi_quality=ndvi_quality,
            ndre_quality=ndre_quality,
            etc_daily_mm=0.0,
        )
        state.sat_ndvi = ndvi
        state.sat_ndre = ndre
        state.sat_ndvi_age_hours = ndvi_age_hours
        state.sat_vv_db = vv_db
        state.sat_data_age_hours = sat_age_hours
        state.sat_ndre_age_hours = ndre_age_hours
        state.sat_ndvi_quality = ndvi_quality
        state.sat_ndre_quality = ndre_quality
        state.weather_data_age_hours = weather_age_hours
        # Preserve source selection for API diagnostics and production audits.
        state.weather_provider = weather.attrs.get("provider", "unknown")
        state.satellite_provider = satellite.attrs.get(
            "provider", "unavailable")
        state.weather_data_summary = weather_summary
        state.satellite_data_summary = satellite_summary
        try:
            forecast_days = max(1, int(np.ceil(float(
                getattr(plot, "forecast_horizon_days", 5) or 5))))
            forecast_weather = fetch_weather_frame(
                lat, lon, start_date=today.strftime("%Y-%m-%d"),
                end_date=(today + timedelta(days=forecast_days)).strftime("%Y-%m-%d"))
            if not isinstance(forecast_weather, pd.DataFrame):
                forecast_weather = pd.DataFrame()
        except Exception as exc:
            log.warning("Forecast weather fetch failed: %s", exc)
            forecast_weather = pd.DataFrame()
        state.weather_forecast_frame = forecast_weather
        historical_et0 = et0_daily.loc[:latest_weather_ts] if latest_weather_ts is not None else et0_daily
        historical_et0 = pd.to_numeric(
            historical_et0, errors="coerce").dropna()
        if len(historical_et0) > 0:
            state.et0_today_mm = float(historical_et0.iloc[-1])
            baseline_window = historical_et0.tail(min(len(historical_et0), 8))
            if len(baseline_window) > 1:
                baseline_series = baseline_window.iloc[:-1]
            else:
                baseline_series = baseline_window
            state.et0_baseline_mm = float(baseline_series.mean()) if len(
                baseline_series) > 0 else None
            state.et0_std_mm = float(baseline_series.std(
                ddof=0)) if len(baseline_series) > 1 else 0.0
        state.satellite_validation = getattr(
            plot, "satellite_validation", None)
        state._computed_at_utc = pd.Timestamp(pd.Timestamp.now(
            tz="UTC").to_pydatetime().replace(tzinfo=None))
        et0_today = float(et0_daily.iloc[-1]) if len(et0_daily) > 0 else 0.0
        state.etc_daily_mm = float(max(0.0, et0_today * state.kc))

        rainfall_summary = _build_rainfall_summary(plot, weather, forecast_weather)
        state.rain_since_planting_mm = rainfall_summary["rain_since_planting_mm"]
        state.rain_last_24h_mm = rainfall_summary["rain_last_24h_mm"]
        state.rain_forecast_mm = rainfall_summary["rain_forecast_mm"]
        kc_gdd_series = compute_kc_gdd_series(gdd_series, crop_type)
        state.cumulative_etc_mm = float(
            max(0.0, (et0_daily * kc_gdd_series).sum()))
        state.historical_rain_credit_mm = float(
            min(
                max(0.0, state.rain_since_planting_mm - state.cumulative_etc_mm),
                state.etc_daily_mm,
            )
        )
        state.rain_effective_mm = float(
            max(0.0, state.rain_last_24h_mm + state.rain_forecast_mm)
        )
        state.recommended_volume_mm = float(
            max(
                0.0,
                state.etc_daily_mm
                - state.historical_rain_credit_mm
                - state.rain_effective_mm,
            )
        )

        area_m2 = _resolve_plot_area_m2(plot)
        if area_m2 is not None:
            efficiency = _resolve_irrigation_efficiency(plot)
            state.recommended_volume_m3 = float(
                state.recommended_volume_mm * area_m2 / (efficiency * 1000.0)
            )

        _runtime_crop_state_cache[cache_key] = (today, state)
        try:
            setattr(plot, 'runtime_crop_state_error', '')
        except (AttributeError, TypeError):
            pass
        return state
    except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
        return _runtime_state_fail(plot, 'runtime_crop_state_exception', exc)


def get_runtime_crop_state(plot):
    """Return the cached or freshly computed runtime crop state."""
    return _compute_runtime_crop_state(plot)

# Function to find next lower and higher value occurrence
def find_next_occurrences(df, column, threshold, timeSpanOverThreshold):
    timezone = str(df.index.tz) if isinstance(df.index, pd.DatetimeIndex) and df.index.tz else "UTC"

    # Start @current time
    # timezone = create_model.get_timezone(Current_config["Gps_info"]["latitude"], Current_config["Gps_info"]["longitude"])
    # TODO: timezone is missing here, replace with timezone, uncomment above
    idx = pd.Timestamp(datetime.now().replace(
        microsecond=0)).tz_localize(timezone)

    # Filter the DataFrame to include only rows with indices greater than or equal to 'idx'
    filtered_df = df[df.index >= idx]
    filtered_df = df[df.index <= idx + timedelta(hours=timeSpanOverThreshold)]

    # Further filter the DataFrame to include only rows where the specified column's value is less than the 'threshold'
    filtered_lower = filtered_df[filtered_df[column] < threshold]

    # Convert the filtered DataFrame's index to a list
    next_lower_idx = filtered_lower.index.tolist()

    # Take first occurrence from list
    if next_lower_idx:
        next_lower_idx = next_lower_idx[0]
    else:
        next_lower_idx = None

    # Find the next occurrence of a value higher than the threshold after the next lower index
    if next_lower_idx is not None:
        # Filter the DataFrame to include only rows with indices greater than or equal to 'next_lower_idx'
        filtered_df_higher = df[df.index >= next_lower_idx]

        # Further filter the DataFrame to include only rows where the specified column's value is greater than the 'threshold'
        filtered_higher = filtered_df_higher[filtered_df_higher[column] > threshold]

        # Convert the filtered DataFrame's index to a list
        next_higher_idx = filtered_higher.index.tolist()

        # Take first occurrence from list
        if next_higher_idx:
            next_higher_idx = next_higher_idx[0]
        else:
            next_higher_idx = None
    # Consequently if there is no occurrence of lower, just take first one from input data
    else:
        next_higher_idx = df.index[0]

    return next_lower_idx, next_higher_idx


def _resolve_runtime_irrigation_volume(plot, crop_state):
    if crop_state is None:
        return None

    if getattr(crop_state, 'recommended_volume_m3', None) is not None:
        try:
            volume_m3 = float(crop_state.recommended_volume_m3)
        except (TypeError, ValueError):
            volume_m3 = None
        if volume_m3 is not None and np.isfinite(volume_m3) and volume_m3 > 0:
            return volume_m3

    volume_mm = getattr(crop_state, "recommended_volume_mm", None)
    if volume_mm is None:
        volume_mm = getattr(crop_state, "etc_daily_mm", None)
    if volume_mm is None:
        return None

    try:
        volume_mm = float(volume_mm)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(volume_mm) or volume_mm <= 0:
        return None

    area_m2 = _resolve_plot_area_m2(plot)
    if area_m2 is None:
        return None

    efficiency = _resolve_irrigation_efficiency(plot)
    return float(volume_mm) * area_m2 / (efficiency * 1000.0)


def get_irrigation_recommendation(plot):
    # Prefer the crop state already used by orchestration; compute one only when
    # the API is called before the first pipeline cycle.
    pipeline_result = getattr(plot, "pipeline_result", None)
    runtime_state = getattr(pipeline_result, "crop_state", None)
    if runtime_state is None:
        runtime_state = _compute_runtime_crop_state(plot)
    if runtime_state is None:
        return {
            "available": False,
            "reason": "Crop state could not be computed.",
        }

    def _finite_or_none(value, decimals=None):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        if decimals is not None:
            return round(value, decimals)
        return value

    weather_age = getattr(
        runtime_state, "weather_data_age_hours", float("inf"))
    sat_age = getattr(runtime_state, "sat_data_age_hours", float("inf"))
    weather_stale = bool(weather_age > WEATHER_STALE_HOURS)
    # Each index reports cadence-derived quality; there is no crop-day expiry.
    quality_values = [
        _finite_or_none(getattr(runtime_state, "sat_ndvi_quality", None)),
        _finite_or_none(getattr(runtime_state, "sat_ndre_quality", None)),
    ]
    sat_quality = max((value for value in quality_values if value is not None), default=0.0)
    sat_stale = bool(sat_quality <= 0.0)
    stale_sources = []
    if weather_stale:
        stale_sources.append("weather")
    if sat_stale:
        stale_sources.append("satellite")

    area_m2 = _resolve_plot_area_m2(plot)
    irrigation_type = _resolve_irrigation_type(plot)
    efficiency = _resolve_irrigation_efficiency(plot)
    actuator_based = (_has_actuator_support(plot)
                      and resolve_irrigation_mode(plot) != "advisory_only")
    recommended_volume_m3 = _resolve_runtime_irrigation_volume(
        plot, runtime_state)

    base = {
        "available": True,
        "actuator_based": actuator_based,
        "irrigation_mode": resolve_irrigation_mode(plot),
        "crop_type": runtime_state.crop_type,
        "growth_stage": runtime_state.growth_stage_name,
        "kc": round(float(runtime_state.kc), 3) if getattr(runtime_state, "kc", None) is not None else None,
        "etc_daily_mm": round(float(runtime_state.etc_daily_mm), 2),
        "recommended_depth_mm": round(float(getattr(runtime_state, "recommended_volume_mm", runtime_state.etc_daily_mm)), 2),
        "recommended_volume_m3": round(float(recommended_volume_m3), 3) if recommended_volume_m3 is not None else None,
        "volume_available": recommended_volume_m3 is not None,
        "irrigation_type": irrigation_type,
        "efficiency": round(float(efficiency), 2),
        "plot_area_m2": round(float(area_m2), 1) if area_m2 is not None else None,
        "threshold_cbar": round(float(runtime_state.stress_threshold_cbar), 1),
        "gdd_cumulative": round(float(runtime_state.gdd_cumulative), 1),
        "rain_since_planting_mm": round(float(getattr(runtime_state, "rain_since_planting_mm", 0.0)), 2),
        "rain_last_24h_mm": round(float(getattr(runtime_state, "rain_last_24h_mm", 0.0)), 2),
        "rain_forecast_mm": round(float(getattr(runtime_state, "rain_forecast_mm", 0.0)), 2),
        "historical_rain_credit_mm": round(float(getattr(runtime_state, "historical_rain_credit_mm", 0.0)), 2),
        "cumulative_etc_mm": round(float(getattr(runtime_state, "cumulative_etc_mm", 0.0)), 2),
        "net_irrigation_need_mm": round(float(getattr(runtime_state, "recommended_volume_mm", runtime_state.etc_daily_mm)), 2),
        "sat_ndvi": (None if pd.isna(getattr(runtime_state, "sat_ndvi", float("nan")))
                     else round(float(runtime_state.sat_ndvi), 3)),
        "sat_ndre": (None if pd.isna(getattr(runtime_state, "sat_ndre", float("nan")))
                     else round(float(runtime_state.sat_ndre), 3)),
        "sat_ndvi_age_hours": _finite_or_none(getattr(runtime_state, "sat_ndvi_age_hours", float("nan")), 1),
        "sat_ndre_age_hours": _finite_or_none(getattr(runtime_state, "sat_ndre_age_hours", float("nan")), 1),
        "sat_ndvi_quality": _finite_or_none(getattr(runtime_state, "sat_ndvi_quality", float("nan")), 3),
        "sat_ndre_quality": _finite_or_none(getattr(runtime_state, "sat_ndre_quality", float("nan")), 3),
        "sat_vv_db": (None if pd.isna(getattr(runtime_state, "sat_vv_db", float("nan")))
                      else round(float(runtime_state.sat_vv_db), 2)),
        "sat_data_age_hours": _finite_or_none(getattr(runtime_state, "sat_data_age_hours", float("nan")), 1),
        "satellite_validation": getattr(runtime_state, "satellite_validation", None),
        "weather_data_age_hours": _finite_or_none(weather_age, 1),
        "weather_data_summary": getattr(runtime_state, "weather_data_summary", None),
        "satellite_data_summary": getattr(runtime_state, "satellite_data_summary", None),
        "data_requirements": {
            "weather": {
                "expected_interval": "hourly",
                "used_for": ["Temperature", "Rain", "Windspeed", "Winddirection", "Et0_evapotranspiration"],
                "model_processing": [
                    "hourly weather is resampled to daily Tmax/Tmin for GDD",
                    "rainfall is summed over the request window",
                    "ET0 is accumulated as a rolling daily demand term",
                ],
            },
            "satellite": {
                "expected_interval": "scene-based / as available",
                "used_for": ["sat_ndvi", "sat_ndre"],
                "model_processing": [
                    "NDRE is preferred for mid-season Kc when point-valid and fresh",
                    "NDVI covers other stages and is the stale-NDRE fallback",
                    "freshness follows each index's observed acquisition intervals",
                ],
            },
        },
        "data_stale": bool(stale_sources),
        "data_stale_sources": stale_sources,
        "data_stale_thresholds": {
            "weather_hours": WEATHER_STALE_HOURS,
            # Preserve the response key while making clear there is no fixed cutoff.
            "satellite_hours": None,
            "satellite_policy": "empirical_acquisition_intervals",
        },
        "reason": (
            "No pump/actuator configured for this irrigation mode. Advice only."
            if not actuator_based else
            "Rainfall-adjusted irrigation need computed from weather history and forecast."
        ),
    }
    decision = {}
    if pipeline_result is not None:
        pipeline_decision = getattr(pipeline_result, "recommendation", None)
        if pipeline_decision is not None:
            if hasattr(pipeline_result, "to_dict"):
                decision = pipeline_result.to_dict().get("recommendation") or {}
            elif hasattr(pipeline_decision, "__dict__"):
                decision = dict(vars(pipeline_decision))
    contract = compose_recommendation(
        plot=plot, base=base, decision=decision, pipeline=pipeline_result,
        forecast_weather=getattr(runtime_state, "weather_forecast_frame", None))
    # Keep v1 flat keys for existing integrations while all first-party UI
    # consumers move to the schema-versioned nested contract.
    return {**base, **decision, **contract, "decision": decision}


def _alert_paths(plot_id):
    base_dir = os.path.join("data", "alerts")
    base_name = f"plot_{plot_id}"
    return (
        os.path.join(base_dir, f"{base_name}.jsonl"),
        os.path.join(base_dir, f"{base_name}.latest.json"),
    )


def _persist_alert_record(
    plot,
    recommendation,
    tension_forecast,
    threshold,
    current_tension,
    satellite_validation=None,
    runtime_state=None,
):
    if recommendation is None:
        return

    try:
        plot_id = getattr(plot, "id", None)
        plot_name = getattr(plot, "user_given_name", "")
        now_utc = pd.Timestamp.now(tz="UTC").floor("s").isoformat()

        validation_factors = {}
        if isinstance(satellite_validation, dict):
            validation_factors = dict(
                satellite_validation.get("factors") or {})

        should_irrigate = bool(
            getattr(recommendation, "should_irrigate", False))
        urgency = getattr(recommendation, "urgency", "unknown")
        irrigation_event_occurred = should_irrigate or urgency == "critical"

        payload = {
            "timestamp_utc": now_utc,
            "plot_id": plot_id,
            "plot_name": plot_name,
            "urgency": urgency,
            "should_irrigate": should_irrigate,
            "irrigation_event_occurred": irrigation_event_occurred,
            "first_breach_horizon": getattr(recommendation, "first_breach_horizon", None),
            "first_breach_timestamp": (
                recommendation.first_breach_timestamp.isoformat()
                if getattr(recommendation, "first_breach_timestamp", None) is not None
                else None
            ),
            "current_tension": float(current_tension) if current_tension is not None else None,
            "stress_threshold": float(threshold) if threshold is not None else None,
            "growth_stage": getattr(runtime_state, "growth_stage", None),
            "growth_stage_name": getattr(runtime_state, "growth_stage_name", None),
            "gdd_cumulative": getattr(runtime_state, "gdd_cumulative", None),
            "stage_sensitivity": validation_factors.get("stage_sensitivity"),
            "et0_adjustment": validation_factors.get("et0_adjustment"),
            "et0_today_mm": getattr(runtime_state, "et0_today_mm", None),
            "et0_baseline_mm": getattr(runtime_state, "et0_baseline_mm", None),
            "et0_std_mm": getattr(runtime_state, "et0_std_mm", None),
            "rain_last_24h_mm": getattr(runtime_state, "rain_last_24h_mm", None),
            "rain_since_planting_mm": getattr(runtime_state, "rain_since_planting_mm", None),
            "sat_ndvi": getattr(runtime_state, "sat_ndvi", None),
            "sat_ndre": getattr(runtime_state, "sat_ndre", None),
            "sat_vv_db": getattr(runtime_state, "sat_vv_db", None),
            "forecast_summary": dict(getattr(recommendation, "forecast_summary", {}) or {}),
            "breach_horizons": list(getattr(recommendation, "breach_horizons", []) or []),
            "tension_forecast": dict(tension_forecast or {}),
            "satellite_validation": satellite_validation,
            "inference_source": getattr(plot, "_inference_source", "live"),
        }

        jsonl_path, latest_path = _alert_paths(plot_id)
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        with open(jsonl_path, "a") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")

        with open(latest_path, "w") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        log.warning("Failed to persist alert record: %s", exc)

# Function to read existing data from the JSON file


def read_data_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            return json.load(json_file)
    else:
        return {"irrigations": []}

# Function to save data to the JSON file


def save_data_to_file(filename, data):
    if not os.path.exists(filename):
        print(f"{filename} does not exist, creating a new one.")

    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Function to add a new record


def add_record(data, timestamp, amount, status="not confirmed"):
    record = {
        "timestamp": timestamp,
        "amount": amount,
        "status": status
    }
    data["irrigations"].append(record)

    return data

# Function to round to the nearest 10 minutes


def round_to_nearest_10_minutes(dt):
    # Truncate milliseconds and seconds
    dt = dt.replace(microsecond=0, second=0)

    # Calculate the number of minutes to add to round to the nearest 10 minutes
    add_minutes = 10 - (dt.minute %
                        10) if dt.minute % 10 >= 5 else -(dt.minute % 10)
    dt += timedelta(minutes=add_minutes)

    return dt

# to json file -> not needed because can just ask api, more consistent state


def save_irrigation_time(amount, plot, status="not confirmed") -> int:
    # Load from file
    filename = 'data/irrigations_plot_' + str(plot.id) + '.json'
    data = read_data_from_file(filename)

    # obtain timezone
    timezone = pytz.timezone(TimeUtils.for_plot(plot))
    # add to current timestamp without converting it
    now = datetime.now(tz=timezone)

    # Round to the nearest 10 minutes
    rounded_tz = round_to_nearest_10_minutes(now)

    # Add new records
    data = add_record(data, str(rounded_tz), amount, status)

    # Save updated data back to the JSON file
    save_data_to_file(filename, data)

    print("Irrigation time has been saved to: ", filename)

    return 0


def update_irrigation_status(plot, status="not_confirmed"):
    # Load from file
    filename = 'data/irrigations_plot_' + str(plot.id) + '.json'
    data = read_data_from_file(filename)

    # Update the status of the last irrigation record
    if data["irrigations"]:
        data["irrigations"][-1]["status"] = status
        save_data_to_file(filename, data)
        print(f"Irrigation status updated to '{status}' for plot {plot.id}.")
    else:
        print(f"No irrigation records found for plot {plot.id} to update.")


def verify_irrigation(plot, amount):
    """
    Verify irrigation completed correctly by checking the flow meter
    confirmation sensor. Called by a Timer ~3 hours after irrigation.

    Uses per-plot retry tracking (_irrigation_retries dict) to avoid
    cross-plot interference when multiple plots verify concurrently.

    Args:
        plot:   Plot object with id, device_and_sensor_ids_flow_confirmation
        amount: Expected irrigation amount in m³
    """
    plot_id = getattr(plot, 'id', 0)

    def transition_latest(status, detail=None):
        stable_id = getattr(plot, "stable_id", None)
        if not stable_id:
            return
        operation = get_operations_store().latest_for_plot(
            stable_id, statuses={"active", "completed"})
        if operation:
            _transition_operation(operation, status, detail)

    # Get sensor id of confirmation device.
    confirmation = plot.device_and_sensor_ids_flow_confirmation
    if not isinstance(confirmation, list) or len(confirmation) == 0 or not isinstance(confirmation[0], str):
        print(
            f"No confirmation sensor configured for plot {plot.id}, cannot verify irrigation.")
        update_irrigation_status(plot, "no_confirmation_sensor_configured")
        transition_latest("completed", {"verification": "no_confirmation_sensor"})
        _release_active_irrigation(plot_id)
        return
    else:
        sensor_id = confirmation[0]

    # Example API call: curl -X GET "http://192.168.188.29/devices/689dad2768f319076487e4c7/sensors/689db4b868f319076487e500/value" -H "accept: application/json"

    check_url = f"{NetworkUtils.ApiUrl}devices/{sensor_id.split('/')[0]}/sensors/{sensor_id.split('/')[1]}"

    headers = {
        'Authorization': f'Bearer {NetworkUtils.Token}'
    }

    try:
        response = requests.get(check_url, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"Verification failed for plot {plot_id}: "
                  f"HTTP {response.status_code} {response.text}")
            update_irrigation_status(plot, "verification_of_irrigation_failed")
            transition_latest("failed", {"error": f"verification_http_{response.status_code}"})
            _release_active_irrigation(plot_id)
            return

        resp = response.json()
        tz = pytz.timezone(TimeUtils.for_plot(plot))
        last_time = datetime.fromisoformat(
            resp.get('time').replace("Z", "+00:00")).astimezone(tz)
        time_passed = datetime.now(tz=tz) - last_time
        last_value = float(resp.get('value'))

        # Tolerance: 10% relative (min 0.01 m³) within 3 hours of irrigation
        amount = float(amount)
        rel_tol = 0.10
        abs_tol = 0.01
        tolerance = max(abs_tol, rel_tol * abs(amount))
        if abs(amount - last_value) <= tolerance and time_passed <= timedelta(hours=3):
            print(f"Irrigation confirmed for plot {plot_id}: "
                  f"delivered={last_value}m³, expected={amount}m³.")
            update_irrigation_status(plot, "confirmed")
            transition_latest("verified", {"delivered_m3": last_value,
                                             "expected_m3": amount})
            _irrigation_retries[plot_id] = 0
            _release_active_irrigation(plot_id)
        else:
            retries = _irrigation_retries.get(plot_id, 0)
            if retries == 0:
                print(f"Irrigation failed for plot {plot_id}: "
                      f"delivered={last_value}m³, expected={amount}m³. "
                      f"Retrying once.")
                update_irrigation_status(
                    plot, "irrigation failed, retrying once")
                _irrigation_retries[plot_id] = 1
                _release_active_irrigation(plot_id)
                irrigate_amount(plot, amount, authorized=True)
            else:
                update_irrigation_status(
                    plot, "irrigation failed, twice, no more retries")
                transition_latest("failed", {"error": "delivery_mismatch_after_retry",
                                              "delivered_m3": last_value,
                                              "expected_m3": amount})
                _release_active_irrigation(plot_id)
                print(
                    f"Irrigation failed for plot {plot.id}: amount_given: {last_value}m³, expected amount: {amount}m³. Irrigation will not be retried.")
    except requests.exceptions.RequestException as e:
        print(f"Verification request error for plot {plot_id}: {e}")
        update_irrigation_status(plot, "verification_failed_request_error")
        transition_latest("failed", {"error": "verification_request_error"})
        _release_active_irrigation(plot_id)
        print(
            f"Request of verification of irrigation failed for plot {plot.id}: expected amount: {amount}m³. Irrigation will not be retried.")


# Load from wazigate API
# TODO: renew the token, make function in NetworkUtils that does a arbitrary API request
def irrigate_amount(plot, amount=0, authorized=False):
    # Example API call:
    # curl -X POST "http://192.168.189.2/devices/6645c4d468f31971148f2ab1/actuators/6673fcb568f31971148ff5f7/value"
    # -H "accept: */*" -H "Content-Type: application/json" -d "7.2"

    # if there is no amount in arguments, take it from config -> It is automatically triggered, retrieve amount!
    if amount == 0:
        amount = plot.irrigation_amount

    mode = resolve_irrigation_mode(plot)
    if mode in {"approval_required", "manual"} and not authorized:
        log.warning("Irrigation command rejected for plot %s: mode '%s' requires an approved/manual operation",
                    getattr(plot, "id", "?"), mode)
        return None
    if mode == "advisory_only":
        log.warning("Irrigation command rejected for advisory-only plot %s",
                    getattr(plot, "id", "?"))
        return None

    if not _has_actuator_support(plot):
        print(
            f"Irrigation skipped for plot {getattr(plot, 'id', '?')}: "
            f"irrigation type '{_resolve_irrigation_type(plot)}' is advisory-only.")
        return None

    # Name of flow meter sensor to initiate irrigation => TODO: decide on using single or multiple
    flow_ids = getattr(plot, 'device_and_sensor_ids_flow', []) or []
    if len(flow_ids) == 0:
        print(
            f"Irrigation skipped for plot {getattr(plot, 'id', '?')}: no actuator configured.")
        return None

    plot_id = getattr(plot, "id", None)
    # Reserve the plot before the network request to close concurrent races.
    with _irrigation_lock:
        if plot_id in _active_irrigations:
            log.warning("Duplicate irrigation prevented for plot %s", plot_id)
            return None
        _active_irrigations.add(plot_id)

    flow_meter_name = flow_ids[0]

    # API URL
    apiUrl = NetworkUtils.ApiUrl

    # Create URL for API call
    request_url = f"{apiUrl}devices/{flow_meter_name.split('/')[0]}/actuators/{flow_meter_name.split('/')[1]}/value"

    # Define headers for the POST request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {NetworkUtils.Token}'
    }

    # Define the payload
    payload = amount

    try:
        # Send a POST request to the API
        response = requests.post(
            request_url, headers=headers, json=payload, timeout=30)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            response_ok = True
            # The hardware command has already succeeded. A local logging
            # problem must not turn that success into an API failure or prevent
            # delivery verification from being scheduled.
            try:
                save_irrigation_time(amount, plot, status="not confirmed")
            except (OSError, ValueError, TypeError) as exc:
                log.exception(
                    "Irrigation command succeeded but event recording failed for plot %s: %s",
                    plot_id,
                    exc,
                )

            # Confirmation timing comes from the shared farm timing contract.
            timer = threading.Timer(
                runtime_config.get_timing_config(plot).irrigation_confirmation_seconds,
                verify_irrigation, args=[plot, amount])
            timer.name = f"IrrigationCheckRoutine-{plot.id}"
            # A long confirmation delay must not prevent graceful process exit.
            timer.daemon = True
            timer.start()

            response_ok = True
        else:
            print("Irrigation failed for plot")
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.text)
            response_ok = None
    except requests.exceptions.RequestException as e:
        # Handle request exceptions (e.g., connection errors)
        print("Request error:", e)
        response_ok = None  # TODO: introduce error handling

    if response_ok is not True:
        _release_active_irrigation(plot_id)

    return response_ok

def main(
    current_value,
    threshold_timestamp,
    predictions,
    plot
) -> int:
    """
    Layer 3 integration: Uses the decision engine to compare soil tension
    (from Layer 1, environment model) against crop stress threshold
    (from Layer 2, crop model) and execute irrigation if needed.

    :param current_value: Current sensor value (cbar for tension, % for capacitive).
    :param threshold_timestamp: Predicted threshold crossing timestamp.
    :param predictions: Predictions data (list or dataframe).
    :param plot: holds amount and strategy or kind.
    :return: 1 if irrigation is triggered, otherwise 0.
    """

    # Actuation fails closed unless one complete, fresh orchestration result is
    # available; it never recomputes a separate decision state here.
    pipeline_result = getattr(plot, "pipeline_result", None)
    if pipeline_result is None:
        log.warning("Actuation skipped: no completed orchestration result")
        return 0
    runtime_state = getattr(pipeline_result, "crop_state", None)
    if getattr(pipeline_result, "error", None) or getattr(
            pipeline_result, "model_status", "failed") == "failed":
        log.warning("Actuation skipped: pipeline result is invalid")
        return 0
    if runtime_state is None:
        log.warning("Actuation skipped: runtime crop state unavailable")
        return 0

    freshness = getattr(pipeline_result, "data_freshness", {}) or {}
    if any(
        value is None or not np.isfinite(float(value)) or float(value) > limit
        for value, limit in (
            (freshness.get("weather_age_hours"), WEATHER_STALE_HOURS),
        )
    ):
        log.warning("Actuation skipped: weather data is missing/stale")
        return 0
    quality_values = []
    for attribute in ("sat_ndvi_quality", "sat_ndre_quality"):
        try:
            value = float(getattr(runtime_state, attribute, 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            quality_values.append(value)
    satellite_quality = max(quality_values, default=0.0)
    if satellite_quality <= 0.0:
        # Cadence quality already removes stale satellite influence, so the
        # primary soil/weather decision remains available and fail-safe.
        log.info("Satellite refinement unavailable or stale; using GDD-based Kc")
    tension_forecast = dict(getattr(pipeline_result, "tension_forecast", {}) or {})
    if not tension_forecast:
        log.warning("Actuation skipped: tension forecast is missing")
        return 0
    threshold = runtime_state.stress_threshold_cbar
    timeSpanOverThreshold = plot.look_ahead_time

    actuator_supported = _has_actuator_support(plot)

    # Layer 3: Decision engine — compare current/forecast vs threshold
    if (
        HAS_DECISION_ENGINE
        and plot.sensor_kind == "tension"
        and getattr(plot, "_inference_source", "live") == "live"
    ):
        # Prefer the forecast already evaluated by orchestration. This keeps
        # actuation on the model's configured cadence and horizon.
        pipeline_result = getattr(plot, "pipeline_result", None)
        tension_forecast = dict(
            getattr(pipeline_result, "tension_forecast", {}) or {})
        if isinstance(predictions, pd.DataFrame) and 'smoothed_values' in predictions.columns:
            timezone = TimeUtils.for_plot(plot)
            try:
                now = pd.Timestamp(datetime.now().replace(microsecond=0))
                if timezone:
                    now = now.tz_localize(timezone)

                pred_index = predictions.index
                now_cmp = now
                if isinstance(pred_index, pd.DatetimeIndex):
                    if pred_index.tz is None and now.tzinfo is not None:
                        now_cmp = now.tz_localize(None)
                    elif pred_index.tz is not None and now.tzinfo is None:
                        now_cmp = now.tz_localize(pred_index.tz)
                    elif pred_index.tz is not None and now.tzinfo is not None and pred_index.tz != now.tzinfo:
                        now_cmp = now.tz_convert(pred_index.tz)

                future = predictions[predictions.index >= now_cmp]
                max_hours = None
                if isinstance(future.index, pd.DatetimeIndex) and len(future.index) > 0:
                    max_hours = (future.index.max() -
                                 now_cmp).total_seconds() / 3600.0
                if not tension_forecast:
                    for timestamp in future.index:
                        hours = (timestamp - now_cmp).total_seconds() / 3600.0
                        if hours <= 0 or (max_hours is not None and hours > max_hours):
                            continue
                        val = future.loc[timestamp, 'smoothed_values']
                        if isinstance(val, pd.Series):
                            val = val.iloc[-1]
                        tension_forecast[f"{hours:.3f}h"] = float(val)
            except (TypeError, ValueError, KeyError, IndexError) as forecast_error:
                print(
                    f"  [Decision Engine] Forecast extraction failed: {forecast_error}"
                )

        recommendation = evaluate_forecast(
            current_tension=float(current_value),
            tension_forecast=tension_forecast,
            stress_threshold=float(threshold),
            current_timestamp=datetime.now(),
            advise_horizon_hours=float(timeSpanOverThreshold),
        )

        satellite_validation = _compute_satellite_validation(
            plot=plot,
            runtime_state=runtime_state,
            current_tension=float(current_value),
            tension_forecast=tension_forecast,
        )
        if satellite_validation is not None:
            runtime_state.satellite_validation = satellite_validation
            try:
                setattr(plot, "satellite_validation", satellite_validation)
            except (AttributeError, TypeError):
                pass

        print(f"  [Decision Engine] {plot.user_given_name}: "
              f"tension={current_value:.1f}, threshold={threshold:.1f}, "
              f"urgency={recommendation.urgency}, "
              f"breaches={recommendation.breach_horizons}")

        _persist_alert_record(
            plot=plot,
            recommendation=recommendation,
            tension_forecast=tension_forecast,
            threshold=threshold,
            current_tension=current_value,
            satellite_validation=satellite_validation,
            runtime_state=runtime_state,
        )

        operation, operation_created = _recommendation_operation(
            plot, pipeline_result, recommendation, runtime_state,
            current_value, threshold, satellite_validation)
        mode = resolve_irrigation_mode(plot)
        if not operation_created:
            log.info("Duplicate recommendation operation suppressed for plot %s",
                     getattr(plot, "stable_id", getattr(plot, "id", "?")))
            return 0

        if not actuator_supported or mode == "advisory_only":
            print(
                f"  [Decision Engine] {plot.user_given_name}: irrigation mode '{_resolve_irrigation_type(plot)}' is advisory-only; no actuator will be triggered.")
            return 0
        if mode == "approval_required":
            print(f"  [Decision Engine] {plot.user_given_name}: irrigation is pending approval.")
            return 0
        if mode == "manual":
            print(f"  [Decision Engine] {plot.user_given_name}: manual mode; recommendation recorded without actuation.")
            return 0

        if recommendation.urgency == "critical":
            print(
                f"  CRITICAL: Immediate irrigation for {plot.user_given_name}!")
            volume = _resolve_runtime_irrigation_volume(plot, runtime_state)
            if volume is None:
                print(
                    f"  No runtime irrigation volume available for {plot.user_given_name}; skipping irrigation.")
                return 0
            if operation is None:
                return irrigate_amount(plot, volume)
            operation["amount_m3"] = volume
            return execute_operation_command(plot, operation)
        elif recommendation.should_irrigate:
            # Check if predictions show natural recovery before irrigating
            if isinstance(predictions, pd.DataFrame) and 'smoothed_values' in predictions.columns:
                next_lower_idx, _ = find_next_occurrences(
                    predictions, 'smoothed_values', threshold, timeSpanOverThreshold)
                if next_lower_idx:
                    print(
                        f"  Recovery expected at {next_lower_idx}, delaying irrigation")
                    return 0
            print(
                f"  Irrigating {plot.user_given_name} (breach at {recommendation.first_breach_horizon})")
            volume = _resolve_runtime_irrigation_volume(plot, runtime_state)
            if volume is None:
                print(
                    f"  No runtime irrigation volume available for {plot.user_given_name}; skipping irrigation.")
                return 0
            if operation is None:
                return irrigate_amount(plot, volume)
            operation["amount_m3"] = volume
            return execute_operation_command(plot, operation)
        else:
            print(f"  No irrigation needed for {plot.user_given_name}")
            return 0

    if getattr(plot, "_inference_source", "live") != "live":
        print(
            f"  [Decision Engine] Skipped: inference source='{getattr(plot, '_inference_source', None)}'"
        )
        return 0

    log.warning(
        "Actuation skipped: decision engine unavailable for sensor kind '%s'",
        getattr(plot, "sensor_kind", None),
    )
    return 0
