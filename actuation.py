import json
import hashlib
import logging
import os
import copy
from sqlite3 import Error as SQLiteError
from datetime import datetime, timedelta
from types import SimpleNamespace
import pytz

import numpy as np
import pandas as pd
import requests

# import create_model
from utils import NetworkUtils, TimeUtils
import threading
import runtime_config
from recommendation_contract import compose_recommendation, threshold_condition
from operations_store import get_operations_store, MODES
from sensor_quality import SENSOR_STALE_HOURS, evaluate_tension_sensor_safety

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
    from phenology_engine import compute_kc_gdd_series, daily_temperature_extrema
    from phenology_engine import get_growth_stage
    from phenology_engine import check_canopy_consistency
    from crops import STAGE_NAMES, get_crop_params
    from eo_observation import analyse_vegetation_history
    HAS_CROP_MODEL = True
except ImportError:
    HAS_CROP_MODEL = False

# Decision engine — compares forecast vs threshold
try:
    from decision_engine import error_recommendation, evaluate_forecast
    HAS_DECISION_ENGINE = True
except ImportError:
    HAS_DECISION_ENGINE = False


# Globals
WEATHER_STALE_HOURS = 6.0
RUNTIME_STATE_CACHE_HOURS = 1.0
# Tracks in-flight requests per plot so repeated recommendations cannot send
# duplicate actuator commands while a command is being created and sent.
_active_irrigations = set()


def _trace_event(event, plot, **details):
    if os.getenv("IRRIGATION_TRACE_EVENTS", "").strip().lower() not in {
            "1", "true", "yes", "on"}:
        return
    payload = {
        "event": event,
        "plot_id": getattr(plot, "stable_id", getattr(plot, "id", None)),
        "plot_name": getattr(plot, "user_given_name", None),
        **details,
    }
    serialized = json.dumps(payload, default=str, sort_keys=True)
    log.info("[IRRIGATION_TRACE] %s", serialized)
    print(f"[IRRIGATION_TRACE] {serialized}", flush=True)
_irrigation_lock = threading.Lock()
_irrigation_history_lock = threading.Lock()


def _release_active_irrigation(plot_id):
    """Release a plot's in-flight reservation under the shared lock."""
    with _irrigation_lock:
        _active_irrigations.discard(plot_id)

# Cache for weather/phenology/EO crop state (recomputed at most once per day)
# key: (crop_type, planting_date, lat, lon) -> (date, cumulative_gdd)
_runtime_crop_state_cache = {}


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
    _trace_event("decision.crop_state.unavailable", plot, reason=reason,
                 error=str(exc) if exc is not None else None)
    return None


def _build_runtime_farm_config(plot):
    """Build a minimal farm-like config object for crop_model helpers."""
    initial_gdd = getattr(plot, 'initial_gdd', 0.0)
    depletion_fraction = getattr(plot, 'depletion_fraction', None)
    return SimpleNamespace(
        crop_type=getattr(plot, 'crop_type', ''),
        soil_texture_class=getattr(plot, 'soil_texture_class', None),
        planting_date=getattr(plot, 'planting_date', ''),
        harvest_date=getattr(plot, 'harvest_date', None),
        timezone=getattr(plot, 'timezone', 'UTC'),
        initial_gdd=float(initial_gdd or 0.0),
        static_threshold_cbar=getattr(
            plot, 'threshold_static', getattr(plot, 'threshold', None)),
        threshold_mode=getattr(plot, 'threshold_mode', 'static'),
        field_capacity_vwc=getattr(plot, 'field_capacity_vwc', None),
        wilting_point_vwc=getattr(plot, 'wilting_point_vwc', None),
        root_depth_m=getattr(plot, 'root_depth_m', None),
        sensor_depth_m=getattr(plot, 'sensor_depth_m', None),
        depletion_fraction=depletion_fraction,
        stage_depletion_fractions=getattr(
            plot, 'stage_depletion_fractions', None),
        stage_thresholds_cbar=getattr(plot, 'stage_thresholds_cbar', None),
        threshold_hysteresis_cbar=getattr(
            plot, 'threshold_hysteresis_cbar', 0.0),
        soil_water_retention_curve=getattr(
            plot, 'soil_water_retention_curve', None),
        soil_calibration=getattr(plot, 'soil_calibration', {}),
        saturation=getattr(plot, 'saturation', 0),
        field_capacity_lower=getattr(plot, 'field_capacity_lower', None),
        permanent_wilting_point=getattr(
            plot, 'permanent_wilting_point', None),
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


def _has_actuator_support(plot):
    irrigation_type = _resolve_irrigation_type(plot)
    if irrigation_type in NON_ACTUATOR_IRRIGATION_TYPES:
        return False

    flow_ids = getattr(plot, 'device_and_sensor_ids_flow', []) or []
    return len(flow_ids) > 0


def resolve_irrigation_mode(plot):
    """Return an explicit mode; legacy actuator plots require approval."""
    configured = str(getattr(plot, "irrigation_mode", "") or "").strip().lower()
    if configured in MODES:
        return configured
    return "approval_required" if _has_actuator_support(plot) else "advisory_only"


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
    water_contract = get_irrigation_recommendation(plot).get("water", {})
    amount = water_contract.get("recommended_volume_m3")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = None
    if amount is not None and (not np.isfinite(amount) or amount <= 0):
        amount = None
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

    # An unavailable evaluation must clear any older active warning without
    # creating a misleading planned irrigation operation.
    if decision["urgency"] in {"error", "unknown"}:
        _, created = store.record_alert(
            idempotency_key=f"alert:{key}", operation_id=None,
            farm_id=getattr(plot, "farm_id", None), plot_id=stable_id,
            urgency=decision["urgency"], payload=decision)
        return None, created

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


def schedule_window_error(operation, now=None):
    """Return a schedule's execution blocker; windows are [start, end).

    Legacy timezone-naive timestamps retain the API's UTC interpretation.
    Schedules without an end remain open-ended for compatibility.
    """
    if operation.get('source') != 'schedule':
        return None
    try:
        def timestamp(value):
            if value is None or not isinstance(value, (str, pd.Timestamp)):
                raise ValueError('Missing or invalid schedule timestamp')
            result = pd.Timestamp(value)
            if pd.isna(result):
                raise ValueError('Invalid schedule timestamp')
            return result.tz_localize('UTC') if result.tzinfo is None else result.tz_convert('UTC')

        start = timestamp(operation.get('planned_start'))
        end_raw = operation.get('planned_end')
        end = timestamp(end_raw) if end_raw is not None else None
        if end is not None and end <= start:
            return 'schedule_window_invalid'
        now = pd.Timestamp.now(tz='UTC') if now is None else timestamp(now)
        if now < start:
            return 'schedule_not_due'
        if end is not None and now >= end:
            return 'schedule_window_expired'
    except (TypeError, ValueError, OverflowError):
        return 'schedule_window_invalid'
    return None


def execute_operation_command(plot, operation):
    """Atomically claim a persisted operation before attempting one command.

    An interrupted claim remains active for reconciliation, never automatic
    replay: a crash or timeout cannot prove the hardware did not receive it.
    """
    if operation is None:
        return None
    store = get_operations_store()
    operation = store.get_operation(operation.get("operation_id"))
    if operation is None:
        return None
    if operation.get("status") not in {"planned", "approved"}:
        return None
    plot_id = str(getattr(plot, "stable_id", getattr(plot, "id", "")))
    if str(operation.get("plot_id")) != plot_id:
        log.warning("Operation plot does not match command target: %s", plot_id)
        return None
    if store.has_pending_flow_verification(plot_id):
        _transition_operation(operation, "failed", {
            "error": "prior_flow_confirmation_pending",
        })
        log.warning(
            "Irrigation command rejected for plot %s: a prior command still "
            "awaits flow confirmation", plot_id)
        return None
    window_error = schedule_window_error(operation)
    if window_error:
        if window_error != 'schedule_not_due':
            _transition_operation(operation, 'failed', {'error': window_error})
        return None
    # transition() checks and updates the status inside BEGIN IMMEDIATE.
    # Only the caller that changes it to active owns the execution attempt.
    try:
        operation, claimed = store.transition(
            operation["operation_id"], "active", {"command": "claimed"})
    except (KeyError, ValueError):
        return None
    if not claimed:
        return None

    if operation.get("source") == "recommendation":
        decision = operation.get("recommendation") or {}
        pipeline = getattr(plot, "pipeline_result", None)
        latest_decision = getattr(pipeline, "recommendation", None)
        current = decision.get("current_tension_cbar")
        sensor = evaluate_tension_sensor_safety(plot, current_value=current)
        reasons = []
        if not sensor.get("safe_for_automatic", False):
            reasons.extend(sensor.get("reasons") or ["unsafe_tension_evidence"])
        if pipeline is None or getattr(pipeline, "error", None):
            reasons.append("latest_pipeline_result_unavailable")
        elif getattr(pipeline, "inference_source", "live") != "live":
            reasons.append("latest_pipeline_result_is_not_live")
        elif latest_decision is None or not bool(getattr(
                latest_decision, "should_irrigate", False)):
            reasons.append("latest_pipeline_no_longer_recommends_irrigation")

        try:
            created = pd.Timestamp(operation.get("created_at"))
            created = (created.tz_localize("UTC") if created.tzinfo is None
                       else created.tz_convert("UTC"))
            age_hours = (
                pd.Timestamp.now(tz="UTC") - created).total_seconds() / 3600.0
            if age_hours > SENSOR_STALE_HOURS:
                reasons.append("recommendation_expired_before_execution")
        except (TypeError, ValueError):
            reasons.append("recommendation_creation_time_invalid")

        if reasons:
            _transition_operation(operation, "failed", {
                "error": "recommendation_execution_safety_failed",
                "reasons": sorted(set(reasons)),
            })
            log.warning(
                "Recommendation command rejected for plot %s: %s",
                getattr(plot, "id", "?"), sorted(set(reasons)))
            return None
    amount = operation.get("amount_m3")
    if amount is None:
        _transition_operation(operation, "failed", {"error": "missing_irrigation_amount"})
        return None
    # Recheck after claiming in case waiting for the database crossed the end.
    window_error = schedule_window_error(operation)
    if window_error:
        _transition_operation(operation, 'failed', {'error': window_error})
        return None
    verification_context = {}
    response = irrigate_amount(
        plot, float(amount), authorized=True,
        operation_id=operation["operation_id"],
        verification_context=verification_context)
    if response is True:
        detail = {"command": "sent"}
        if verification_context:
            detail["flow_verification"] = verification_context
        completed = _transition_operation(operation, "completed", detail)
        if verification_context and completed is not None:
            _schedule_irrigation_verification(
                plot, float(amount), verification_context)
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
            lat, lon = _parse_weather_coordinates(plot)
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
        if field not in frame.columns:
            missing_fields.append(field)
            field_values[field] = None
            continue

        numeric_series = pd.to_numeric(frame[field], errors="coerce")
        if numeric_series.notna().any():
            available_fields.append(field)
            field_values[field] = round(float(numeric_series.dropna().iloc[-1]), 4)
            continue

        text_series = frame[field].dropna().astype(str)
        text_series = text_series[text_series.str.strip() != ""]
        if not text_series.empty:
            available_fields.append(field)
            field_values[field] = text_series.iloc[-1]
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


def _parse_weather_coordinates(plot):
    """Return the farm location shared by every plot's weather workflow."""
    gps_info = getattr(plot, 'farm_gps_info', None)
    if isinstance(gps_info, dict):
        lat = gps_info.get('latitude', gps_info.get('lattitude'))
        lon = gps_info.get('longitude')
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    # Compatibility for isolated tests and installations loaded before the
    # farm registry migration; active runtime plots receive farm_gps_info.
    return _parse_plot_coordinates(plot)


def _resolve_phenology_reference(plot, crop_type, plot_lat, plot_lon):
    """Resolve an explicit planting date or a quality-gated EO emergence date.

    EO does not reveal the literal sowing date. When a crop is known but its
    planting date is not, a sustained green-up interval can anchor GDD at the
    crop's emergence requirement without changing the saved farmer input.
    """
    explicit = str(getattr(plot, "planting_date", "") or "").strip()
    if explicit:
        return {
            "date": explicit,
            "source": "farmer_reported_planting_date",
            "anchor_gdd": 0.0,
            "confidence": "reported",
            "interval": None,
        }
    if not crop_type:
        return None

    _trace_event(
        "decision.phenology_reference.started", plot, scope="plot",
        method="eo_greening", latitude=plot_lat, longitude=plot_lon)
    try:
        history = fetch_satellite_history(
            plot_lat, plot_lon, lookback_days=365, limit=100,
            include_ndre=False)
        observation = analyse_vegetation_history(history)
        greening = observation.get("greening") or {}
        confidence = greening.get("confidence")
        reference_date = greening.get("estimated_end")
        accepted = (
            bool(greening.get("detected"))
            and confidence in {"moderate", "high"}
            and bool(reference_date)
        )
        _trace_event(
            "decision.phenology_reference.completed", plot, scope="plot",
            accepted=accepted, method="eo_estimated_emergence",
            confidence=confidence,
            estimated_start=greening.get("estimated_start"),
            estimated_end=reference_date,
            observations=observation.get("valid_observations", 0),
            status=greening.get("status"),
        )
        if not accepted:
            return None
        return {
            "date": reference_date,
            "source": "eo_estimated_emergence",
            "anchor_gdd": float(get_crop_params(crop_type).gdd_emergence),
            "confidence": confidence,
            "interval": {
                "start": greening.get("estimated_start"),
                "end": reference_date,
            },
        }
    except Exception as exc:
        _trace_event(
            "decision.phenology_reference.failed", plot, scope="plot",
            method="eo_greening", error=str(exc))
        return None


def _compute_satellite_validation(plot, runtime_state, current_tension, tension_forecast):
    if runtime_state is None or not HAS_CROP_MODEL:
        return None

    try:
        lat, lon = _parse_plot_coordinates(plot)
    except (ValueError, AttributeError, TypeError):
        return None

    now_ts = pd.Timestamp.now(tz="UTC")
    cached_history = getattr(runtime_state, "satellite_history", None)
    if isinstance(cached_history, pd.DataFrame):
        satellite_history = cached_history
    else:
        try:
            # Sample a bounded history once per cached runtime state.
            satellite_history = fetch_satellite_history(
                lat,
                lon,
                as_of=now_ts,
                lookback_days=180,
                limit=24,
                include_ndre=True,
            )
            runtime_state.satellite_history = satellite_history
        except Exception as exc:
            log.warning("Satellite validation history fetch failed: %s", exc)
            satellite_history = pd.DataFrame()

    return check_canopy_consistency(
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

    crop_type = getattr(plot, 'crop_type', '')

    try:
        evaluation_at = pd.Timestamp.now(tz="UTC")
        plot_lat, plot_lon = _parse_plot_coordinates(plot)
        weather_lat, weather_lon = _parse_weather_coordinates(plot)
        reference = _resolve_phenology_reference(
            plot, crop_type, plot_lat, plot_lon)
        if reference is None:
            reason = ('missing_crop' if not crop_type
                      else 'missing_planting_date_and_no_reliable_eo_emergence')
            return _runtime_state_fail(plot, reason)
        planting_date_str = reference["date"]
        planting_date = pd.Timestamp(planting_date_str)
        farm_cfg = _build_runtime_farm_config(plot)
        farm_cfg.planting_date = planting_date_str
    except (ValueError, AttributeError, TypeError):
        return _runtime_state_fail(plot, 'invalid_crop_or_coordinates')

    cache_payload = {
        "plot_id": getattr(plot, "stable_id", getattr(plot, "id", None)),
        "crop": crop_type,
        "planting": planting_date_str,
        "phenology_reference_source": reference["source"],
        "harvest": getattr(plot, "harvest_date", None),
        "plot_lat": plot_lat,
        "plot_lon": plot_lon,
        "weather_lat": weather_lat,
        "weather_lon": weather_lon,
        "soil_texture": getattr(plot, "soil_texture_class", None),
        "initial_gdd": getattr(plot, "initial_gdd", 0.0),
        "threshold_static": getattr(plot, "threshold_static", None),
        "threshold_mode": getattr(plot, "threshold_mode", "static"),
        "dynamic_threshold_inputs": {
            "soil_calibration": getattr(plot, "soil_calibration", {}),
            "field_capacity_lower": getattr(plot, "field_capacity_lower", None),
            "permanent_wilting_point": getattr(plot, "permanent_wilting_point", None),
            "saturation": getattr(plot, "saturation", 0),
            "field_capacity_vwc": getattr(plot, "field_capacity_vwc", None),
            "wilting_point_vwc": getattr(plot, "wilting_point_vwc", None),
            "root_depth_m": getattr(plot, "root_depth_m", None),
            "sensor_depth_m": getattr(plot, "sensor_depth_m", None),
            "depletion_fraction": getattr(plot, "depletion_fraction", None),
            "stage_depletion_fractions": getattr(
                plot, "stage_depletion_fractions", None),
            "stage_thresholds_cbar": getattr(plot, "stage_thresholds_cbar", None),
            "threshold_hysteresis_cbar": getattr(
                plot, "threshold_hysteresis_cbar", 0.0),
            "retention_curve": getattr(plot, "soil_water_retention_curve", None),
        },
        "area": getattr(plot, "plot_area_m2", None),
        "irrigation_type": _resolve_irrigation_type(plot),
        "application_efficiency": getattr(plot, "application_efficiency", 0.85),
        "effective_rainfall_fraction": getattr(
            plot, "effective_rainfall_fraction", 0.80),
        "experimental_ndre": bool(getattr(plot, "enable_experimental_ndre_kc", False)),
    }
    cache_key = hashlib.sha256(
        _json_for_key(cache_payload).encode("utf-8")).hexdigest()
    today = evaluation_at.tz_convert(TimeUtils.for_plot(plot)).date()
    if cache_key in _runtime_crop_state_cache:
        cached_date, cached_state = _runtime_crop_state_cache[cache_key]
        if cached_date == today:
            computed_at = getattr(cached_state, "_computed_at_utc", None)
            if computed_at is not None:
                now_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
                delta_hours = max(0.0, (now_ts - pd.Timestamp(
                    computed_at)).total_seconds() / 3600.0)
                if delta_hours <= RUNTIME_STATE_CACHE_HOURS:
                    result = copy.copy(cached_state)
                    for attr in (
                        "weather_data_age_hours", "sat_data_age_hours",
                        "sat_ndvi_age_hours", "sat_ndre_age_hours",
                    ):
                        base_age = getattr(cached_state, attr, None)
                        try:
                            base_age = float(base_age)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(base_age):
                            setattr(result, attr, base_age + delta_hours)
                    _trace_event(
                        "decision.crop_state.cache_hit", plot,
                        age_hours=round(delta_hours, 3),
                        threshold_mode=getattr(result, "threshold_mode", None),
                        threshold_cbar=getattr(result, "stress_threshold_cbar", None),
                    )
                    return result

    try:
        start_str = planting_date.strftime("%Y-%m-%d")
        end_str = today.strftime("%Y-%m-%d")
        _trace_event(
            "decision.weather.started", plot, scope="farm",
            farm_id=getattr(plot, "farm_id", None), latitude=weather_lat,
            longitude=weather_lon, start=start_str, end=end_str)
        weather = fetch_weather_frame(
            weather_lat, weather_lon, start_date=start_str, end_date=end_str)
        if weather.empty:
            return _runtime_state_fail(
                plot,
                f'no_weather_for_range:{start_str}:{end_str}'
            )

        weather_summary = _summarize_data_frame(weather, WEATHER_FIELDS)
        _trace_event(
            "decision.weather.completed", plot,
            provider=weather.attrs.get("provider", "unknown"),
            rows=len(weather),
            fallback_reason=weather.attrs.get("fallback_reason"),
            missing_fields=weather_summary.get("missing_fields"),
        )

        timezone_name = TimeUtils.for_plot(plot)
        weather.index = pd.to_datetime(
            weather.index, utc=True).tz_convert(timezone_name)
        weather = weather.sort_index()

        now_ts = evaluation_at.tz_convert(timezone_name)
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

        # GDD represents development only through the evaluation instant. Some
        # providers include later hours from the current day in this response.
        daily_tmax, daily_tmin = daily_temperature_extrema(
            temp_series, timezone_name, through=now_ts)
        if daily_tmax.dropna().empty:
            raise ValueError("No historical temperature is available through now")
        daily_tmax = daily_tmax.interpolate(limit_direction="both")
        daily_tmin = daily_tmin.interpolate(limit_direction="both")

        if "Et0_evapotranspiration" in weather.columns:
            et0 = pd.to_numeric(
                weather["Et0_evapotranspiration"], errors="coerce")
        else:
            et0 = pd.Series(dtype=float)

        et0_available = not et0.empty and not et0.dropna().empty
        if not et0_available:
            et0_daily = pd.Series(0.0, index=daily_tmax.index, dtype=float)
        else:
            et0_daily = et0.resample("D").sum(min_count=1).fillna(0.0)

        gdd_series = compute_gdd_from_weather(daily_tmax, daily_tmin, farm_cfg)
        if reference["anchor_gdd"] > 0:
            gdd_series = gdd_series + reference["anchor_gdd"]
        current_gdd = gdd_series.iloc[-1] if len(gdd_series) > 0 else 0.0

        _trace_event("decision.eo.started", plot, scope="plot",
                     latitude=plot_lat, longitude=plot_lon)
        satellite = fetch_satellite_snapshot(
            plot_lat,
            plot_lon,
            as_of=evaluation_at,
            include_ndre=True,
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
        _trace_event(
            "decision.eo.completed", plot,
            provider=satellite.attrs.get("provider", "unavailable"),
            rows=len(satellite),
            missing_fields=satellite_summary.get("missing_fields"),
        )
        et0_today = (
            float(et0_daily.loc[now_ts.normalize()])
            if et0_available and now_ts.normalize() in et0_daily.index else None
        )
        try:
            state = get_crop_state(
                farm_cfg,
                current_gdd,
                ndvi=ndvi,
                ndre=ndre,
                ndvi_age_hours=ndvi_age_hours,
                ndre_age_hours=ndre_age_hours,
                ndvi_quality=ndvi_quality,
                ndre_quality=ndre_quality,
                allow_experimental_ndre=bool(getattr(
                    plot, "enable_experimental_ndre_kc", False)),
                et0_daily_mm=et0_today,
                as_of=now_ts,
            )
        except ValueError as exc:
            return _runtime_state_fail(
                plot, f"invalid_crop_or_threshold_configuration:{exc}")
        state.sat_ndvi = ndvi
        state.sat_ndre = ndre
        state.sat_ndvi_age_hours = ndvi_age_hours
        state.sat_vv_db = vv_db
        state.sat_data_age_hours = sat_age_hours
        state.sat_ndre_age_hours = ndre_age_hours
        state.sat_ndvi_quality = ndvi_quality
        state.sat_ndre_quality = ndre_quality
        state.weather_data_age_hours = weather_age_hours
        state.phenology_reference_date = planting_date_str
        state.phenology_reference_source = reference["source"]
        state.phenology_reference_confidence = reference["confidence"]
        state.phenology_reference_interval = reference["interval"]
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
                weather_lat, weather_lon,
                start_date=today.strftime("%Y-%m-%d"),
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
        state.etc_daily_mm = (
            float(max(0.0, et0_today * state.kc))
            if et0_today is not None else 0.0
        )
        state.eo_etc_delta_mm = (
            float(et0_today * (state.kc - state.kc_gdd))
            if et0_today is not None else 0.0
        )

        rainfall_summary = _build_rainfall_summary(plot, weather, forecast_weather)
        state.rain_since_planting_mm = rainfall_summary["rain_since_planting_mm"]
        state.rain_last_24h_mm = rainfall_summary["rain_last_24h_mm"]
        state.rain_forecast_mm = rainfall_summary["rain_forecast_mm"]
        kc_gdd_series = compute_kc_gdd_series(gdd_series, crop_type)
        state.cumulative_etc_mm = float(
            max(0.0, (et0_daily * kc_gdd_series).sum()))
        rain_fraction = float(getattr(
            plot, "effective_rainfall_fraction", 0.80))
        state.historical_rain_credit_mm = float(
            state.rain_since_planting_mm * rain_fraction)
        state.rain_effective_mm = state.historical_rain_credit_mm

        _runtime_crop_state_cache[cache_key] = (today, state)
        try:
            setattr(plot, 'runtime_crop_state_error', '')
        except (AttributeError, TypeError):
            pass
        _trace_event(
            "decision.crop_state.completed", plot,
            crop=crop_type,
            phenology_reference_date=planting_date_str,
            phenology_reference_source=reference["source"],
            phenology_reference_confidence=reference["confidence"],
            stage=getattr(state, "growth_stage_name", None),
            gdd=getattr(state, "gdd_cumulative", None),
            kc=getattr(state, "kc", None),
            threshold_mode=getattr(state, "threshold_mode", None),
            threshold_cbar=getattr(state, "stress_threshold_cbar", None),
            threshold_source=(getattr(state, "threshold_details", {}) or {}).get(
                "source"),
        )
        return state
    except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
        return _runtime_state_fail(plot, 'runtime_crop_state_exception', exc)


def get_runtime_crop_state(plot):
    """Return the cached or freshly computed runtime crop state."""
    return _compute_runtime_crop_state(plot)


def find_recovery_after_breach(
    df,
    column,
    breach_timestamp,
    current_threshold,
    threshold_timestamps=None,
    recovery_horizon_hours=24.0,
    hysteresis_cbar=0.0,
):
    """Return the first genuine recovery strictly after a forecast breach."""
    if (not isinstance(df, pd.DataFrame) or df.empty or column not in df
            or breach_timestamp is None):
        return None
    index = pd.DatetimeIndex(df.index)
    breach = pd.Timestamp(breach_timestamp)
    if index.tz is None and breach.tzinfo is not None:
        breach = breach.tz_localize(None)
    elif index.tz is not None and breach.tzinfo is None:
        breach = breach.tz_localize(index.tz)
    elif index.tz is not None and breach.tzinfo is not None:
        breach = breach.tz_convert(index.tz)
    try:
        horizon = max(0.0, float(recovery_horizon_hours))
        hysteresis = max(0.0, float(hysteresis_cbar or 0.0))
        fallback_threshold = float(current_threshold)
    except (TypeError, ValueError):
        return None

    values = pd.to_numeric(df[column], errors="coerce")
    candidate_mask = (index > breach) & (
        index <= breach + pd.Timedelta(hours=horizon))
    candidates = values.loc[candidate_mask]
    if candidates.empty:
        return None

    thresholds = pd.Series(fallback_threshold, index=index, dtype=float)
    if isinstance(threshold_timestamps, dict) and threshold_timestamps:
        mapped = {}
        for timestamp, value in threshold_timestamps.items():
            try:
                key = pd.Timestamp(timestamp)
                if index.tz is None and key.tzinfo is not None:
                    key = key.tz_localize(None)
                elif index.tz is not None and key.tzinfo is None:
                    key = key.tz_localize(index.tz)
                elif index.tz is not None and key.tzinfo is not None:
                    key = key.tz_convert(index.tz)
                mapped[key] = float(value)
            except (TypeError, ValueError):
                continue
        if mapped:
            projected = pd.Series(mapped, dtype=float).sort_index()
            thresholds = projected.reindex(index, method="ffill").fillna(
                fallback_threshold)

    recovered = candidates < (thresholds.loc[candidates.index] - hysteresis)
    matches = recovered[recovered].index
    return matches[0] if len(matches) else None


def assess_plot_capabilities(plot, crop_state=None, current_tension=None) -> dict:
    """Report readiness using observable runtime data and hardware settings."""
    sensor = evaluate_tension_sensor_safety(plot, current_tension)
    area = _resolve_plot_area_m2(plot)
    actuator_ready = _has_actuator_support(plot) and area is not None
    actuator_reasons = []
    if not _has_actuator_support(plot):
        actuator_reasons.append("actuator_missing")
    if area is None:
        actuator_reasons.append("positive_plot_area_m2")
    eo_quality_values = []
    if crop_state is not None:
        for name in ("sat_ndvi_quality", "sat_ndre_quality"):
            try:
                value = float(getattr(crop_state, name, 0.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                eo_quality_values.append(value)
    eo_ready = max(eo_quality_values, default=0.0) > 0.0
    try:
        threshold = float(getattr(
            plot, "threshold_static", getattr(plot, "threshold", None)))
    except (TypeError, ValueError):
        threshold = np.nan
    threshold_ready = np.isfinite(threshold) and threshold > 0
    threshold_mode = str(getattr(plot, "threshold_mode", "static") or "static")
    threshold_reasons = [] if threshold_ready else ["positive_field_threshold"]
    if threshold_mode == "dynamic":
        dynamic_source = (getattr(crop_state, "threshold_details", None) or {}).get(
            "source") if crop_state is not None else None
        threshold_ready = threshold_ready and dynamic_source in {
            "fao56_root_zone_depletion", "existing_soil_curve_crop_demand",
            "farmer_calibrated_stage_curve",
        }
        if not threshold_ready:
            threshold_reasons = ["complete_dynamic_threshold_calibration"]
    return {
        "tension_evidence": {
            "implemented": True,
            "ready_for_automatic": sensor["safe_for_automatic"],
            "missing_or_invalid": sensor["reasons"],
            "details": sensor,
        },
        "tension_trigger": {
            "implemented": True,
            "ready_for_automatic": threshold_ready,
            "missing_or_invalid": threshold_reasons,
        },
        "eo_kc_refinement": {
            "implemented": True,
            "ready": eo_ready,
            "missing_or_invalid": [] if eo_ready else [
                "fresh_quality_checked_prepared_eo_observation"],
        },
        "calculated_irrigation": {
            "implemented": True,
            "ready": actuator_ready,
            "missing_or_invalid": actuator_reasons,
        },
    }


def get_irrigation_recommendation(plot):
    # Prefer the crop state already used by orchestration; compute one only when
    # the API is called before the first pipeline cycle.
    pipeline_result = getattr(plot, "pipeline_result", None)
    runtime_state = getattr(pipeline_result, "crop_state", None)
    if runtime_state is None:
        runtime_state = _compute_runtime_crop_state(plot)
    if runtime_state is None:
        threshold = getattr(
            plot, "threshold_static", getattr(plot, "threshold", None))
        reason_code = getattr(
            plot, "runtime_crop_state_error", "crop_state_unavailable")
        reason = (
            "Planting date is not available and EO did not establish a "
            "reliable emergence reference. Tension observations and model "
            "forecasts can still be shown, but crop-stage irrigation advice "
            "is intentionally disabled."
            if reason_code == "missing_planting_date_and_no_reliable_eo_emergence"
            else (
                "Dynamic threshold calibration is invalid or incomplete: "
                + reason_code.split(":", 1)[1]
                if str(reason_code).startswith(
                    "invalid_crop_or_threshold_configuration:")
                else f"Crop state could not be computed: {reason_code}."
            )
        )
        return {
            "available": False,
            "reason": reason,
            "availability_reason": reason,
            "crop_type": getattr(plot, "crop_type", None),
            "growth_stage": None,
            "crop": {
                "type": getattr(plot, "crop_type", None),
                "current_stage": None,
                "status": "not_planted_or_unknown",
            },
            "condition": threshold_condition(None, threshold),
            "threshold_cbar": threshold,
            "threshold_mode": getattr(plot, "threshold_mode", "static"),
            "model_forecast_available": bool(
                pipeline_result is not None
                and getattr(pipeline_result, "model_status", "failed") != "failed"
                and bool(getattr(pipeline_result, "tension_forecast", {}))
            ),
            "forecast_timestamps": list(
                getattr(pipeline_result, "forecast_timestamps", []) or []),
            "data_sources": dict(
                getattr(pipeline_result, "data_sources", {}) or {}),
        }

    pipeline_error = getattr(pipeline_result, "error", None)
    pipeline_recommendation = getattr(
        pipeline_result, "recommendation", None) if pipeline_result is not None else None
    model_ready = (
        pipeline_result is not None
        and pipeline_error is None
        and pipeline_recommendation is not None
    )
    if pipeline_result is None:
        availability_reason = (
            "No completed model forecast is available. Train this plot to create one."
        )
    elif pipeline_error is not None:
        availability_reason = f"The latest model cycle failed: {pipeline_error}"
    elif pipeline_recommendation is None:
        availability_reason = "The latest model cycle did not produce a recommendation."
    else:
        availability_reason = None

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
    actuator_based = (_has_actuator_support(plot)
                      and resolve_irrigation_mode(plot) != "advisory_only")
    contract_generated_at = pd.Timestamp.now(tz="UTC")
    irrigation_credit = {
        "volume_m3": 0.0,
        "operation_count": 0,
        "since": None,
        "until": contract_generated_at.isoformat(),
        "available": False,
    }
    if pipeline_result is not None:
        calculated_at = getattr(pipeline_result, "calculated_at", None)
        previous_calculated_at = getattr(
            pipeline_result, "previous_calculated_at", None)
        # On the first cycle, count only irrigation completed after that cycle.
        # On later cycles, count from the preceding successful calculation so
        # recently applied water is not recommended again while sensors catch up.
        credit_since = previous_calculated_at or calculated_at
        if credit_since is not None:
            try:
                irrigation_credit = {
                    **get_operations_store().applied_irrigation_since(
                        getattr(plot, "stable_id", getattr(plot, "id", None)),
                        credit_since,
                        contract_generated_at.to_pydatetime(),
                    ),
                    "available": True,
                }
            except (OSError, SQLiteError, ValueError) as exc:
                log.warning(
                    "Could not calculate applied-irrigation credit for plot %s: %s",
                    getattr(plot, "stable_id", getattr(plot, "id", "?")),
                    exc,
                )
    base = {
        "available": model_ready,
        "availability_reason": availability_reason,
        "actuator_based": actuator_based,
        "irrigation_mode": resolve_irrigation_mode(plot),
        "crop_type": runtime_state.crop_type,
        "growth_stage": runtime_state.growth_stage_name,
        "season_active": bool(getattr(runtime_state, "season_active", True)),
        "season_status": getattr(runtime_state, "season_status", "active"),
        "season_end_reason": getattr(runtime_state, "season_end_reason", None),
        "kc": round(float(runtime_state.kc), 3) if getattr(runtime_state, "kc", None) is not None else None,
        "kc_gdd": _finite_or_none(getattr(runtime_state, "kc_gdd", None), 3),
        "kc_eo": _finite_or_none(getattr(runtime_state, "kc_eo", None), 3),
        "eo_source": getattr(runtime_state, "eo_source", "none"),
        "eo_weight": _finite_or_none(getattr(runtime_state, "eo_weight", 0.0), 3),
        "eo_quality_factor": _finite_or_none(getattr(runtime_state, "eo_quality_factor", 0.0), 3),
        "eo_plausibility_factor": _finite_or_none(getattr(runtime_state, "eo_plausibility_factor", 0.0), 3),
        "eo_kc_delta": _finite_or_none(getattr(runtime_state, "eo_kc_delta", 0.0), 3),
        "eo_etc_delta_mm": _finite_or_none(getattr(runtime_state, "eo_etc_delta_mm", 0.0), 3),
        "eo_reason": getattr(runtime_state, "eo_reason", ""),
        "eo_indices": getattr(runtime_state, "eo_indices", None),
        "etc_daily_mm": round(float(runtime_state.etc_daily_mm), 2),
        "application_efficiency": _finite_or_none(getattr(
            plot, "application_efficiency", 0.85), 3),
        "effective_rainfall_fraction": _finite_or_none(getattr(
            plot, "effective_rainfall_fraction", 0.80), 3),
        "demand_horizon_hours": _finite_or_none(getattr(
            plot, "look_ahead_time", 24.0), 1),
        "capability_readiness": assess_plot_capabilities(
            plot, runtime_state, getattr(pipeline_result, "current_tension", None)),
        "irrigation_type": irrigation_type,
        "plot_area_m2": round(float(area_m2), 1) if area_m2 is not None else None,
        "applied_irrigation_credit": irrigation_credit,
        "threshold_cbar": round(float(runtime_state.stress_threshold_cbar), 1),
        "threshold_static_cbar": _finite_or_none(getattr(
            plot, "threshold_static", getattr(plot, "threshold", None)), 1),
        "threshold_mode": getattr(runtime_state, "threshold_mode", "static"),
        "threshold_reason": getattr(runtime_state, "threshold_reason", ""),
        "threshold_details": getattr(runtime_state, "threshold_details", None),
        "gdd_cumulative": round(float(runtime_state.gdd_cumulative), 1),
        "rain_since_planting_mm": round(float(getattr(runtime_state, "rain_since_planting_mm", 0.0)), 2),
        "rain_last_24h_mm": round(float(getattr(runtime_state, "rain_last_24h_mm", 0.0)), 2),
        "rain_forecast_mm": round(float(getattr(runtime_state, "rain_forecast_mm", 0.0)), 2),
        "historical_rain_credit_mm": _finite_or_none(
            getattr(runtime_state, "historical_rain_credit_mm", None), 2),
        "cumulative_etc_mm": round(float(getattr(runtime_state, "cumulative_etc_mm", 0.0)), 2),
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
                    "prepared NDVI is checked for freshness, cadence quality, and plausible values",
                    "quality-weighted NDVI can make a bounded correction to GDD-based Kc",
                    "NDRE-to-Kc is disabled unless an experimental field trial explicitly enables it",
                    "canopy history is assessed against expected stage direction, not tension forecasts",
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
            "Irrigation timing uses soil tension, weather, and bounded EO refinement."
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
        forecast_weather=getattr(runtime_state, "weather_forecast_frame", None),
        generated_at=contract_generated_at)
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
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)

    temporary = filename + ".tmp"
    with open(temporary, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        json_file.flush()
        os.fsync(json_file.fileno())
    os.replace(temporary, filename)

# Function to add a new record


def add_record(data, timestamp, amount, status="commanded", operation_id=None):
    record = {
        "timestamp": timestamp,
        "amount": amount,
        "status": status
    }
    if operation_id:
        record["operation_id"] = operation_id
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


def save_irrigation_time(
        amount, plot, status="commanded", operation_id=None) -> int:
    filename = 'data/irrigations_plot_' + str(plot.id) + '.json'
    timezone = pytz.timezone(TimeUtils.for_plot(plot))
    rounded_tz = round_to_nearest_10_minutes(datetime.now(tz=timezone))
    with _irrigation_history_lock:
        data = read_data_from_file(filename)
        data = add_record(
            data, str(rounded_tz), amount, status,
            operation_id=operation_id)
        save_data_to_file(filename, data)

    print("Irrigation time has been saved to: ", filename)

    return 0


def update_irrigation_status(
        plot, status="not_confirmed", operation_id=None, detail=None):
    """Update the matching compatibility irrigation record.

    New callers identify the operation explicitly. Updating the final record
    is retained only for old records/callers that have no operation ID.
    """
    filename = 'data/irrigations_plot_' + str(plot.id) + '.json'
    with _irrigation_history_lock:
        data = read_data_from_file(filename)
        records = data.get("irrigations", [])
        target = None
        if operation_id:
            target = next((record for record in reversed(records)
                           if record.get("operation_id") == operation_id), None)
        elif records:
            target = records[-1]
        if target is None:
            log.warning("No irrigation record found for plot %s operation %s",
                        getattr(plot, "id", "?"), operation_id)
            return False
        target["status"] = status
        if detail:
            target["verification"] = detail
        save_data_to_file(filename, data)
    return True


def _confirmation_sensor_reference(plot, actuator_ids=None):
    configured = getattr(
        plot, "device_and_sensor_ids_flow_confirmation", []) or []
    if isinstance(configured, list) and configured:
        reference = configured[0]
    else:
        discover = getattr(plot, "getConfirmationDeviceID", None)
        reference = discover(actuator_ids) if callable(discover) else ""
    reference = str(reference or "").strip()
    parts = reference.split("/", 1)
    return reference if len(parts) == 2 and all(parts) else ""


def _read_flow_confirmation(sensor_reference):
    device_id, sensor_id = sensor_reference.split("/", 1)
    url = (f"{NetworkUtils.ApiUrl}devices/{device_id}/sensors/"
           f"{sensor_id}")
    response = requests.get(
        url,
        headers={'Authorization': f'Bearer {NetworkUtils.Token}'},
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"flow_confirmation_http_{response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("flow confirmation response must be an object")
    value = payload.get("value")
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        raise ValueError("flow confirmation value must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("flow confirmation value is not finite")
    timestamp = payload.get("time") or payload.get("timestamp")
    parsed_time = None
    if timestamp:
        parsed_time = pd.Timestamp(timestamp)
        if parsed_time.tzinfo is None:
            parsed_time = parsed_time.tz_localize("UTC")
        else:
            parsed_time = parsed_time.tz_convert("UTC")
    return {"value_m3": value,
            "timestamp": parsed_time.isoformat() if parsed_time is not None else None}


def _schedule_irrigation_verification(
        plot, amount, context, delay_seconds=None):
    delay = (runtime_config.get_timing_config(plot)
             .irrigation_confirmation_seconds if delay_seconds is None
             else max(0.0, float(delay_seconds)))
    timer = threading.Timer(
        delay, verify_irrigation, args=(plot, amount, context))
    timer.name = f"IrrigationCheckRoutine-{getattr(plot, 'id', 'unknown')}"
    timer.daemon = True
    timer.start()
    return timer


def resume_pending_irrigation_verifications(plots):
    """Recreate verification timers for accepted commands after a restart."""
    by_id = {
        str(getattr(plot, "stable_id", getattr(plot, "id", ""))): plot
        for plot in (plots.values() if isinstance(plots, dict) else plots)
    }
    store = get_operations_store()
    resumed = 0
    for operation in store.list_operations(status="completed", limit=500):
        plot = by_id.get(str(operation.get("plot_id")))
        if plot is None:
            continue
        completed_event = next((event for event in reversed(
            store.events(operation["operation_id"]))
            if event.get("to_status") == "completed"), None)
        context = ((completed_event or {}).get("detail") or {}).get(
            "flow_verification")
        if not isinstance(context, dict) or not context.get("baseline"):
            continue
        try:
            command_time = pd.Timestamp(context.get("command_time"))
            if pd.isna(command_time):
                raise ValueError("missing command timestamp")
            if command_time.tzinfo is None:
                command_time = command_time.tz_localize("UTC")
            configured_delay = runtime_config.get_timing_config(
                plot).irrigation_confirmation_seconds
            elapsed = max(0.0, (pd.Timestamp.now(tz="UTC") - command_time)
                          .total_seconds())
        except (TypeError, ValueError, OverflowError) as exc:
            detail = {
                "error": "invalid_persisted_verification_context",
                "reason": str(exc),
            }
            _transition_operation(operation, "failed", detail)
            try:
                update_irrigation_status(
                    plot, "verification_failed", operation["operation_id"],
                    detail)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as history_exc:
                log.warning(
                    "Could not update compatibility history while rejecting "
                    "persisted verification context for %s: %s",
                    operation["operation_id"], history_exc)
            continue
        _schedule_irrigation_verification(
            plot, operation["amount_m3"], context,
            delay_seconds=max(0.0, configured_delay - elapsed))
        resumed += 1
    return resumed


def verify_irrigation(plot, amount, context=None):
    """Confirm delivered volume without ever issuing an automatic retry."""
    context = dict(context or {})
    operation_id = context.get("operation_id")
    if operation_id:
        existing = get_operations_store().get_operation(operation_id)
        if existing and existing.get("status") == "verified":
            return True
        if existing and existing.get("status") == "failed":
            return False
    sensor_reference = context.get("sensor_reference") or (
        _confirmation_sensor_reference(plot))
    baseline = context.get("baseline") or {}
    try:
        if isinstance(amount, bool):
            raise ValueError("boolean amount")
        expected = float(amount)
    except (TypeError, ValueError):
        expected = None
    detail = {
        "expected_m3": expected,
        "sensor_reference": sensor_reference,
    }

    def update_compatibility_record(status):
        try:
            update_irrigation_status(
                plot, status, operation_id, detail)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            log.warning(
                "Could not update compatibility irrigation history for "
                "plot %s operation %s: %s",
                getattr(plot, "id", "?"), operation_id, exc)

    def fail(error):
        detail["error"] = error
        if operation_id:
            operation = get_operations_store().get_operation(operation_id)
            if operation and operation.get("status") in {"active", "completed"}:
                _transition_operation(operation, "failed", detail)
        update_compatibility_record("verification_failed")
        log.error("Irrigation verification failed for plot %s: %s",
                  getattr(plot, "id", "?"), error)
        return False

    if expected is None or not np.isfinite(expected) or expected <= 0:
        return fail("invalid_expected_irrigation_amount")
    if not sensor_reference:
        return fail("confirmation_sensor_unavailable")
    try:
        final = _read_flow_confirmation(sensor_reference)
        final_raw = final["value_m3"]
        baseline_raw = baseline["value_m3"]
        if isinstance(final_raw, bool) or isinstance(baseline_raw, bool):
            return fail("non_numeric_flow_reading")
        final_value = float(final_raw)
        baseline_value = float(baseline_raw)
        if not np.isfinite(final_value) or not np.isfinite(baseline_value):
            return fail("non_finite_flow_reading")
        command_time = pd.Timestamp(context.get("command_time"))
        if not final.get("timestamp"):
            return fail("confirmation_reading_has_no_timestamp")
        final_time = pd.Timestamp(final["timestamp"])
        if command_time.tzinfo is None:
            command_time = command_time.tz_localize("UTC")
        if final_time.tzinfo is None:
            final_time = final_time.tz_localize("UTC")
        if final_time < command_time:
            return fail("confirmation_reading_predates_command")
        confirmation_seconds = runtime_config.get_timing_config(
            plot).irrigation_confirmation_seconds
        sampling_grace = max(
            300,
            runtime_config.get_timing_config(
                plot).sensor_sampling_interval_minutes * 60,
        )
        if final_time > command_time + pd.Timedelta(
                seconds=confirmation_seconds + sampling_grace):
            return fail("confirmation_reading_outside_operation_window")
        if final_time > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5):
            return fail("confirmation_reading_is_in_the_future")
    except (requests.exceptions.RequestException, RuntimeError, ValueError,
            TypeError, KeyError) as exc:
        return fail(str(exc) or "confirmation_read_failed")

    configured_mode = str(getattr(
        plot, "flow_confirmation_mode", "event") or "event").strip().lower()
    if configured_mode == "cumulative":
        measurement_mode = "cumulative_delta"
        delivered = final_value - baseline_value
    elif configured_mode == "event":
        measurement_mode = "event_value"
        delivered = final_value
    else:
        return fail("unsupported_flow_confirmation_mode")
    rel_tol = float(getattr(plot, "flow_confirmation_relative_tolerance", 0.10))
    abs_tol = float(getattr(plot, "flow_confirmation_absolute_tolerance_m3", 0.01))
    if (not np.isfinite(rel_tol) or rel_tol < 0
            or not np.isfinite(abs_tol) or abs_tol < 0):
        return fail("invalid_flow_confirmation_tolerance")
    tolerance = max(abs_tol, rel_tol * abs(expected))
    detail.update({
        "baseline_m3": baseline_value,
        "final_m3": final_value,
        "delivered_m3": delivered,
        "measurement_mode": measurement_mode,
        "tolerance_m3": tolerance,
        "reading_time": final["timestamp"],
    })
    if delivered < 0 or abs(delivered - expected) > tolerance:
        return fail("delivery_mismatch")

    if operation_id:
        operation = get_operations_store().get_operation(operation_id)
        if operation and operation.get("status") in {"active", "completed"}:
            _transition_operation(operation, "verified", detail)
    update_compatibility_record("verified")
    return True


# Load from wazigate API
# TODO: renew the token, make function in NetworkUtils that does a arbitrary API request
def irrigate_amount(
        plot, amount=None, authorized=False, operation_id=None,
        verification_context=None):
    # Example API call:
    # curl -X POST "http://192.168.189.2/devices/6645c4d468f31971148f2ab1/actuators/6673fcb568f31971148ff5f7/value"
    # -H "accept: */*" -H "Content-Type: application/json" -d "7.2"

    try:
        if isinstance(amount, bool):
            raise ValueError("boolean amount")
        amount = float(amount)
    except (TypeError, ValueError):
        log.error("Irrigation command rejected: calculated amount is unavailable")
        return None
    if not np.isfinite(amount) or amount <= 0:
        log.error("Irrigation command rejected: calculated amount must be positive and finite")
        return None

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

    stable_plot_id = str(getattr(
        plot, "stable_id", getattr(plot, "id", "")))
    if not operation_id and stable_plot_id:
        try:
            confirmation_pending = get_operations_store().has_pending_flow_verification(
                stable_plot_id)
        except (OSError, SQLiteError) as exc:
            log.error("Irrigation rejected for plot %s: cannot check pending "
                      "flow confirmations: %s", stable_plot_id, exc)
            return None
        if confirmation_pending:
            log.error("Irrigation rejected for plot %s: a prior command "
                      "still awaits flow confirmation", stable_plot_id)
            return None

    # Name of flow meter sensor to initiate irrigation => TODO: decide on using single or multiple
    flow_ids = getattr(plot, 'device_and_sensor_ids_flow', []) or []
    if len(flow_ids) == 0:
        print(
            f"Irrigation skipped for plot {getattr(plot, 'id', '?')}: no actuator configured.")
        return None

    confirmation_sensor = _confirmation_sensor_reference(plot, flow_ids)
    confirmation_mode = str(getattr(
        plot, "flow_confirmation_mode", "event") or "event").strip().lower()
    if confirmation_mode not in {"event", "cumulative"}:
        log.error("Irrigation rejected for plot %s: unsupported flow "
                  "confirmation mode '%s'",
                  getattr(plot, "id", "?"), confirmation_mode)
        return None
    baseline = None
    if confirmation_sensor:
        try:
            baseline = _read_flow_confirmation(confirmation_sensor)
        except (requests.exceptions.RequestException, RuntimeError, ValueError,
                TypeError, KeyError) as exc:
            log.error("Cannot read flow baseline for plot %s: %s",
                      getattr(plot, "id", "?"), exc)
    if mode == "automatic" and baseline is None:
        log.error("Automatic irrigation rejected for plot %s: a readable "
                  "flow confirmation sensor is required",
                  getattr(plot, "id", "?"))
        return None

    plot_id = getattr(plot, "id", None)
    # Reserve the plot before the network request to close concurrent races.
    with _irrigation_lock:
        if plot_id in _active_irrigations:
            log.warning("Duplicate irrigation prevented for plot %s", plot_id)
            return None
        _active_irrigations.add(plot_id)

    response_ok = None
    try:
        flow_meter_name = flow_ids[0]

        # API URL
        apiUrl = NetworkUtils.ApiUrl

        # Create URL for API call
        flow_parts = flow_meter_name.split('/')
        request_url = (
            f"{apiUrl}devices/{flow_parts[0]}/actuators/"
            f"{flow_parts[1]}/value"
        )

        # Define headers for the POST request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {NetworkUtils.Token}'
        }

        # Send a POST request to the API
        command_time = pd.Timestamp.now(tz="UTC").isoformat()
        response = requests.post(
            request_url, headers=headers, json=amount, timeout=30)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            response_ok = True
            # The hardware command has already succeeded. A local logging
            # problem must not turn that success into an API failure.
            try:
                save_irrigation_time(
                    amount, plot, status="commanded",
                    operation_id=operation_id)
            except (OSError, ValueError, TypeError) as exc:
                log.exception(
                    "Irrigation command succeeded but event recording failed for plot %s: %s",
                    plot_id,
                    exc,
                )

            response_ok = True
            if baseline is not None:
                context = {
                    "operation_id": operation_id,
                    "sensor_reference": confirmation_sensor,
                    "baseline": baseline,
                    "command_time": command_time,
                }
                if verification_context is not None:
                    verification_context.update(context)
                else:
                    _schedule_irrigation_verification(plot, amount, context)
        else:
            print("Irrigation failed for plot")
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.text)
            response_ok = None
    except (requests.exceptions.RequestException, ValueError, TypeError,
            KeyError, AttributeError, IndexError) as e:
        # Handle request exceptions (e.g., connection errors)
        print("Request error:", e)
        response_ok = None  # TODO: introduce error handling
    finally:
        # This is an in-flight request guard, not an irrigation-state flag.
        # Once the HTTP attempt finishes, later commands must be allowed.
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
    mode = resolve_irrigation_mode(plot)
    sensor_safety = evaluate_tension_sensor_safety(
        plot, current_value=current_value)
    freshness["soil_tension_age_hours"] = sensor_safety.get(
        "latest_age_hours")
    freshness["soil_tension_safe_for_automatic"] = sensor_safety.get(
        "safe_for_automatic", False)
    freshness["soil_tension_quality"] = sensor_safety
    automatic_block_reasons = []
    if mode == "automatic" and not sensor_safety["safe_for_automatic"]:
        automatic_block_reasons.extend(sensor_safety["reasons"])

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

        # Orchestration owns the decision. Re-evaluating it here used to allow
        # stale/failed raw evidence to become a new critical alert even though
        # automatic hardware execution was later blocked.
        recommendation = getattr(pipeline_result, "recommendation", None)
        if recommendation is None:
            recommendation = error_recommendation(
                current_tension=current_value,
                tension_forecast=tension_forecast,
                stress_threshold=threshold,
                message="The completed pipeline did not contain a decision.",
            )
        if (not sensor_safety.get("safe_for_advisory", False)
                and getattr(recommendation, "urgency", None) != "error"):
            recommendation = error_recommendation(
                current_tension=current_value,
                tension_forecast=tension_forecast,
                stress_threshold=threshold,
                message=(
                    "No irrigation alert was evaluated because current raw "
                    "soil-tension evidence is unsafe for advisory use: "
                    + ", ".join(sensor_safety.get(
                        "advisory_reasons") or ["unavailable"])
                ),
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
        if not operation_created:
            log.info("Duplicate recommendation operation suppressed for plot %s",
                     getattr(plot, "stable_id", getattr(plot, "id", "?")))
            return 0

        calculated_amount = operation.get("amount_m3") if operation else None
        if recommendation.should_irrigate and (
                calculated_amount is None or float(calculated_amount) <= 0):
            log.info(
                "Irrigation skipped for plot %s: calculated crop water requirement is zero or unavailable",
                getattr(plot, "id", "?"))
            return 0

        if automatic_block_reasons:
            log.warning(
                "Automatic actuation skipped after recording a non-executable "
                "evaluation: %s", sorted(set(automatic_block_reasons)))
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
            volume = (operation.get("amount_m3") if operation is not None else
                      get_irrigation_recommendation(plot).get(
                          "water", {}).get("recommended_volume_m3"))
            if volume is None:
                print(
                    f"  Calculated irrigation volume is unavailable for {plot.user_given_name}; skipping irrigation.")
                return 0
            if operation is None:
                return irrigate_amount(plot, volume)
            operation["amount_m3"] = volume
            return execute_operation_command(plot, operation)
        elif recommendation.should_irrigate:
            # A recovery can delay actuation only when it occurs after the
            # breach. Safe values before the breach are not recovery evidence.
            if isinstance(predictions, pd.DataFrame) and 'smoothed_values' in predictions.columns:
                recovery = find_recovery_after_breach(
                    predictions,
                    'smoothed_values',
                    recommendation.first_breach_timestamp,
                    threshold,
                    threshold_timestamps=getattr(
                        pipeline_result, "stress_threshold_timestamps", {}),
                    recovery_horizon_hours=timeSpanOverThreshold,
                    hysteresis_cbar=getattr(
                        plot, "threshold_hysteresis_cbar", 0.0),
                )
                if recovery is not None:
                    print(
                        f"  Recovery expected at {recovery}, delaying irrigation")
                    return 0
            print(
                f"  Irrigating {plot.user_given_name} (breach at {recommendation.first_breach_horizon})")
            volume = (operation.get("amount_m3") if operation is not None else
                      get_irrigation_recommendation(plot).get(
                          "water", {}).get("recommended_volume_m3"))
            if volume is None:
                print(
                    f"  Calculated irrigation volume is unavailable for {plot.user_given_name}; skipping irrigation.")
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
