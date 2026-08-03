import json
import logging
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
import pytz

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests

# import create_model
from utils import NetworkUtils, TimeUtils
import threading

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
    from crop_model import get_stress_threshold, compute_gdd_from_weather, get_crop_state
    from phenology_engine import compute_kc_gdd_series
    from phenology_engine import get_growth_stage
    from phenology_engine import check_satellite_tension_consistency
    from crops import STAGE_NAMES
    HAS_CROP_MODEL = True
except ImportError:
    HAS_CROP_MODEL = False

# Decision engine — compares forecast vs threshold
try:
    from decision_engine import evaluate_forecast, IrrigationRecommendation
    HAS_DECISION_ENGINE = True
except ImportError:
    HAS_DECISION_ENGINE = False


# Globals
# 20% allowed, fixed value TODO: make configurable
OverThresholdAllowed = 1.2
# 3 hours until verification of irrigation is done, should be more than 1 h, fixed value TODO: make configurable DEBUG
Irrigation_confirmation_sec = 10800
WEATHER_STALE_HOURS = 6.0
SATELLITE_STALE_HOURS = 72.0
# Per-plot irrigation retry counts (thread-safe: keyed by plot.id)
# Prevents cross-plot interference when multiple plots verify concurrently.
_irrigation_retries: dict = {}

# Cache for dynamic threshold (recomputed at most once per day)
# key: (crop_type, planting_date, lat, lon) -> (date, cumulative_gdd)
_runtime_crop_state_cache = {}

FORECAST_HORIZON_HOURS = [24, 48, 120, 168]

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


def _sum_rainfall_mm(frame):
    if frame is None or frame.empty or "Rain" not in frame.columns:
        return 0.0

    rain = pd.to_numeric(frame["Rain"], errors="coerce").fillna(0.0)
    rain = rain.clip(lower=0.0)
    return float(rain.sum())


def _build_rainfall_summary(plot, weather=None):
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

    try:
        lat, lon = _parse_plot_coordinates(plot)
        today = datetime.now().date()
        look_ahead_hours = float(getattr(plot, 'look_ahead_time', 24) or 24)
        forecast_days = max(1, int(np.ceil(look_ahead_hours / 24.0)))
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=forecast_days)).strftime("%Y-%m-%d")
        forecast_frame = fetch_weather_frame(
            lat,
            lon,
            start_date=start_date,
            end_date=end_date,
        )
        rain_forecast_mm = _sum_rainfall_mm(forecast_frame)
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
        validation_lookback_days = max(
            30, int(getattr(runtime_state, "sat_ndvi_age_hours", 0.0) // 24) + 45)
    except (TypeError, ValueError):
        validation_lookback_days = 60

    try:
        satellite_history = fetch_satellite_history(
            lat,
            lon,
            as_of=now_ts,
            lookback_days=validation_lookback_days,
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
            lookback_days=max(days_ago + 14, 30),
        )
        ndvi = float("nan")
        ndre = float("nan")
        ndvi_age_hours = float("inf")
        vv_db = float("nan")
        sat_age_hours = float("inf")
        if isinstance(satellite, pd.DataFrame) and not satellite.empty:
            latest_sat = satellite.iloc[-1]
            ndvi = float(latest_sat.get("sat_ndvi", float("nan")))
            ndre = float(latest_sat.get("sat_ndre", float("nan")))
            ndvi_age_hours = float(latest_sat.get(
                "sat_ndvi_age", latest_sat.get("satellite_data_age", float("inf"))))
            vv_db = float(latest_sat.get("sat_vv_db", float("nan")))
            sat_age_hours = float(latest_sat.get(
                "satellite_data_age", float("inf")))

        satellite_summary = _summarize_data_frame(satellite, SATELLITE_FIELDS)

        state = get_crop_state(
            farm_cfg,
            current_gdd,
            ndvi=ndvi,
            ndre=ndre,
            ndvi_age_hours=ndvi_age_hours,
            etc_daily_mm=0.0,
        )
        state.sat_ndvi = ndvi
        state.sat_ndre = ndre
        state.sat_ndvi_age_hours = ndvi_age_hours
        state.sat_vv_db = vv_db
        state.sat_data_age_hours = sat_age_hours
        state.weather_data_age_hours = weather_age_hours
        state.weather_data_summary = weather_summary
        state.satellite_data_summary = satellite_summary
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

        rainfall_summary = _build_rainfall_summary(plot, weather)
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


def compute_runtime_dynamic_threshold(plot) -> float:
    """
    Crop Model at runtime: compute the growth-stage-aware
    irrigation threshold for the current moment.

    Uses the crop model to:
            1. Fetch daily temperature data from SpaceIoTBox (cached per day)
      2. Accumulate GDD from planting date to today
      3. Determine current growth stage
      4. Return the crop stress threshold (PWP_base + Δ_offset)

    Falls back to plot.threshold (static) if:
      - Crop model not available
      - No planting date configured
      - Weather API fails
      - use_dynamic_threshold is False

    Args:
        plot: Plot object with crop_type, planting_date, permanent_wilting_point,
              gps_info (lat/lon), use_dynamic_threshold, threshold (static fallback)

    Returns:
        Threshold value in cbar (or humidity units for capacitive sensors)
    """
    state = _compute_runtime_crop_state(plot)
    if state is None:
        return plot.threshold

    threshold = state.stress_threshold_cbar
    print(f"  [Crop Model] GDD={state.gdd_cumulative:.0f}, "
          f"stage={state.growth_stage_name}, threshold={threshold:.1f} cbar "
          f"(crop={state.crop_type})")
    return threshold

# Find global max and min => not used any more


def get_max_min(df, target_col='smoothed_values'):
    # reset "new" index
    df = df.reset_index(inplace=False)

    # index
    global_min_index = df[target_col].idxmin()
    global_max_index = df[target_col].idxmax()
    # value
    global_min = df[target_col].min()
    global_max = df[target_col].max()

    print(global_min_index, "value:", global_min, global_max_index,
          "value:", global_max, "length:", global_max_index-global_min_index)
    print(global_min_index, "value:", df[target_col][global_min_index], global_max_index,
          "value:", df[target_col][global_max_index], "length:", global_max_index-global_min_index)

    return global_min_index, global_max_index, global_min, global_max


# Function to find next lower and higher value occurrence
def find_next_occurrences(df, column, threshold, timeSpanOverThreshold):
    timezone = TimeUtils.Timezone

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

    volume_mm = crop_state.recommended_volume_mm
    if volume_mm is None:
        volume_mm = crop_state.etc_daily_mm

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
    sat_stale = bool(sat_age > SATELLITE_STALE_HOURS)
    stale_sources = []
    if weather_stale:
        stale_sources.append("weather")
    if sat_stale:
        stale_sources.append("satellite")

    area_m2 = _resolve_plot_area_m2(plot)
    irrigation_type = _resolve_irrigation_type(plot)
    efficiency = _resolve_irrigation_efficiency(plot)
    actuator_based = _has_actuator_support(plot)
    recommended_volume_m3 = _resolve_runtime_irrigation_volume(
        plot, runtime_state)

    return {
        "available": True,
        "actuator_based": actuator_based,
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
                "used_for": ["sat_ndvi", "sat_ndre", "sat_vv_db"],
                "model_processing": [
                    "NDVI and NDRE feed Kc blending",
                    "VV is used only to cap NDVI staleness when SAR confirms canopy",
                ],
            },
        },
        "data_stale": bool(stale_sources),
        "data_stale_sources": stale_sources,
        "data_stale_thresholds": {
            "weather_hours": WEATHER_STALE_HOURS,
            "satellite_hours": SATELLITE_STALE_HOURS,
        },
        "reason": (
            "No pump/actuator configured for this irrigation mode. Advice only."
            if not actuator_based else
            "Rainfall-adjusted irrigation need computed from weather history and forecast."
        ),
    }


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
        now_utc = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

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


def save_irrigation_time(amount, plotid, status="not confirmed") -> int:
    # Load from file
    filename = 'data/irrigations_plot_' + str(plotid) + '.json'
    data = read_data_from_file(filename)

    # obtain timezone
    timezone = pytz.timezone(TimeUtils.Timezone)
    # add to current timestamp without converting it
    now = timezone.localize(datetime.now())

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

    # Get sensor id of confirmation device.
    confirmation = plot.device_and_sensor_ids_flow_confirmation
    if not isinstance(confirmation, list) or len(confirmation) == 0 or not isinstance(confirmation[0], str):
        print(
            f"No confirmation sensor configured for plot {plot.id}, cannot verify irrigation.")
        update_irrigation_status(plot, "no_confirmation_sensor_configured")
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
            return

        resp = response.json()
        tz = pytz.timezone(TimeUtils.Timezone)
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
            _irrigation_retries[plot_id] = 0
        else:
            retries = _irrigation_retries.get(plot_id, 0)
            if retries == 0:
                print(f"Irrigation failed for plot {plot_id}: "
                      f"delivered={last_value}m³, expected={amount}m³. "
                      f"Retrying once.")
                update_irrigation_status(
                    plot, "irrigation failed, retrying once")
                irrigate_amount(plot, amount)
                _irrigation_retries[plot_id] = 1
            else:
                if Irrigation_retries == 0:
                    print(
                        f"Irrigation failed for plot {plot.id}: amount_given: {last_value}m³, expected amount: {amount}m³. Irrigation will be retried once.")
                    update_irrigation_status(
                        plot, "irrigation failed, retrying once")
                    irrigate_amount(plot, amount)
                    Irrigation_retries += 1
                    # Here another action could be triggered, like sending notification
                else:
                    update_irrigation_status(
                        plot, "irrigation failed, twice, no more retries")
                    print(
                        f"Irrigation failed for plot {plot.id}: amount_given: {last_value}m³, expected amount: {amount}m³. Irrigation will not be retried.")
                    Irrigation_retries = 0
        else:
            print("Verification failed:", response.status_code, response.text)
            update_irrigation_status(plot, "verification_of_irrigation_failed")
            print(
                f"Verification of irrigation failed for plot {plot.id}: expected amount: {amount}m³. Irrigation will not be retried.")
    except requests.exceptions.RequestException as e:
        print(f"Verification request error for plot {plot_id}: {e}")
        update_irrigation_status(plot, "verification_failed_request_error")
        print(
            f"Request of verification of irrigation failed for plot {plot.id}: expected amount: {amount}m³. Irrigation will not be retried.")


# Load from wazigate API
# TODO: renew the token, make function in NetworkUtils that does a arbitrary API request
def irrigate_amount(plot, amount=0):
    # Example API call:
    # curl -X POST "http://192.168.189.2/devices/6645c4d468f31971148f2ab1/actuators/6673fcb568f31971148ff5f7/value"
    # -H "accept: */*" -H "Content-Type: application/json" -d "7.2"

    # if there is no amount in arguments, take it from config -> It is automatically triggered, retrieve amount!
    if amount == 0:
        amount = plot.irrigation_amount

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
            # Save times on when there was an irrigation TODO: wait for confirmation from microcontroller, irrigation could be skipped, needs to be implemented!!
            save_irrigation_time(amount, plot.id, status="not confirmed")
            response_ok = True

            # Schedule verification after 3 hours (10800 sec), the varification is only called once
            timer = threading.Timer(
                Irrigation_confirmation_sec, verify_irrigation, args=[plot, amount])
            timer.name = f"IrrigationCheckRoutine-{plot.id}"
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

    return response_ok

# Mighty main function TODO: encapsulate


def main_old(currentSoilTension, threshold_timestamp, predictions, plot) -> int:
    # Get configuration
    threshold = plot.threshold
    timeSpanOverThreshold = plot.look_ahead_time

    future_time = datetime.now() + timedelta(hours=timeSpanOverThreshold)
    threshold_timestamp = future_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    now = datetime.now().replace(microsecond=0)

    # "Weak" irrigation strategy
    # If threshold was met
    if currentSoilTension > threshold:
        print(
            f"Threshold: {threshold} was reached with a value of {currentSoilTension}.")

        # If current soil tension exceeds threshold by more than allowed margin, irrigate immediately
        if currentSoilTension > threshold * OverThresholdAllowed:
            print(
                f"Threshold: {threshold} was exceeded by 20%, irrigate immediately!")
            e = irrigate_amount(plot)
            return e

        # Check predictions
        next_lower_idx, next_higher_idx = find_next_occurrences(
            predictions, 'smoothed_values', threshold, timeSpanOverThreshold)

        # If no lower value is predicted within the forecast horizon, trigger irrigation
        if not next_lower_idx:
            print(
                f"No lower value predicted within {timeSpanOverThreshold} hours, irrigate now!")
            e = irrigate_amount(plot)
            return e

        # Otherwise, no immediate irrigation is needed
        else:
            print(
                f"Soil tension is high, but irrigation can wait. Since it is expected go below the threshold at: {next_lower_idx}")
            return 0

    # Threshold was not met, so do not irrigate
    else:
        print(
            f"Threshold: {threshold} is not reached, current soil tension is: {currentSoilTension}, do not irrigate.")
        return 0


# Mighty main function TODO: encapsulate
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

    runtime_state = _compute_runtime_crop_state(plot)
    threshold = (runtime_state.stress_threshold_cbar
                 if runtime_state is not None
                 else compute_runtime_dynamic_threshold(plot))
    timeSpanOverThreshold = plot.look_ahead_time

    actuator_supported = _has_actuator_support(plot)

    # Layer 3: Decision engine — compare current/forecast vs threshold
    if (
        HAS_DECISION_ENGINE
        and plot.sensor_kind == "tension"
        and getattr(plot, "_inference_source", "live") == "live"
    ):
        # Build forecast dict from predictions if available
        tension_forecast = {}
        if isinstance(predictions, pd.DataFrame) and 'smoothed_values' in predictions.columns:
            timezone = TimeUtils.Timezone
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
                horizons_to_sample = [
                    h for h in FORECAST_HORIZON_HOURS
                    if max_hours is None or h <= max_hours + 1e-6
                ]
                # Sample at standard horizons (match training horizons)
                for hours in horizons_to_sample:
                    target_time = now_cmp + timedelta(hours=hours)
                    closest_idx = future.index[future.index <= target_time]
                    if len(closest_idx) > 0:
                        val = future.loc[closest_idx[-1], 'smoothed_values']
                        tension_forecast[f"{hours}h"] = float(val)
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

        if not actuator_supported:
            print(
                f"  [Decision Engine] {plot.user_given_name}: irrigation mode '{_resolve_irrigation_type(plot)}' is advisory-only; no actuator will be triggered.")
            return 0

        if recommendation.urgency == "critical":
            print(
                f"  CRITICAL: Immediate irrigation for {plot.user_given_name}!")
            volume = _resolve_runtime_irrigation_volume(plot, runtime_state)
            if volume is None:
                print(
                    f"  No runtime irrigation volume available for {plot.user_given_name}; skipping irrigation.")
                return 0
            return irrigate_amount(plot, volume)
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
            return irrigate_amount(plot, volume)
        else:
            print(f"  No irrigation needed for {plot.user_given_name}")
            return 0

    if getattr(plot, "_inference_source", "live") != "live":
        print(
            f"  [Decision Engine] Skipped: inference source='{getattr(plot, '_inference_source', None)}'"
        )
        return 0

    # Fallback: capacitive sensors or no decision engine — use legacy comparison
    comparison_fn = (lambda value, thresh: value > thresh) if plot.sensor_kind == "tension" else (
        lambda value, thresh: value < thresh
    )

    over_threshold_fn = (
        (lambda value, thresh: value > thresh * OverThresholdAllowed)
        if plot.sensor_kind == "tension"
        else (lambda value, thresh: value < thresh / OverThresholdAllowed)
    )

    # "Weak" irrigation strategy
    if comparison_fn(current_value, threshold):
        print(
            f"Threshold: {threshold} was reached with a value of {current_value} on {plot.user_given_name}.")

        # Immediate irrigation if over-threshold logic is satisfied
        if over_threshold_fn(current_value, threshold):
            print(
                f"Immediate irrigation triggered for sensor_kind '{plot.sensor_kind} on {plot.user_given_name}'!")
            volume = _resolve_runtime_irrigation_volume(plot, runtime_state)
            if volume is None:
                print(
                    f"  No runtime irrigation volume available for {plot.user_given_name}; skipping irrigation.")
                return 0
            return irrigate_amount(plot, volume)

        # Check predictions for next occurrence below/above threshold
        next_lower_idx, next_higher_idx = find_next_occurrences(
            predictions, 'smoothed_values', threshold, timeSpanOverThreshold)

        # No recovery predicted within forecast horizon
        if (plot.sensor_kind == "tension" and not next_lower_idx) or (plot.sensor_kind == "humidity" and not next_higher_idx):
            print(
                f"No recovery predicted within {timeSpanOverThreshold} hours on {plot.user_given_name}, irrigate now!")
            volume = _resolve_runtime_irrigation_volume(plot, runtime_state)
            if volume is None:
                print(
                    f"  No runtime irrigation volume available for {plot.user_given_name}; skipping irrigation.")
                return 0
            return irrigate_amount(plot, volume)

        # Otherwise, delay irrigation
        else:
            target_time = next_lower_idx if plot.sensor_kind == "tension" else next_higher_idx
            print(
                f"Irrigation can wait. Recovery expected at: {target_time} on {plot.user_given_name}")
            return 0

    # Threshold was not met, so do not irrigate
    else:
        print(
            f"Threshold: {threshold} is not reached on {plot.user_given_name}, current value is: {current_value}. Do not irrigate.")
        return 0
