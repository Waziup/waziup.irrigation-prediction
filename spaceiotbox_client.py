from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests


DEFAULT_BASE_URL = "https://www.smartafrihub.com/spaceiotbox/api"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("SPACEIOTBOX_TIMEOUT_SECONDS", "30"))
REQUEST_ATTEMPTS = max(1, int(os.getenv("SPACEIOTBOX_REQUEST_ATTEMPTS", "3")))

# The documented agro-climate service is limited to the Lake Victoria area.
SPACEIOTBOX_LAT_RANGE = (-5.1, 2.5)
SPACEIOTBOX_LON_RANGE = (28.95, 36.7)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_DELAY_DAYS = 5

OPEN_METEO_ARCHIVE_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "rain", "cloud_cover",
    "shortwave_radiation", "wind_speed_10m", "wind_direction_10m",
    "soil_temperature_7_to_28cm", "soil_moisture_0_to_7cm",
    "et0_fao_evapotranspiration",
]
OPEN_METEO_FORECAST_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "rain", "cloud_cover",
    "shortwave_radiation", "wind_speed_10m", "wind_direction_10m",
    "soil_temperature_18cm", "soil_moisture_3_to_9cm",
    "et0_fao_evapotranspiration",
]

LEGACY_WEATHER_COLUMNS = [
    "Temperature",
    "Humidity",
    "Rain",
    "Cloudcover",
    "Shortwave_Radiation",
    "Windspeed",
    "Winddirection",
    "Soil_temperature_7-28",
    "Soil_moisture_0-7",
    "Et0_evapotranspiration",
]

TIME_ALIASES = [
    "Timestamp",
    "timestamp",
    "time",
    "datetime",
    "date",
    "valid_time",
    "validTime",
]

TIMEZONE_ALIASES = {
    "EAT": "Africa/Nairobi",
    "GMT": "UTC",
    "UTC": "UTC",
}

COLUMN_ALIASES = {
    "Temperature": [
        "temperature",
        "temperature_2m",
        "temperature_2m_mean",
        "temp",
        "temp_c",
        "air_temperature",
        "t2m",
    ],
    "Humidity": [
        "relativehumidity_2m",
        "relative_humidity_2m",
        "relative_humidity",
        "humidity",
    ],
    "Rain": ["rain", "precipitation", "rainfall", "precipitation_sum"],
    "Cloudcover": ["cloudcover", "cloud_cover", "cloudiness"],
    "Shortwave_Radiation": [
        "shortwave_radiation",
        "shortwave",
        "solar_radiation",
        "shortwave_radiation_sum",
    ],
    "Windspeed": [
        "windspeed_10m",
        "wind_speed_10m",
        "windspeed",
        "wind_speed",
    ],
    "Winddirection": [
        "winddirection_10m",
        "wind_direction_10m",
        "winddirection",
        "wind_direction",
    ],
    "Soil_temperature_7-28": [
        "soil_temperature_7_to_28cm",
        "soil_temperature_18cm",
        "soil_temperature",
        "soil_temp",
    ],
    "Soil_moisture_0-7": [
        "soil_moisture_0_to_7cm",
        "soil_moisture_3_to_9cm",
        "soil_moisture",
    ],
    "Et0_evapotranspiration": [
        "et0_fao_evapotranspiration",
        "et0",
        "evapotranspiration",
    ],
}


def get_base_url() -> str:
    return os.getenv("SPACEIOTBOX_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def get_api_key() -> Optional[str]:
    for name in ("SPACEIOTBOX_API_KEY", "API_KEY", "api_key"):
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def build_headers() -> dict:
    headers = {"Accept": "application/json"}
    api_key = get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _request_json(path: str, params: Optional[dict] = None) -> object:
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.get(
                f"{get_base_url()}{path}",
                params=params,
                headers=build_headers(),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.ok:
                return response.json()
            # Retry transient server failures, but preserve client errors such
            # as 422 for the bounded-query fallback handled by the caller.
            if response.status_code < 500 or attempt == REQUEST_ATTEMPTS - 1:
                response.raise_for_status()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == REQUEST_ATTEMPTS - 1:
                raise
        time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("SpaceIoTBox request exhausted without a response")


def _request_external_json(url: str, params: dict) -> object:
    """Apply the same bounded retry policy to the Open-Meteo fallback."""
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.get(
                url, params=params, headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.ok:
                return response.json()
            if response.status_code < 500 or attempt == REQUEST_ATTEMPTS - 1:
                response.raise_for_status()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == REQUEST_ATTEMPTS - 1:
                raise
        time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("Open-Meteo request exhausted without a response")


def fetch_agro_climate(endpoint: str, lat: float, lon: float, params: Optional[dict] = None) -> object:
    request_params = {"lat": float(lat), "lon": float(lon)}
    if params:
        request_params.update(params)
    try:
        return _request_json(f"/v1/agro_climate/{endpoint}", request_params)
    except requests.HTTPError as exc:
        # Some deployments expose the endpoint without start/end query filters.
        # Retry with required coords only and rely on local date filtering.
        response = getattr(exc, "response", None)
        if response is None or response.status_code != 422 or not params:
            raise
        return _request_json(
            f"/v1/agro_climate/{endpoint}",
            {"lat": float(lat), "lon": float(lon)},
        )


def coordinates_supported_by_spaceiotbox(lat: float, lon: float) -> bool:
    """Return whether coordinates satisfy the documented `/land` bounds."""
    return (
        SPACEIOTBOX_LAT_RANGE[0] <= float(lat) <= SPACEIOTBOX_LAT_RANGE[1]
        and SPACEIOTBOX_LON_RANGE[0] <= float(lon) <= SPACEIOTBOX_LON_RANGE[1]
    )


def _tabular_frame_from_value(value: object) -> Optional[pd.DataFrame]:
    if isinstance(value, list):
        frame = pd.DataFrame(value)
        return frame if not frame.empty else None

    if isinstance(value, dict):
        try:
            frame = pd.DataFrame(value)
        except ValueError:
            frame = pd.DataFrame([value])
        return frame if not frame.empty else None

    return None


def _extract_table(payload: object) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if not isinstance(payload, dict):
        return pd.DataFrame([payload])

    for key in ("forecast_data", "forecast", "historical_data", "historical", "current", "hourly", "daily", "weather", "observations", "features", "items", "results", "data"):
        value = payload.get(key)
        frame = _tabular_frame_from_value(value)
        if frame is not None:
            return frame

    return pd.json_normalize(payload)


def _payload_timezone(payload) -> str:
    """Resolve provider-local timestamps; normalized frames are already UTC."""
    if isinstance(payload, pd.DataFrame):
        return str(payload.attrs.get("normalized_timezone", "UTC"))
    if not isinstance(payload, dict):
        return "UTC"
    location = payload.get("location", {})
    candidates = [
        payload.get("timezone"),
        location.get("timezone") if isinstance(location, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            value = candidate.strip()
            return TIMEZONE_ALIASES.get(value.upper(), value)
    return "UTC"


def _time_index_from_frame(frame: pd.DataFrame, start_date=None, end_date=None) -> pd.DatetimeIndex:
    for column in TIME_ALIASES:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                return pd.DatetimeIndex(parsed)

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        if len(frame) <= 1:
            return pd.DatetimeIndex([start_ts])
        return pd.date_range(start=start_ts, periods=len(frame), freq="H")

    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        if len(frame) <= 1:
            return pd.DatetimeIndex([end_ts])
        return pd.date_range(end=end_ts, periods=len(frame), freq="H")

    return pd.DatetimeIndex(frame.index)


def _coerce_numeric_series(frame: pd.DataFrame, column_names: Iterable[str]) -> pd.Series:
    series = None
    for column_name in column_names:
        if column_name in frame.columns:
            current = pd.to_numeric(frame[column_name], errors="coerce")
            series = current if series is None else series.combine_first(
                current)
    if series is None:
        return pd.Series(np.nan, index=frame.index)
    return series


def _normalize_weather_bound(value) -> pd.Timestamp:
    bound = pd.Timestamp(value)
    if bound.tzinfo is not None:
        bound = bound.tz_convert("UTC").tz_localize(None)
    # Retain clock time so model forecasts can require the complete final day,
    # while date-only callers naturally remain midnight-bounded.
    return bound


def normalize_weather_frame(payload: object, start_date=None, end_date=None) -> pd.DataFrame:
    source_timezone = _payload_timezone(payload)
    raw_frame = _extract_table(payload)
    if raw_frame.empty:
        return raw_frame

    result = pd.DataFrame(index=raw_frame.index)
    result.index = _time_index_from_frame(
        raw_frame, start_date=start_date, end_date=end_date)
    if isinstance(result.index, pd.DatetimeIndex):
        if result.index.tz is not None:
            result.index = result.index.tz_convert("UTC").tz_localize(None)
        else:
            try:
                result.index = result.index.tz_localize(
                    source_timezone).tz_convert("UTC").tz_localize(None)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Unsupported weather response timezone: {source_timezone}")

    for column in LEGACY_WEATHER_COLUMNS:
        alias_columns = [column] + COLUMN_ALIASES.get(column, [])
        result[column] = _coerce_numeric_series(
            raw_frame, alias_columns).values

    if "Timestamp" not in result.columns:
        result["Timestamp"] = result.index

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.sort_index()
    result.attrs["source_timezone"] = source_timezone
    result.attrs["normalized_timezone"] = "UTC"

    # Filter only after columns are aligned with the original response. The
    # previous ordering could assign an unfiltered array to a shorter frame.
    if start_date is not None:
        result = result[
            result.index.normalize() >= _normalize_weather_bound(start_date).normalize()]
    if end_date is not None:
        result = result[
            result.index.normalize() <= _normalize_weather_bound(end_date).normalize()]
    if result.empty and (start_date is not None or end_date is not None):
        start_label = _normalize_weather_bound(
            start_date).date() if start_date is not None else "-"
        end_label = _normalize_weather_bound(
            end_date).date() if end_date is not None else "-"
        raise ValueError(
            f"Weather response had no rows within requested window {start_label}..{end_label}"
        )
    return result


def _covers_requested_window(frame: pd.DataFrame, start_bound, end_bound) -> bool:
    """Require both requested date boundaries, not merely an overlap."""
    if frame is None or frame.empty:
        return False
    first_timestamp = pd.Timestamp(frame.index.min())
    last_timestamp = pd.Timestamp(frame.index.max())
    return (
        (start_bound is None or first_timestamp <= start_bound)
        and (end_bound is None or last_timestamp >= end_bound)
    )


def _fetch_open_meteo_segment(url, fields, lat, lon, start_bound, end_bound):
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_bound.strftime("%Y-%m-%d"),
        "end_date": end_bound.strftime("%Y-%m-%d"),
        "hourly": ",".join(fields),
        "timezone": "UTC",
    }
    return normalize_weather_frame(
        _request_external_json(url, params), start_bound, end_bound)


def fetch_open_meteo_weather_frame(lat, lon, start_bound, end_bound):
    """Fetch complete archive/forecast coverage when `/land` cannot provide it."""
    if start_bound is None or end_bound is None:
        raise ValueError("Open-Meteo fallback requires start_date and end_date")

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    archive_cutoff = today - pd.Timedelta(days=OPEN_METEO_ARCHIVE_DELAY_DAYS)
    frames = []

    if start_bound <= archive_cutoff:
        archive_end = min(end_bound, archive_cutoff)
        frames.append(_fetch_open_meteo_segment(
            OPEN_METEO_ARCHIVE_URL, OPEN_METEO_ARCHIVE_FIELDS,
            lat, lon, start_bound, archive_end,
        ))

    forecast_start = max(start_bound, archive_cutoff + pd.Timedelta(days=1))
    if forecast_start <= end_bound:
        frames.append(_fetch_open_meteo_segment(
            OPEN_METEO_FORECAST_URL, OPEN_METEO_FORECAST_FIELDS,
            lat, lon, forecast_start, end_bound,
        ))

    if not frames:
        raise ValueError("Open-Meteo fallback returned no weather segments")
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    if not _covers_requested_window(combined, start_bound, end_bound):
        raise ValueError(
            f"Open-Meteo did not cover requested window "
            f"{start_bound.date()}..{end_bound.date()}"
        )
    combined.attrs["provider"] = "Open-Meteo"
    return combined


def fetch_weather_frame(lat: float, lon: float, start_date=None, end_date=None) -> pd.DataFrame:
    start_bound = _normalize_weather_bound(
        start_date) if start_date is not None else None
    end_bound = _normalize_weather_bound(
        end_date) if end_date is not None else None
    if start_bound is not None and end_bound is not None and start_bound > end_bound:
        raise ValueError(
            f"start_date must be <= end_date (got {start_bound.date()} > {end_bound.date()})")

    fallback_reasons = []
    auth_error = None
    spaceiotbox_partial = None
    if coordinates_supported_by_spaceiotbox(lat, lon):
        try:
            # `/land` documents only lat/lon; never send unsupported range keys.
            full_frame = normalize_weather_frame(
                fetch_agro_climate("land", lat, lon))
            if _covers_requested_window(full_frame, start_bound, end_bound):
                frame = normalize_weather_frame(
                    full_frame, start_date=start_bound, end_date=end_bound)
                frame.attrs["provider"] = "SpaceIoTBox"
                return frame
            if not full_frame.empty:
                try:
                    spaceiotbox_partial = normalize_weather_frame(
                        full_frame, start_date=start_bound,
                        end_date=end_bound)
                except ValueError:
                    spaceiotbox_partial = None
                fallback_reasons.append(
                    "SpaceIoTBox returned only "
                    f"{full_frame.index.min().date()}..{full_frame.index.max().date()}"
                )
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status in (401, 403):
                auth_error = status
            fallback_reasons.append(f"SpaceIoTBox HTTP {status}")
        except (requests.RequestException, TypeError, ValueError, KeyError) as exc:
            fallback_reasons.append(f"SpaceIoTBox error: {exc}")
    else:
        fallback_reasons.append("coordinates outside SpaceIoTBox coverage")

    # Historical completeness is essential for GDD; partial SpaceIoTBox data
    # is discarded instead of producing a confidently wrong crop stage.
    try:
        fallback = fetch_open_meteo_weather_frame(
            lat, lon, start_bound, end_bound)
        if spaceiotbox_partial is not None and not spaceiotbox_partial.empty:
            # SpaceIoTBox remains authoritative on overlapping timestamps;
            # Open-Meteo supplies only the missing historical/forecast range.
            combined = spaceiotbox_partial.combine_first(fallback).sort_index()
            if not _covers_requested_window(combined, start_bound, end_bound):
                raise ValueError("merged providers did not cover the full window")
            combined.attrs["provider"] = "SpaceIoTBox + Open-Meteo"
            combined.attrs["fallback_reason"] = "; ".join(fallback_reasons)
            return combined
        fallback.attrs["fallback_reason"] = "; ".join(fallback_reasons)
        return fallback
    except Exception as fallback_exc:
        if auth_error is not None:
            raise PermissionError(
                f"SpaceIoTBox authentication failed ({auth_error}) and "
                f"Open-Meteo fallback failed: {fallback_exc}"
            ) from fallback_exc
        start_label = start_bound.date() if start_bound is not None else "-"
        end_label = end_bound.date() if end_bound is not None else "-"
        raise ValueError(
            f"No complete weather coverage for {start_label}.."
            f"{end_label}: {'; '.join(fallback_reasons)}; "
            f"Open-Meteo fallback failed: {fallback_exc}"
        ) from fallback_exc
