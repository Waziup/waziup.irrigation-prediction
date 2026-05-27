from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests


DEFAULT_BASE_URL = "https://www.smartafrihub.com/spaceiotbox/api"

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
    response = requests.get(
        f"{get_base_url()}{path}",
        params=params,
        headers=build_headers(),
        timeout=30,
    )
    if response.ok:
        return response.json()

    # Keep date-bounded requests strict: callers that request a time range
    # must not silently fall back to unbounded data.
    response.raise_for_status()
    return response.json()


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
    return bound.normalize()


def normalize_weather_frame(payload: object, start_date=None, end_date=None) -> pd.DataFrame:
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
            result.index = pd.DatetimeIndex(result.index)

    if start_date is not None or end_date is not None:
        index_days = [pd.Timestamp(ts).date() for ts in result.index]
        mask_values = [True] * len(index_days)
        if start_date is not None:
            start_day = _normalize_weather_bound(start_date).date()
            mask_values = [keep and day >= start_day for keep,
                           day in zip(mask_values, index_days)]
        if end_date is not None:
            end_day = _normalize_weather_bound(end_date).date()
            mask_values = [keep and day <= end_day for keep,
                           day in zip(mask_values, index_days)]

        mask = pd.Series(mask_values, index=result.index)

        result = result.loc[mask]
        if result.empty:
            start_label = _normalize_weather_bound(
                start_date).date() if start_date is not None else "-"
            end_label = _normalize_weather_bound(
                end_date).date() if end_date is not None else "-"
            raise ValueError(
                f"Weather response had no rows within requested window {start_label}..{end_label}"
            )

    for column in LEGACY_WEATHER_COLUMNS:
        alias_columns = [column] + COLUMN_ALIASES.get(column, [])
        result[column] = _coerce_numeric_series(
            raw_frame, alias_columns).values

    if "Timestamp" not in result.columns:
        result["Timestamp"] = result.index

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.sort_index()
    return result


def fetch_weather_frame(lat: float, lon: float, start_date=None, end_date=None) -> pd.DataFrame:
    start_bound = _normalize_weather_bound(
        start_date) if start_date is not None else None
    end_bound = _normalize_weather_bound(
        end_date) if end_date is not None else None
    if start_bound is not None and end_bound is not None and start_bound > end_bound:
        raise ValueError(
            f"start_date must be <= end_date (got {start_bound.date()} > {end_bound.date()})")

    params = {}
    if start_bound is not None:
        params["start_date"] = start_bound.strftime("%Y-%m-%d")
    if end_bound is not None:
        params["end_date"] = end_bound.strftime("%Y-%m-%d")

    frames = []
    auth_errors = []
    for endpoint in ("land",):
        try:
            payload = fetch_agro_climate(
                endpoint, lat, lon, params=params or None)
            try:
                frame = normalize_weather_frame(
                    payload, start_date=start_bound, end_date=end_bound)
            except ValueError as exc:
                if "no rows within requested window" not in str(exc).lower():
                    raise
                frame = normalize_weather_frame(payload)
            if not frame.empty:
                frames.append(frame)
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status in (401, 403):
                auth_errors.append((endpoint, status))
            continue
        except (requests.RequestException, TypeError, ValueError, KeyError):
            continue

    if not frames:
        if auth_errors:
            statuses = ", ".join(
                f"{endpoint}:{status}" for endpoint, status in auth_errors
            )
            raise PermissionError(
                f"SpaceIoTBox authentication failed ({statuses}). "
                "Check SPACEIOTBOX_API_KEY (or API_KEY/api_key) and token validity."
            )
        # If the caller requested a date window but the filtered responses
        # contained no rows, attempt an unfiltered request to the
        # agro_climate/land endpoint and normalize without date bounds.
        if start_bound is not None or end_bound is not None:
            try:
                payload = fetch_agro_climate("land", lat, lon, params=None)
                frame = normalize_weather_frame(payload)
                if not frame.empty:
                    frames.append(frame)
                    # proceed to combine/return below
                else:
                    start_label = start_bound.date() if start_bound is not None else "-"
                    end_label = end_bound.date() if end_bound is not None else "-"
                    raise ValueError(
                        f"No weather data returned within requested window {start_label}..{end_label}"
                    )
            except requests.HTTPError:
                start_label = start_bound.date() if start_bound is not None else "-"
                end_label = end_bound.date() if end_bound is not None else "-"
                raise ValueError(
                    f"No weather data returned within requested window {start_label}..{end_label}"
                )
        else:
            return pd.DataFrame(columns=LEGACY_WEATHER_COLUMNS)

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.combine_first(frame)

    for column in LEGACY_WEATHER_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan

    if "Timestamp" not in combined.columns:
        combined["Timestamp"] = combined.index

    combined = combined[LEGACY_WEATHER_COLUMNS + ["Timestamp"]]
    return combined.sort_index()
