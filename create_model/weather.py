"""Weather adapter shared by model training and future prediction.

SpaceIoTBox is always attempted first. The shared client accepts it only when
it covers the entire requested date window and otherwise obtains complete
Open-Meteo archive/forecast coverage. This module then trims that daily API
window to the tension model's exact timestamp interval.
"""

from __future__ import annotations

import pandas as pd

from spaceiotbox_client import LEGACY_WEATHER_COLUMNS, fetch_weather_frame
from utils import TimeUtils


def _plot_coordinates(plot):
    gps = getattr(plot, "gps_info", {}) or {}
    latitude = gps.get("latitude", gps.get("lattitude"))
    longitude = gps.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError("Plot coordinates are required for weather data")
    return float(latitude), float(longitude)


def _local_timestamp(value, timezone_name):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone_name)
    return timestamp.tz_convert(timezone_name)


def _fetch_model_window(start, end, plot):
    """Fetch daily bounds, then return exactly the model's timestamp window."""
    timezone_name = TimeUtils.for_plot(plot)
    start_local = _local_timestamp(start, timezone_name)
    end_local = _local_timestamp(end, timezone_name)
    if start_local > end_local:
        raise ValueError("Weather window start must not be after its end")

    latitude, longitude = _plot_coordinates(plot)
    frame = fetch_weather_frame(
        latitude,
        longitude,
        start_date=start_local,
        end_date=end_local,
    )
    provider_metadata = dict(frame.attrs)

    # The normalized provider contract is UTC; expose model features in the
    # plot timezone so their index matches the soil-tension sensor series.
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(timezone_name)
    frame = frame.loc[(frame.index >= start_local) & (frame.index <= end_local)]
    if frame.empty:
        raise ValueError(
            f"Weather provider returned no rows for exact model window "
            f"{start_local.isoformat()}..{end_local.isoformat()}"
        )

    # The shared provider includes a compatibility Timestamp column for API
    # consumers, but the model contract stores time only in its named index.
    # Keeping both makes later reset_index() calls fail with
    # "cannot insert Timestamp, already exists".
    frame = frame.drop(columns=["Timestamp"], errors="ignore")

    for column in LEGACY_WEATHER_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.attrs.update(provider_metadata)
    frame.attrs["model_window_start"] = start_local.isoformat()
    frame.attrs["model_window_end"] = end_local.isoformat()
    return frame


def get_historical_weather_api(data, plot):
    """Fetch weather matching the complete soil-sensor training interval."""
    if data is None or data.empty:
        raise ValueError("Cannot fetch training weather for an empty sensor frame")
    fetched = _fetch_model_window(data.index[0], data.index[-1], plot)
    # Keep the compatibility cache plot-specific; never mix farm timelines.
    plot.data_w = fetched
    return plot.data_w


def only_get_historical_weather_api(start, end, plot):
    """Fetch an explicit historical window used by CSV/debug prediction."""
    return _fetch_model_window(start, end, plot)


def get_weather_forecast_api(start_date, end_date, plot, data=None):
    """Fetch the configured future prediction window for the tension model."""
    del data  # Retained in the public signature for existing callers.
    return _fetch_model_window(start_date, end_date, plot)
