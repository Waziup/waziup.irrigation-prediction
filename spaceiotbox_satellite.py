"""SpaceIoTBox `/land` and EO/STAC vegetation-index adapter.

`/land` supplies a dated NDVI value. The EO proxy supplies derived NDVI and
NDRE Cloud-Optimized GeoTIFFs. We sample those published products directly;
raw-band scaling, cloud masking, and index calculation stay outside this app.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from spaceiotbox_client import (
    REQUEST_ATTEMPTS,
    REQUEST_TIMEOUT_SECONDS,
    _request_json,
    coordinates_supported_by_spaceiotbox,
    fetch_agro_climate,
    get_api_key,
    get_base_url,
)

log = logging.getLogger(__name__)

SATELLITE_COLUMNS = [
    "sat_ndvi", "sat_ndre", "sat_vv_db", "satellite_data_age",
    "sat_ndvi_age", "sat_ndre_age", "sat_ndvi_quality", "sat_ndre_quality",
    "sat_ndvi_cadence_hours", "sat_ndre_cadence_hours",
]


def _utc_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _find_vegetation_indices(payload) -> dict:
    """Read the documented `/land` vegetation section."""
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return {}
    vegetation = data.get("vegetation_indices", {})
    return vegetation if isinstance(vegetation, dict) else {}


def _land_observation(lat: float, lon: float, as_of) -> Optional[dict]:
    if not coordinates_supported_by_spaceiotbox(lat, lon):
        return None
    try:
        vegetation = _find_vegetation_indices(fetch_agro_climate("land", lat, lon))
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None
    ndvi = vegetation.get("NDVI", vegetation.get("ndvi"))
    observed_at = _utc_timestamp(
        vegetation.get("date") or vegetation.get("datetime") or vegetation.get("timestamp"))
    as_of_timestamp = _utc_timestamp(as_of) or pd.Timestamp.now(tz="UTC")
    if ndvi is None or observed_at is None or observed_at > as_of_timestamp:
        return None
    try:
        ndvi = float(ndvi)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ndvi) or not -1.0 <= ndvi <= 1.0:
        return None
    return {"timestamp": observed_at, "sat_ndvi": ndvi, "source": "agro_climate"}


def _bbox_contains(bbox: object, lat: float, lon: float) -> bool:
    try:
        west, south, east, north = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError, IndexError):
        return False
    return west <= float(lon) <= east and south <= float(lat) <= north


def _stac_path(href: str) -> str:
    """Route advertised upstream hrefs back through the configured proxy."""
    path = urlparse(str(href)).path
    marker = "/v1/eo/stac/"
    position = path.find(marker)
    if position < 0:
        raise ValueError(f"Unsupported STAC href: {href!r}")
    query = urlparse(str(href)).query
    return path[position:] + (f"?{query}" if query else "")


def _collection_for_point(lat: float, lon: float) -> Optional[str]:
    payload = _request_json("/v1/eo/stac/collections")
    collections = payload.get("collections", []) if isinstance(payload, dict) else []
    for collection in collections:
        bbox_list = collection.get("extent", {}).get("spatial", {}).get("bbox", [])
        if any(_bbox_contains(bbox, lat, lon) for bbox in bbox_list):
            return str(collection.get("id"))
    return None


def _item_timestamp(item: dict) -> Optional[pd.Timestamp]:
    properties = item.get("properties", {}) if isinstance(item, dict) else {}
    return _utc_timestamp(properties.get("datetime") or item.get("id"))


def _fetch_stac_items(lat: float, lon: float, as_of) -> tuple[Optional[str], list[dict]]:
    """Fetch collection items and filter locally; proxy query filters are ignored."""
    collection_id = _collection_for_point(lat, lon)
    if not collection_id:
        return None, []
    path = f"/v1/eo/stac/collections/{collection_id}/items"
    items: list[dict] = []
    visited: set[str] = set()
    while path and path not in visited:
        visited.add(path)
        payload = _request_json(path)
        if not isinstance(payload, dict):
            break
        items.extend(item for item in payload.get("features", []) if isinstance(item, dict))
        next_link = next(
            (link.get("href") for link in payload.get("links", [])
             if isinstance(link, dict) and link.get("rel") == "next"), None)
        path = _stac_path(next_link) if next_link else ""

    as_of_timestamp = _utc_timestamp(as_of) or pd.Timestamp.now(tz="UTC")
    filtered = []
    for item in items:
        timestamp = _item_timestamp(item)
        if timestamp is None or timestamp > as_of_timestamp:
            continue
        # Item bboxes can be tile unions; raster bounds are checked when sampled.
        if item.get("bbox") and not _bbox_contains(item["bbox"], lat, lon):
            continue
        item = dict(item)
        item["_timestamp"] = timestamp
        filtered.append(item)
    filtered.sort(key=lambda item: item["_timestamp"], reverse=True)
    return collection_id, filtered


def _asset(item: dict, index_name: str) -> Optional[dict]:
    for name, asset in item.get("assets", {}).items():
        if str(name).upper() == index_name.upper() and isinstance(asset, dict):
            return asset
    return None


def _proxy_asset_url(asset: dict) -> str:
    return f"{get_base_url()}{_stac_path(asset.get('href', ''))}"


def _sample_with_rasterio(url: str, lat: float, lon: float) -> Optional[float]:
    import rasterio
    from rasterio.warp import transform

    options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_TIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "GDAL_HTTP_CONNECTTIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "GDAL_HTTP_MAX_RETRY": str(max(0, REQUEST_ATTEMPTS - 1)),
        "GDAL_HTTP_RETRY_DELAY": "1",
    }
    api_key = get_api_key()
    if api_key:
        options["GDAL_HTTP_HEADERS"] = f"X-API-Key: {api_key}"
    with rasterio.Env(**options), rasterio.open(f"/vsicurl/{url}") as dataset:
        xs, ys = transform("EPSG:4326", dataset.crs, [float(lon)], [float(lat)])
        row, column = dataset.index(xs[0], ys[0])
        if not (0 <= row < dataset.height and 0 <= column < dataset.width):
            return None
        value = dataset.read(
            1, window=((row, row + 1), (column, column + 1)), masked=True)[0, 0]
        return None if np.ma.is_masked(value) else float(value)


def _sample_asset(asset: dict, lat: float, lon: float) -> Optional[float]:
    """Read one COG pixel through authenticated HTTP range requests."""
    url = _proxy_asset_url(asset)
    try:
        value = _sample_with_rasterio(url, lat, lon)
    except ImportError as exc:
        # Missing raster support must degrade to GDD, never to an unsafe native
        # library fallback that can terminate the worker process.
        log.warning("EO asset sampling is unavailable: %s", exc)
        return None
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning("EO asset sampling failed: %s", exc)
        return None
    if value is None or not np.isfinite(value) or not -1.0 <= value <= 1.0:
        return None
    return float(value)


def _index_dates(items: Iterable[dict], index_name: str) -> list[pd.Timestamp]:
    return sorted({item["_timestamp"] for item in items if _asset(item, index_name)})


def _cadence_hours(dates: Iterable[pd.Timestamp]) -> float:
    ordered = pd.DatetimeIndex(sorted(set(dates)))
    if len(ordered) < 2:
        return float("nan")
    gaps = np.diff(ordered.asi8) / 3.6e12
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if len(gaps) else float("nan")


def empirical_freshness_quality(age_hours: float, dates: Iterable[pd.Timestamp]) -> float:
    """Score age from the empirical survival curve of acquisition gaps.

    No crop-day cutoff is encoded. Confidence is the fraction of historical
    gaps at least as long as the current wait; beyond every observed gap the
    index is stale and contributes no satellite weight.
    """
    if not np.isfinite(age_hours):
        return 0.0
    if age_hours <= 0:
        return 1.0
    ordered = pd.DatetimeIndex(sorted(set(dates)))
    if len(ordered) < 2:
        return 0.0
    gaps = np.diff(ordered.asi8) / 3.6e12
    gaps = gaps[gaps > 0]
    return float(np.mean(gaps >= float(age_hours))) if len(gaps) else 0.0


def _latest_sample(items: list[dict], index_name: str, lat: float, lon: float) -> Optional[dict]:
    # Try acquisitions in order because a union item bbox can overstate one asset's coverage.
    for item in items:
        asset = _asset(item, index_name)
        if not asset:
            continue
        value = _sample_asset(asset, lat, lon)
        if value is not None:
            return {"timestamp": item["_timestamp"], "value": value}
    return None


def fetch_satellite_snapshot(
    lat: float,
    lon: float,
    as_of: Optional[object] = None,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """Return the newest point-valid NDVI and NDRE with cadence-based quality."""
    as_of_timestamp = _utc_timestamp(as_of) or pd.Timestamp.now(tz="UTC")
    land = _land_observation(lat, lon, as_of_timestamp)
    try:
        collection_id, items = _fetch_stac_items(lat, lon, as_of_timestamp)
    except (requests.RequestException, TypeError, ValueError, KeyError) as exc:
        log.warning("EO/STAC catalog fetch failed: %s", exc)
        collection_id, items = None, []
    if lookback_days is not None:
        cutoff = as_of_timestamp - pd.Timedelta(days=max(0, int(lookback_days)))
        items = [item for item in items if item["_timestamp"] >= cutoff]
        if land and land["timestamp"] < cutoff:
            land = None

    ndvi_dates = _index_dates(items, "NDVI")
    ndre_dates = _index_dates(items, "NDRE")
    ndvi = land
    # Avoid a raster request when `/land` is newer than every catalog NDVI.
    if not land or (ndvi_dates and ndvi_dates[-1] > land["timestamp"]):
        sample = _latest_sample(items, "NDVI", lat, lon)
        if sample and (not land or sample["timestamp"] > land["timestamp"]):
            ndvi = {**sample, "sat_ndvi": sample["value"], "source": "eo_stac"}
    ndre = _latest_sample(items, "NDRE", lat, lon)
    if ndvi is None and ndre is None:
        return pd.DataFrame(columns=SATELLITE_COLUMNS)

    ndvi_timestamp = ndvi.get("timestamp") if ndvi else None
    ndre_timestamp = ndre.get("timestamp") if ndre else None
    ndvi_age = ((as_of_timestamp - ndvi_timestamp).total_seconds() / 3600.0
                if ndvi_timestamp is not None else float("inf"))
    ndre_age = ((as_of_timestamp - ndre_timestamp).total_seconds() / 3600.0
                if ndre_timestamp is not None else float("inf"))
    # `/land` is another real observation and extends the empirical NDVI series.
    quality_ndvi_dates = ndvi_dates + ([land["timestamp"]] if land else [])
    observation = {
        "sat_ndvi": float(ndvi["sat_ndvi"]) if ndvi else np.nan,
        "sat_ndre": float(ndre["value"]) if ndre else np.nan,
        "sat_vv_db": np.nan,
        "satellite_data_age": min(ndvi_age, ndre_age),
        "sat_ndvi_age": ndvi_age,
        "sat_ndre_age": ndre_age,
        "sat_ndvi_quality": empirical_freshness_quality(ndvi_age, quality_ndvi_dates),
        "sat_ndre_quality": empirical_freshness_quality(ndre_age, ndre_dates),
        "sat_ndvi_cadence_hours": _cadence_hours(quality_ndvi_dates),
        "sat_ndre_cadence_hours": _cadence_hours(ndre_dates),
    }
    timestamp = max(value for value in (ndvi_timestamp, ndre_timestamp) if value is not None)
    frame = pd.DataFrame([observation], index=pd.DatetimeIndex([timestamp], name="Timestamp"))
    providers = []
    if land:
        providers.append("SpaceIoTBox agro_climate/land")
    if collection_id:
        providers.append(f"SpaceIoTBox EO/STAC ({collection_id})")
    frame.attrs.update({
        "provider": " + ".join(providers) or "unavailable",
        "ndvi_observed_at": ndvi_timestamp.isoformat() if ndvi_timestamp else None,
        "ndre_observed_at": ndre_timestamp.isoformat() if ndre_timestamp else None,
        "freshness_policy": "empirical_acquisition_intervals",
    })
    return frame[SATELLITE_COLUMNS]


def fetch_satellite_history(
    lat: float,
    lon: float,
    as_of: Optional[object] = None,
    lookback_days: Optional[int] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Return catalog acquisition history plus the current sampled values."""
    as_of_timestamp = _utc_timestamp(as_of) or pd.Timestamp.now(tz="UTC")
    try:
        _, items = _fetch_stac_items(lat, lon, as_of_timestamp)
    except (requests.RequestException, TypeError, ValueError, KeyError):
        items = []
    cutoff = None
    if lookback_days is not None:
        cutoff = as_of_timestamp - pd.Timedelta(days=max(0, int(lookback_days)))
        items = [item for item in items if item["_timestamp"] >= cutoff]
    rows = [{
        "timestamp": item["_timestamp"], "source": "eo_stac",
        "has_ndvi": bool(_asset(item, "NDVI")), "has_ndre": bool(_asset(item, "NDRE")),
        **{column: np.nan for column in SATELLITE_COLUMNS},
    } for item in items[:max(0, int(limit))]]
    # Add `/land` without re-sampling every COG. Runtime state already carries
    # the point sample; validation only needs acquisition times and real values.
    land = _land_observation(lat, lon, as_of_timestamp)
    if land is not None and cutoff is not None and land["timestamp"] < cutoff:
        land = None
    if land is not None:
        ndvi_dates = _index_dates(items, "NDVI") + [land["timestamp"]]
        ndvi_age = (as_of_timestamp - land["timestamp"]).total_seconds() / 3600.0
        rows.append({
            "timestamp": land["timestamp"], "source": "agro_climate",
            "has_ndvi": True, "has_ndre": False,
            **{column: np.nan for column in SATELLITE_COLUMNS},
            "sat_ndvi": land["sat_ndvi"], "sat_ndvi_age": ndvi_age,
            "satellite_data_age": ndvi_age,
            "sat_ndvi_quality": empirical_freshness_quality(ndvi_age, ndvi_dates),
            "sat_ndvi_cadence_hours": _cadence_hours(ndvi_dates),
        })
    columns = ["timestamp", "source", "has_ndvi", "has_ndre"] + SATELLITE_COLUMNS
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).drop_duplicates(subset=["timestamp", "source"])
    return frame.sort_values("timestamp").reset_index(drop=True)[columns]
