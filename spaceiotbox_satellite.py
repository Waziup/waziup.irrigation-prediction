from __future__ import annotations

import importlib
import math
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

# Import the SpaceIoTBox client; when loaded by SourceFileLoader the package import may fail,
# so try several fallbacks to locate the client module by filename.
try:
    if __package__:
        _spaceiotbox_client = importlib.import_module(
            ".spaceiotbox_client", package=__package__)
    else:
        _spaceiotbox_client = importlib.import_module("spaceiotbox_client")
except Exception:
    # Fallback: search for the file in the repository and load it directly
    from importlib.machinery import SourceFileLoader
    import os as _os

    _spaceiotbox_client = None
    for root, dirs, files in _os.walk(_os.getcwd()):
        if 'spaceiotbox_client.py' in files:
            candidate_path = _os.path.join(root, 'spaceiotbox_client.py')
            try:
                _spaceiotbox_client = SourceFileLoader(
                    'spaceiotbox_client_fallback', candidate_path).load_module()
                break
            except Exception:
                continue
    if _spaceiotbox_client is None:
        raise

build_headers = _spaceiotbox_client.build_headers
get_base_url = _spaceiotbox_client.get_base_url


SATELLITE_COLUMNS = [
    "sat_ndvi",
    "sat_ndre",
    "sat_vv_db",
    "satellite_data_age",
    "sat_ndvi_age",
]

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_LIMIT = 100

_RAW_BAND_KEYS = {
    "red": ["B04", "B4", "red"],
    "red_edge_1": ["B05", "B5", "rededge1", "red_edge_1"],
    "red_edge_2": ["B06", "B6", "rededge2", "red_edge_2"],
    "nir": ["B08", "B8", "nir"],
    "vv": ["VV", "vv"],
}

_PROPERTY_ALIASES = {
    "sat_ndvi": ["ndvi", "NDVI"],
    "sat_ndre": ["ndre", "NDRE"],
    "sat_vv_db": ["vv", "VV"],
}


def _utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_naive_utc(value) -> pd.Timestamp:
    ts = _utc_timestamp(value)
    return ts.tz_localize(None)


def _base_url() -> str:
    return get_base_url().rstrip("/")


def _stac_url(request_path: str) -> str:
    return f"{_base_url()}/v1/eo/stac{request_path}"


def _request_json(request_path: str, params: Optional[dict] = None) -> object:
    response = requests.get(
        _stac_url(request_path),
        params=params,
        headers=build_headers(),
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1)
def _discover_collection_records() -> List[dict]:
    try:
        payload = _request_json("/collections")
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return []

    if isinstance(payload, dict):
        collections = payload.get("collections", [])
    elif isinstance(payload, list):
        collections = payload
    else:
        collections = []

    return [item for item in collections if isinstance(item, dict)]


@lru_cache(maxsize=1)
def _discover_collection_names() -> List[str]:
    collections = _discover_collection_records()
    names: List[str] = []
    for item in collections:
        for key in ("id", "title", "description"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
                break
    return names


def _collection_hint(patterns: Iterable[str]) -> Optional[str]:
    collections = _discover_collection_names()
    if not collections:
        return None

    for collection in collections:
        lowered = collection.lower()
        if any(re.search(pattern, lowered) for pattern in patterns):
            return collection
    return None


def _classify_collection(collection_name: str) -> Optional[str]:
    lowered = collection_name.lower()
    if any(token in lowered for token in ["sentinel-2", "/s2", "s2", "s2_sr"]):
        return "s2"
    if any(token in lowered for token in ["sentinel-1", "/s1", "s1", "s1_grd"]):
        return "s1"
    if any(token in lowered for token in ["mod16", "modis_et", "pet"]):
        return "et"
    if any(token in lowered for token in ["mod11", "modis_lst", "lst"]):
        return "lst"
    return None


def _infer_source_from_item(item: dict, collection_name: str) -> Optional[str]:
    source = _classify_collection(collection_name)
    if source is not None:
        return source

    assets = _iter_assets(item)
    asset_keys = {str(key).lower() for key in assets.keys()}
    properties = item.get("properties", {})
    property_keys = {str(key).lower() for key in properties.keys()} if isinstance(
        properties, dict) else set()
    searchable_keys = asset_keys | property_keys

    if any(key.startswith("raw_sentinel2_") or key in {"ndvi", "ndvi_colormap", "tci", "tci_tn"} for key in searchable_keys):
        return "s2"
    if any(key in {"vv", "vh"} or key.startswith("raw_sentinel1_") for key in searchable_keys):
        return "s1"
    if any("lst" in key for key in searchable_keys):
        return "lst"
    if any(key in {"et", "pet"} or "evapotranspiration" in key for key in searchable_keys):
        return "et"
    return None


def _collection_bbox(collection: dict) -> Optional[List[float]]:
    extent = collection.get("extent", {})
    if not isinstance(extent, dict):
        return None
    spatial = extent.get("spatial", {})
    if not isinstance(spatial, dict):
        return None
    bbox_list = spatial.get("bbox", [])
    if not isinstance(bbox_list, list) or not bbox_list:
        return None
    first_bbox = bbox_list[0]
    if not isinstance(first_bbox, list) or len(first_bbox) < 4:
        return None
    try:
        return [float(first_bbox[0]), float(first_bbox[1]), float(first_bbox[2]), float(first_bbox[3])]
    except (TypeError, ValueError):
        return None


def _point_in_bbox(lon: float, lat: float, bbox: Optional[List[float]]) -> bool:
    if not bbox or len(bbox) < 4:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox[:4]
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _collection_items_href(collection: dict) -> Optional[str]:
    links = collection.get("links", [])
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("rel") == "items":
            href = link.get("href")
            if isinstance(href, str) and href:
                return href
    return None


def _normalize_items_payload(payload: object) -> List[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("features", payload.get("items", []))
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _collection_items(collection: dict) -> List[dict]:
    href = _collection_items_href(collection)
    if not href:
        return []
    try:
        response = requests.get(href, headers=build_headers(), timeout=45)
        response.raise_for_status()
        return _normalize_items_payload(response.json())
    except (requests.RequestException, TypeError, ValueError):
        return []


def _items_from_collections(lat: float, lon: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[dict]:
    collections = _discover_collection_records()
    if not collections:
        return []

    matching_items: List[dict] = []
    for collection in collections:
        if not isinstance(collection, dict):
            continue
        if not _point_in_bbox(lon, lat, _collection_bbox(collection)):
            continue

        collection_name = str(collection.get(
            "id") or collection.get("title") or "").strip()
        if not collection_name:
            continue

        for item in _collection_items(collection):
            item_timestamp = _normalize_timestamp(
                (item.get("properties", {}) if isinstance(
                    item.get("properties", {}), dict) else {}).get("datetime")
                or item.get("datetime")
            )
            if item_timestamp is None or item_timestamp < start_date or item_timestamp > end_date:
                continue

            item_bbox = item.get("bbox")
            if isinstance(item_bbox, list) and len(item_bbox) >= 4 and not _point_in_bbox(lon, lat, item_bbox):
                continue

            if "collection" not in item:
                item = dict(item)
                item["collection"] = collection_name

            matching_items.append(item)

    return matching_items


def _search_items(
    lat: float,
    lon: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    limit: int = DEFAULT_LIMIT,
) -> List[dict]:
    params = {
        "bbox": f"{lon},{lat},{lon},{lat}",
        "datetime": f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "limit": int(limit),
    }

    try:
        payload = _request_json("/search", params=params)
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return []

    if isinstance(payload, dict):
        items = payload.get("features", payload.get("items", []))
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _iter_assets(item: dict) -> Dict[str, dict]:
    assets = item.get("assets", {})
    return assets if isinstance(assets, dict) else {}


def _sample_asset(asset_href: str, lat: float, lon: float) -> Optional[float]:
    if importlib.util.find_spec("rasterio") is None:
        return None

    try:
        rasterio = importlib.import_module("rasterio")
        with rasterio.open(asset_href) as dataset:
            sample = next(dataset.sample([(lon, lat)], masked=True))
            if len(sample) == 0:
                return None
            value = sample[0]
            if np.ma.is_masked(value):
                return None
            return float(value)
    except (ImportError, StopIteration, TypeError, ValueError, OSError):
        return None


def _extract_property_value(properties: dict, aliases: Iterable[str]) -> Optional[float]:
    for alias in aliases:
        if alias in properties:
            try:
                return float(properties[alias])
            except (TypeError, ValueError):
                continue
    return None


def _extract_band_values(item: dict, lat: float, lon: float) -> Dict[str, float]:
    properties = item.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    values: Dict[str, float] = {}

    for column, aliases in _PROPERTY_ALIASES.items():
        value = _extract_property_value(properties, aliases)
        if value is not None:
            values[column] = value

    assets = _iter_assets(item)
    for _alias_name, candidate_keys in _RAW_BAND_KEYS.items():
        if any(key in values for key in candidate_keys):
            continue
        for candidate_key in candidate_keys:
            asset = assets.get(candidate_key)
            if not isinstance(asset, dict):
                continue
            href = asset.get("href")
            if not isinstance(href, str) or not href:
                continue
            sample_value = _sample_asset(href, lat, lon)
            if sample_value is not None:
                values[candidate_key] = sample_value
                break

    return values


def _scale_optical_reflectance(value: float) -> float:
    if abs(value) > 1.5:
        return float(value) / 10000.0
    return float(value)


def _postprocess_collection_values(source: str, values: Dict[str, float]) -> Dict[str, float]:
    processed = dict(values)

    if source == "s2":
        for key in ("B04", "B4", "red", "B05", "B5", "rededge1", "red_edge_1", "B06", "B6", "rededge2", "red_edge_2", "B08", "B8", "nir"):
            if key in processed and processed[key] is not None and not pd.isna(processed[key]):
                processed[key] = _scale_optical_reflectance(
                    float(processed[key]))

    return processed


def _compute_indices(values: Dict[str, float]) -> Dict[str, float]:
    def _get(*keys: str) -> float:
        for key in keys:
            if key in values and values[key] is not None and not math.isnan(float(values[key])):
                return float(values[key])
        return float("nan")

    red = _get("B04", "B4", "red")
    red_edge_1 = _get("B05", "B5", "rededge1", "red_edge_1")
    red_edge_2 = _get("B06", "B6", "rededge2", "red_edge_2")
    nir = _get("B08", "B8", "nir")
    vv = _get("VV", "vv")

    computed = dict(values)

    if not math.isnan(nir) and not math.isnan(red) and (nir + red) != 0:
        computed["sat_ndvi"] = (nir - red) / (nir + red)

    if not math.isnan(red_edge_1) and not math.isnan(red_edge_2) and (red_edge_1 + red_edge_2) != 0:
        computed["sat_ndre"] = (red_edge_2 - red_edge_1) / \
            (red_edge_2 + red_edge_1)

    if not math.isnan(vv):
        computed["sat_vv_db"] = vv

    return computed


def _normalize_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        return _utc_timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _feature_row(item: dict, lat: float, lon: float, as_of: pd.Timestamp) -> Optional[dict]:
    props = item.get("properties", {})
    if not isinstance(props, dict):
        props = {}

    timestamp = _normalize_timestamp(
        props.get("datetime") or item.get("datetime"))
    if timestamp is None or timestamp > as_of:
        return None

    collection_name = str(item.get("collection")
                          or props.get("collection") or "")
    source = _infer_source_from_item(item, collection_name)
    if source is None:
        return None

    row = _compute_indices(_postprocess_collection_values(
        source, _extract_band_values(item, lat, lon)))
    row["timestamp"] = timestamp
    row["source"] = source
    row["collection"] = collection_name

    age_hours = float((as_of - timestamp).total_seconds() / 3600.0)
    row["satellite_data_age"] = age_hours
    if source == "s2":
        row["sat_ndvi_age"] = age_hours

    return row


def _latest_by_source(items: List[dict], lat: float, lon: float, as_of: pd.Timestamp) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {"s2": [], "s1": [], "lst": [], "et": []}
    for item in items:
        row = _feature_row(item, lat, lon, as_of)
        if row is None:
            continue
        source = row.pop("source")
        grouped.setdefault(source, []).append(row)

    for source, rows in grouped.items():
        rows.sort(key=lambda entry: entry["timestamp"])
    return grouped


def _merge_rows(rows: Iterable[dict]) -> dict:
    merged: Dict[str, float] = {column: np.nan for column in SATELLITE_COLUMNS}
    merged["satellite_data_age"] = np.inf
    merged["sat_ndvi_age"] = np.inf

    for row in rows:
        for key, value in row.items():
            if key in ("timestamp", "collection"):
                continue
            if value is None or pd.isna(value):
                continue

            if key in ("satellite_data_age", "sat_ndvi_age"):
                current = merged.get(key, np.inf)
                try:
                    merged[key] = min(float(current), float(value))
                except (TypeError, ValueError):
                    merged[key] = value
                continue

            current = merged.get(key)
            if current is None or pd.isna(current):
                merged[key] = value

        if "timestamp" in row:
            existing_ts = merged.get("timestamp")
            if not isinstance(existing_ts, pd.Timestamp) or row["timestamp"] > existing_ts:
                merged["timestamp"] = row["timestamp"]

    return merged


def _fetch_agro_climate_vegetation(lat: float, lon: float, as_of: pd.Timestamp) -> Optional[dict]:
    """
    Fetch vegetation indices from SpaceIoTBox agro_climate endpoint.

    Returns sat_ndvi and freshness metadata when available.
    """
    try:
        base = get_base_url().rstrip("/")
        url = f"{base}/v1/agro_climate/land?lat={lat}&lon={lon}"
        response = requests.get(url, headers=build_headers(), timeout=45)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, TypeError, ValueError):
        return None

    try:
        veg = data.get("data", {}).get("vegetation_indices", {})
        if not isinstance(veg, dict):
            return None

        ndvi = veg.get("NDVI")
        date_str = veg.get("date")

        if ndvi is None:
            return None

        row = {}
        if ndvi is not None:
            try:
                row["sat_ndvi"] = float(ndvi)
            except (TypeError, ValueError):
                pass

        # Compute age from date string
        if date_str:
            try:
                veg_ts = _normalize_timestamp(date_str)
                if veg_ts is not None:
                    row["timestamp"] = veg_ts
                    age_hours = float(
                        (as_of - veg_ts).total_seconds() / 3600.0)
                    row["satellite_data_age"] = age_hours
                    # Mark NDVI specifically for freshness
                    row["sat_ndvi_age"] = age_hours
            except (TypeError, ValueError):
                pass

        return row if row else None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


def fetch_satellite_snapshot(
    lat: float,
    lon: float,
    as_of: Optional[object] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Fetch the latest satellite snapshot for a point.

    Uses agro_climate for NDVI fallback and STAC proxy for NDVI/NDRE and VV.
    Derives indices from item metadata or asset samples when rasterio is available.
    """
    as_of_ts = _normalize_timestamp(as_of) or pd.Timestamp.now(tz="UTC")
    start_ts = as_of_ts - pd.Timedelta(days=int(lookback_days))

    items = _search_items(lat, lon, start_ts, as_of_ts, limit=DEFAULT_LIMIT)
    if not items:
        items = _items_from_collections(lat, lon, start_ts, as_of_ts)
    # If STAC items are not available, try agro_climate indices.
    if not items:
        agro_veg = _fetch_agro_climate_vegetation(lat, lon, as_of_ts)
        if agro_veg:
            for column in SATELLITE_COLUMNS:
                if column not in agro_veg:
                    agro_veg[column] = np.nan
            timestamp = agro_veg.pop("timestamp", as_of_ts)
            frame = pd.DataFrame(
                [agro_veg], index=pd.DatetimeIndex([timestamp]))
            frame.index.name = "Timestamp"
            frame = frame[SATELLITE_COLUMNS]
            return frame
        return pd.DataFrame(columns=SATELLITE_COLUMNS)

    grouped = _latest_by_source(items, lat, lon, as_of_ts)
    latest_rows: List[dict] = []

    s2_rows = grouped.get("s2", [])
    if s2_rows:
        s2_latest = s2_rows[-1]
        latest_rows.append(s2_latest)

    for source in ("s1",):
        rows = grouped.get(source, [])
        if rows:
            latest_rows.append(rows[-1])

    if not latest_rows:
        return pd.DataFrame(columns=SATELLITE_COLUMNS)

    merged = _merge_rows(latest_rows)
    timestamp = merged.pop("timestamp", as_of_ts)
    timestamp = _normalize_timestamp(timestamp) or as_of_ts

    for column in SATELLITE_COLUMNS:
        if column not in merged:
            merged[column] = np.nan

    # Backfill NDVI/freshness from agro_climate when STAC items exist but do not
    # provide numeric NDVI at this point.
    agro_veg = _fetch_agro_climate_vegetation(lat, lon, as_of_ts)
    if agro_veg:
        ndvi_value = merged.get("sat_ndvi")
        if ndvi_value is None or pd.isna(ndvi_value):
            if "sat_ndvi" in agro_veg and not pd.isna(agro_veg.get("sat_ndvi")):
                merged["sat_ndvi"] = float(agro_veg["sat_ndvi"])
        ndvi_age = merged.get("sat_ndvi_age")
        agro_ndvi_age = agro_veg.get("sat_ndvi_age")
        if agro_ndvi_age is not None and not pd.isna(agro_ndvi_age):
            if ndvi_age is None or pd.isna(ndvi_age) or float(ndvi_age) == float("inf"):
                merged["sat_ndvi_age"] = float(agro_ndvi_age)

    frame = pd.DataFrame([merged], index=pd.DatetimeIndex([timestamp]))
    frame.index.name = "Timestamp"
    frame = frame[SATELLITE_COLUMNS]
    return frame
