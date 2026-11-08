"""Shared plot EO observations for display, phenology and crop calculations.

Satellite basemap tiles are unrelated. This cache contains actual timestamped
NDVI/NDRE observations at the saved plot reference point, without training.
"""
import hashlib
from io import StringIO
import json
import threading
import numpy as np
import pandas as pd
import spaceiotbox_satellite as provider
from state_store import get_app_state_store

_LOCK = threading.Lock()
# Check availability daily; acquisitions themselves can be days apart.
TTL_SECONDS = 24 * 60 * 60


def shared_history(store, fetcher, lat, lon, *, as_of=None, lookback_days=None,
                   limit=100, include_ndre=True, now=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    as_of = provider._utc_timestamp(as_of) or now
    lookback = max(0, int(lookback_days)) if lookback_days is not None else 365
    canonical_days, canonical_limit = max(365, lookback), max(100, int(limit))
    key = 'satellite:plot:' + hashlib.sha256(json.dumps(
        [float(lat), float(lon)]).encode()).hexdigest()
    record = store.get_metadata(key)
    history = None
    try:
        age = (now - pd.Timestamp(record['retrieved_at'])).total_seconds()
        if (0 <= age < (300 if record.get('empty') else TTL_SECONDS) and abs((as_of - pd.Timestamp(record['as_of'])).total_seconds()) < TTL_SECONDS
                and record['lookback_days'] >= canonical_days
                and record['limit'] >= canonical_limit):
            history = pd.read_json(StringIO(record['frame']), orient='split', precise_float=True)
            if history.empty and age >= 300:
                history = None
    except (KeyError, TypeError, ValueError):
        pass
    if history is None:
        history = fetcher(lat, lon, as_of=as_of, lookback_days=canonical_days,
                          limit=canonical_limit, include_ndre=True)
        # Empty results remain empty, not fabricated zero index values.
        store.set_metadata(key, {'retrieved_at': now.isoformat(), 'as_of': as_of.isoformat(),
            'lookback_days': canonical_days, 'limit': canonical_limit, 'empty': history.empty,
            'frame': history.to_json(orient='split', date_format='iso', double_precision=15)})
    history = history.copy()
    if history.empty:
        return history
    history['timestamp'] = pd.to_datetime(history['timestamp'], utc=True)
    history = history[(history['timestamp'] <= as_of) &
                      (history['timestamp'] >= as_of - pd.Timedelta(days=lookback))]
    history = history.sort_values('timestamp', kind='stable').tail(max(0, int(limit))).copy()
    for index in ('ndvi', 'ndre'):
        column = 'sat_' + index
        values = pd.to_numeric(history[column], errors='coerce')
        valid = values.between(-1, 1)
        dates = list(history.loc[valid, 'timestamp'])
        ages = (as_of - history['timestamp']).dt.total_seconds() / 3600
        history[column + '_age'] = ages.where(valid, np.inf)
        history[column + '_quality'] = [provider._observation_quality(age, dates) if ok else 0.
            for age, ok in zip(ages, valid)]
        history[column + '_cadence_hours'] = provider._cadence_hours(dates)
    history['satellite_data_age'] = history[['sat_ndvi_age', 'sat_ndre_age']].min(axis=1)
    return history.reset_index(drop=True)


def snapshot_from_history(history, *, as_of=None, include_ndre=True):
    as_of = provider._utc_timestamp(as_of) or pd.Timestamp.now(tz='UTC')
    if history.empty:
        return pd.DataFrame(columns=provider.SATELLITE_COLUMNS)
    result = {column: np.nan for column in provider.SATELLITE_COLUMNS}
    for index in ('ndvi', 'ndre'):
        result['sat_' + index + '_age'] = np.inf
        result['sat_' + index + '_quality'] = 0.
    observed = {}
    for index in ('ndvi', 'ndre'):
        if index == 'ndre' and not include_ndre:
            continue
        column = 'sat_' + index
        rows = history[pd.to_numeric(history[column], errors='coerce').between(-1, 1)]
        rows = rows.sort_values('timestamp', kind='stable').drop_duplicates('timestamp', keep='last')
        if rows.empty:
            result[column + '_age'], result[column + '_quality'] = np.inf, 0.
            continue
        latest = rows.iloc[-1]
        timestamp = pd.Timestamp(latest['timestamp'])
        observed[index] = timestamp
        result[column] = float(latest[column])
        result[column + '_age'] = max(0., (as_of - timestamp).total_seconds() / 3600)
        result[column + '_quality'] = provider._observation_quality(result[column + '_age'], list(rows['timestamp']))
        result[column + '_cadence_hours'] = provider._cadence_hours(list(rows['timestamp']))
    if not observed:
        return pd.DataFrame(columns=provider.SATELLITE_COLUMNS)
    result['satellite_data_age'] = min(result['sat_ndvi_age'], result.get('sat_ndre_age', np.inf))
    frame = pd.DataFrame([result], index=pd.DatetimeIndex([max(observed.values())], name='Timestamp'))
    frame.attrs.update(provider='SpaceIoTBox agro-climate/land + EO/STAC',
        ndvi_observed_at=observed['ndvi'].isoformat() if 'ndvi' in observed else None,
        ndre_observed_at=observed['ndre'].isoformat() if 'ndre' in observed else None,
        freshness_policy='empirical_acquisition_intervals')
    return frame[provider.SATELLITE_COLUMNS]


def fetch_satellite_history(lat, lon, as_of=None, lookback_days=None, limit=100, include_ndre=True):
    with _LOCK:
        return shared_history(get_app_state_store(), provider.fetch_satellite_history,
            lat, lon, as_of=as_of, lookback_days=lookback_days, limit=limit, include_ndre=include_ndre)


def fetch_satellite_snapshot(lat, lon, as_of=None, lookback_days=None, include_ndre=True):
    as_of = provider._utc_timestamp(as_of) or pd.Timestamp.now(tz='UTC')
    history = fetch_satellite_history(lat, lon, as_of=as_of, lookback_days=lookback_days,
                                      limit=100, include_ndre=True)
    return snapshot_from_history(history, as_of=as_of, include_ndre=include_ndre)
