"""Canonical hourly farm weather shared by the UI, tension and phenology.

Cache complete UTC days so overlapping requests use identical values despite
requesting different training/season/forecast ranges. Consumers may aggregate
or trim timestamps; they must not fetch independent weather datasets.

Recent weather refreshes after 30 minutes. History fetched more than five days
after its observation date is retained without routine expiry. Successful
provider segments commit independently so outages cannot erase progress.
"""
from io import StringIO
import threading
import pandas as pd
from spaceiotbox_client import fetch_weather_frame, _weather_inputs_complete
from state_store import get_app_state_store

_LOCK = threading.Lock()
_CACHE_KEY = 'weather:hourly:shared:v1'


def fetch_shared_weather(store, fetcher, lat, lon, start_date, end_date, *, now=None, stream_blocks=False):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    def utc_day(value):
        stamp = pd.Timestamp(value)
        return (stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')).normalize()
    start, end = utc_day(start_date), utc_day(end_date)
    if start > end:
        raise ValueError('Weather start must be before its end')
    days = pd.date_range(start, end, freq='D')
    location = [float(lat), float(lon)]
    record = store.get_metadata(_CACHE_KEY)
    cached = record.get('days', {}) if isinstance(record, dict) and record.get('location') == location else {}
    available, missing = {}, []
    for day in days:
        key = day.date().isoformat()
        item = cached.get(key)
        try:
            age = (now - pd.Timestamp(item['retrieved_at'])).total_seconds()
            retrieved = pd.Timestamp(item['retrieved_at'])
            # History obtained after the archive delay is established data.
            established = day < now.normalize() - pd.Timedelta(days=5) and retrieved.normalize() > day + pd.Timedelta(days=5)
            if age < 0 or (not established and age >= 1800):
                raise ValueError('Expired')
            frame = pd.read_json(StringIO(item['frame']), orient='split')
            frame.index = pd.to_datetime(frame.index, utc=True)
            frame.index.name = item.get('index_name')
            if frame.empty:
                raise ValueError('Empty cached day')
            if stream_blocks and not _weather_inputs_complete(frame):
                raise ValueError('Incomplete cached weather inputs')
            frame.attrs.update(item['attrs'])
            available[key] = frame
        except (KeyError, TypeError, ValueError):
            missing.append(day)
    def publish_block(fetched):
        frame_source = fetched.copy()
        frame_source.index = pd.to_datetime(frame_source.index, utc=True)
        for day in days:
            frame = frame_source[(frame_source.index >= day) & (frame_source.index < day + pd.Timedelta(days=1))].copy()
            if frame.empty:
                continue
            if stream_blocks and not pd.date_range(day, periods=24, freq='h').isin(frame.index).all():
                # Never persist a partial day as complete coverage.
                continue
            if stream_blocks and not _weather_inputs_complete(frame):
                continue
            key = day.date().isoformat()
            # A streamed provider response must not overwrite a fresh cached day.
            if key in available:
                continue
            frame.attrs.update(fetched.attrs)
            frame.attrs['retrieved_at'] = now.isoformat()
            available[key] = frame
            cached[key] = {'frame': frame.to_json(orient='split', date_format='iso', double_precision=15),
                           'index_name': frame.index.name,
                           'attrs': dict(frame.attrs), 'retrieved_at': now.isoformat()}
        cutoff = min(start, now.normalize() - pd.Timedelta(days=730)).date().isoformat()
        retained = {key: item for key, item in cached.items() if key >= cutoff}
        # Commit each successful provider segment before another request begins.
        store.set_metadata(_CACHE_KEY, {'location': location, 'days': retained})

    if missing:
        groups = []
        for day in missing:
            if not groups or day != groups[-1][-1] + pd.Timedelta(days=1):
                groups.append([])
            groups[-1].append(day)
        for group in groups:
            kwargs = {'on_block': publish_block} if stream_blocks else {}
            fetched = fetcher(lat, lon, start_date=group[0].date().isoformat(),
                              end_date=group[-1].date().isoformat(), **kwargs)
            publish_block(fetched)
            for day in group:
                if day.date().isoformat() not in available:
                    raise ValueError('Weather provider did not cover requested day ' + day.date().isoformat())
    frames = [available[day.date().isoformat()] for day in days]
    result = pd.concat(frames).sort_index()
    result.attrs = dict(frames[0].attrs)
    providers = list(dict.fromkeys(frame.attrs.get('provider', 'unknown') for frame in frames))
    result.attrs['provider'] = ' + '.join(providers)
    result.attrs['retrieved_at'] = min(frame.attrs['retrieved_at'] for frame in frames)
    return result


def fetch_farm_weather_frame(lat, lon, start_date=None, end_date=None):
    if start_date is None or end_date is None:
        return fetch_weather_frame(lat, lon, start_date=start_date, end_date=end_date)
    # Serialize fetch+publish so UI/model/phenology cannot publish competing
    # weather versions when their requests arrive together.
    with _LOCK:
        return fetch_shared_weather(get_app_state_store(), fetch_weather_frame,
                                    lat, lon, start_date, end_date, stream_blocks=True)
