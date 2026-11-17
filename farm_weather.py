"""Canonical hourly weather: nonblocking cached reads and short atomic merges.

Providers run outside cache locks. Overlapping day requests coalesce; current
weather is fetched before older history. Training and UI share the same values.
"""
from io import StringIO
import hashlib
import json
import threading
import time
import pandas as pd
from spaceiotbox_client import fetch_weather_frame, _weather_inputs_complete, _timeout, provider_response_scope
from state_store import get_app_state_store

_CACHE_KEY = 'weather:hourly:shared:v1'
_CONDITION = threading.Condition()
_FLIGHTS = set()
_MERGE_LOCK = threading.Lock()  # Compatibility with simple in-memory test stores.


def _day(value):
    stamp = pd.Timestamp(value)
    return (stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')).normalize()


def _cached_frame(item, day, now, complete):
    try:
        retrieved = pd.Timestamp(item['retrieved_at'])
        age = (now-retrieved).total_seconds()
        established = day < now.normalize()-pd.Timedelta(days=5) and retrieved.normalize() > day+pd.Timedelta(days=5)
        if age < 0 or (not established and age >= 1800):
            return None
        frame = pd.read_json(StringIO(item['frame']), orient='split')
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = item.get('index_name')
        if frame.empty or (complete and (not _weather_inputs_complete(frame)
                or not pd.date_range(day, periods=24, freq='h').isin(frame.index).all())):
            return None
        frame.attrs.update(item['attrs'])
        return frame
    except (KeyError, TypeError, ValueError):
        return None


def _groups(days, now):
    # Keep recent weather out of an old archive block and publish it first.
    groups = []
    for day in days:
        recent = day > now.normalize()-pd.Timedelta(days=5)
        if (not groups or len(groups[-1]) >= 31
                or day != groups[-1][-1]+pd.Timedelta(days=1)
                or recent != (groups[-1][-1] > now.normalize()-pd.Timedelta(days=5))):
            groups.append([])
        groups[-1].append(day)
    return list(reversed(groups))


def _fetch_shared_weather(store, fetcher, lat, lon, start_date, end_date, *, now=None, stream_blocks=False):
    now = pd.Timestamp(now or pd.Timestamp.now(tz='UTC'))
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    began = time.monotonic()
    def observed_now():
        return now + pd.Timedelta(seconds=time.monotonic()-began)
    start, end = _day(start_date), _day(end_date)
    if start > end:
        raise ValueError('Weather start must be before its end')
    days = list(pd.date_range(start, end, freq='D'))
    location = [float(lat), float(lon)]
    key = 'weather:hourly:shared:v2:' + hashlib.sha256(json.dumps(location).encode()).hexdigest()[:24]
    identity = (str(getattr(store, 'path', id(store))), key)

    def record():
        result = store.get_metadata(key)
        if not isinstance(result, dict):
            legacy = store.get_metadata(_CACHE_KEY)
            result = legacy if isinstance(legacy, dict) and legacy.get('location') == location else {}
        return result

    def available():
        cached = record().get('days', {})
        return {day: frame for day in days
                if (frame := _cached_frame(cached.get(day.date().isoformat()), day, observed_now(), stream_blocks)) is not None}

    def publish_block(fetched):
        source = fetched.copy()
        source.index = pd.to_datetime(source.index, utc=True)
        updates = {}
        for day in days:
            frame = source[(source.index >= day) & (source.index < day+pd.Timedelta(days=1))].copy()
            if frame.empty or (stream_blocks and (not _weather_inputs_complete(frame)
                    or not pd.date_range(day, periods=24, freq='h').isin(frame.index).all())):
                continue
            frame.attrs.update(fetched.attrs)
            frame.attrs['retrieved_at'] = now.isoformat()
            updates[day.date().isoformat()] = {
                'frame': frame.to_json(orient='split', date_format='iso', double_precision=15),
                'index_name': frame.index.name, 'attrs': dict(frame.attrs), 'retrieved_at': frame.attrs['retrieved_at']}
        if not updates:
            return
        legacy = record()
        def merge(latest):
            latest = latest if isinstance(latest, dict) else legacy
            cached = dict(latest.get('days', {}))
            for date, item in updates.items():
                if _cached_frame(cached.get(date), _day(date), observed_now(), stream_blocks) is None:
                    cached[date] = item
            cutoff = min(start, _day(latest.get('retain_from', start)), now.normalize()-pd.Timedelta(days=730))
            return {'location': location, 'retain_from': cutoff.date().isoformat(),
                    'days': {date: item for date, item in cached.items() if date >= cutoff.date().isoformat()}}
        if callable(getattr(store, 'update_metadata', None)):
            store.update_metadata(key, merge)
        else:
            with _MERGE_LOCK:
                store.set_metadata(key, merge(store.get_metadata(key)))
        with _CONDITION:
            _CONDITION.notify_all()

    while True:
        ready = available()
        missing = [day for day in days if day not in ready]
        if not missing:
            frames = [ready[day] for day in days]
            result = pd.concat(frames).sort_index()
            result.attrs = dict(frames[0].attrs)
            result.attrs['provider'] = ' + '.join(dict.fromkeys(f.attrs.get('provider', 'unknown') for f in frames))
            result.attrs['retrieved_at'] = min(f.attrs['retrieved_at'] for f in frames)
            return result
        with _CONDITION:
            owned = [day for day in missing if (identity, day) not in _FLIGHTS]
            _FLIGHTS.update((identity, day) for day in owned)
            if not owned:
                # UI callers have an overall deadline; model callers retain
                # their existing provider retry policy.
                _CONDITION.wait(timeout=min(.1, _timeout()))
                continue
        try:
            # Recheck after claiming: another publisher may already have won.
            ready = available()
            errors = []
            for group in _groups([d for d in owned if d not in ready], now):
                try:
                    kwargs = {'on_block': publish_block} if stream_blocks else {}
                    fetched = fetcher(lat, lon, start_date=group[0].date().isoformat(),
                                      end_date=group[-1].date().isoformat(), **kwargs)
                    publish_block(fetched)
                    covered = available()
                    for day in group:
                        if day not in covered:
                            raise ValueError('Weather provider did not cover requested day '+day.date().isoformat())
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
        finally:
            with _CONDITION:
                _FLIGHTS.difference_update((identity, day) for day in owned)
                _CONDITION.notify_all()


def fetch_shared_weather(store, fetcher, lat, lon, start_date, end_date, *, now=None, stream_blocks=False):
    with provider_response_scope():
        return _fetch_shared_weather(store, fetcher, lat, lon, start_date, end_date,
                                     now=now, stream_blocks=stream_blocks)


def fetch_farm_weather_frame(lat, lon, start_date=None, end_date=None):
    if start_date is None or end_date is None:
        return fetch_weather_frame(lat, lon, start_date=start_date, end_date=end_date)
    return fetch_shared_weather(get_app_state_store(), fetch_weather_frame,
                                lat, lon, start_date, end_date, stream_blocks=True)
