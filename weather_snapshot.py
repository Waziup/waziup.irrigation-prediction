"""Persist the farm forecast separately from agronomic weather history.

A 30-minute gateway TTL permits a foreground check without fetching on every
screen load. Clients normally poll every two hours. Failed provider requests
must never advance the snapshot's generation time or make it look fresh.
"""
from datetime import datetime, timezone
import hashlib
import json

FRESH_SECONDS = 30 * 60


def snapshot_key(farm, latitude, longitude, start_date, horizon_days):
    identity = [farm.get('farm_id'), latitude, longitude,
                farm.get('timezone'), start_date, horizon_days]
    return 'weather:forecast:' + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_snapshot(store, key, *, allow_stale=False, now=None):
    record = store.get_metadata('weather:history:snapshot' if ':historical:' in key else 'weather:forecast:snapshot')
    if not isinstance(record, dict) or record.get('key') != key:
        return None
    payload = record.get('payload')
    if not isinstance(payload, dict) or payload.get('available') is not True:
        return None
    try:
        generated = datetime.fromisoformat(payload['generated_at'].replace('Z', '+00:00'))
        age = ((now or datetime.now(timezone.utc)) - generated).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None
    if age < 0 or (age >= FRESH_SECONDS and not allow_stale):
        return None
    result = dict(payload)
    result['cache_state'] = 'stale' if age >= FRESH_SECONDS else 'fresh'
    return result


def save_snapshot(store, key, payload):
    if payload.get('available') is True:
        store.set_metadata('weather:history:snapshot' if ':historical:' in key else 'weather:forecast:snapshot', {'key': key, 'payload': payload})
