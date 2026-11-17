"""Bounded single-flight cache for slow, read-only UI computations."""
from collections import OrderedDict
import logging
import threading
import time


class ReadCache:
    def __init__(self, workers=4, capacity=256, clock=None):
        self._clock = clock or time.monotonic
        self._slots = threading.BoundedSemaphore(workers)
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._capacity = max(workers, capacity)

    def get(self, key, load, *, pending, error, ttl=300, ttl_for=None, failed=None, retain=None, max_wait=35):
        now = self._clock()
        with self._lock:
            while key not in self._entries and len(self._entries) >= self._capacity:
                old = next((k for k, v in self._entries.items() if not v['busy']), None)
                if old is None:
                    return error
                del self._entries[old]
            entry = self._entries.setdefault(key, dict(value=None, at=0., busy=False, queued_at=now))
            self._entries.move_to_end(key)
            value = entry['value']
            if entry['busy']:
                if now-entry['started_at'] >= max_wait:
                    return retain(value, error) if value is not None and retain else (value if value is not None else error)
                return value if value is not None else pending
            if value is not None and now-entry['at'] < entry.get('ttl', ttl):
                return value
            if not self._slots.acquire(blocking=False):
                return value if value is not None else (error if now-entry['queued_at'] >= max_wait else pending)
            entry.update(busy=True, started_at=now, queued_at=now)

        def run():
            result = error
            unhealthy = False
            try:
                loaded = load()
                if loaded is None:
                    raise ValueError('Background read returned no result')
                result = loaded
                unhealthy = bool(failed(result)) if failed else False
            except Exception:
                unhealthy = True
                logging.getLogger(__name__).exception('Read-only background calculation failed')
            try:
                lifetime = 30 if unhealthy else (ttl_for(result) if ttl_for else ttl)
                if unhealthy and value is not None and retain:
                    result = retain(value, result)
                with self._lock:
                    entry.update(value=result, at=self._clock(), busy=False, ttl=lifetime)
            except Exception:
                logging.getLogger(__name__).exception('Background read publication failed')
                with self._lock:
                    entry.update(value=error, at=self._clock(), busy=False, ttl=30)
            finally:
                self._slots.release()
        try:
            threading.Thread(target=run, daemon=True, name='ui-read').start()
        except RuntimeError:
            with self._lock:
                entry.update(value=error, at=self._clock(), busy=False, ttl=30)
            self._slots.release()
            return error
        return value if value is not None else pending


UI_READS = ReadCache()
