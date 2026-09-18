"""Persistent cache for the latest unified pipeline output per plot."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import hmac
import pickle
import math
import os


# Increment whenever the persisted payload shape changes.
CACHE_VERSION = 4


def _signature_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.sig")


def _signing_key(path: Path, create: bool) -> bytes | None:
    configured = os.getenv("IRRIGATION_CACHE_SIGNING_KEY", "")
    if configured:
        return configured.encode("utf-8")
    key_path = path.parent / ".cache_hmac_key"
    try:
        return key_path.read_bytes()
    except FileNotFoundError:
        if not create:
            return None
    key = os.urandom(32)
    temporary = key_path.with_suffix(".tmp")
    temporary.write_bytes(key)
    os.chmod(temporary, 0o600)
    temporary.replace(key_path)
    return key


def _signature(data: bytes, key: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def cache_path(plot_id, root=None) -> Path:
    cache_root = root or os.getenv("IRRIGATION_CACHE_DIR", "data/cache")
    return Path(cache_root) / f"saved_variables_plot_{plot_id}.pkl"


def save_cycle(plot, current_tension, threshold_timestamp, predictions, result=None, root=None) -> Path:
    """Atomically save the worker-compatible and unified cycle payload."""
    path = cache_path(plot.id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "currentSoilTension": current_tension,
        "threshold_timestamp": threshold_timestamp,
        "predictions": predictions,
        "pipeline_result": result if result is not None else getattr(plot, "pipeline_result", None),
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    key = _signing_key(path, create=True)
    signature_path = _signature_path(path)
    temporary_signature = signature_path.with_suffix(
        f"{signature_path.suffix}.tmp")
    with temporary_path.open("wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_signature.write_text(
        _signature(serialized, key), encoding="ascii")
    # Publish only a fully written payload.
    temporary_path.replace(path)
    temporary_signature.replace(signature_path)
    return path


def load_cycle(plot_id, root=None):
    """Load and validate a cache payload, returning (payload, reason)."""
    path = cache_path(plot_id, root)
    if not path.is_file():
        return None, "missing"
    try:
        serialized = path.read_bytes()
        key = _signing_key(path, create=False)
        signature = _signature_path(path).read_text(encoding="ascii").strip()
        if key is None or not hmac.compare_digest(
                signature, _signature(serialized, key)):
            return None, "invalid"
        payload = pickle.loads(serialized)
    except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError):
        return None, "invalid"
    if not isinstance(payload, dict) or payload.get("cache_version") != CACHE_VERSION:
        return None, "unsupported_version"
    # Worker values and the unified API result are restored as one snapshot.
    required = {
        "saved_at_utc",
        "currentSoilTension",
        "threshold_timestamp",
        "predictions",
        "pipeline_result",
    }
    if not required.issubset(payload):
        return None, "incomplete"
    return payload, None


def cache_age_hours(payload, now=None):
    """Return cache age in hours, or None when its timestamp is invalid."""
    try:
        saved_at = datetime.fromisoformat(str(payload["saved_at_utc"]).replace("Z", "+00:00"))
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        age = (current - saved_at.astimezone(timezone.utc)).total_seconds() / 3600.0
        # A future save time is not fresh evidence. This also rejects invalid
        # reference clocks instead of letting NaN bypass the expiry comparison.
        return age if math.isfinite(age) and age >= 0 else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def restore_cycle(plot, payload, max_age_hours, now=None):
    """Restore a fresh cache payload onto a plot.

    Returns ``(worker_tuple, reason)``. Cached inference is deliberately marked
    as fallback data so the actuator's live-only guard cannot irrigate from it.
    """
    age_hours = cache_age_hours(payload, now=now)
    if age_hours is None:
        return None, "invalid_timestamp"
    if age_hours > float(max_age_hours):
        return None, "stale"

    result = payload.get("pipeline_result")
    if result is None:
        return None, "missing_pipeline_result"

    current_tension = payload["currentSoilTension"]
    threshold_timestamp = payload["threshold_timestamp"]
    predictions = payload["predictions"]
    plot.threshold_timestamp = threshold_timestamp
    plot.predictions = predictions
    plot.pipeline_result = result
    # Cached forecasts remain visible but are never treated as live actuation data.
    plot._inference_source = "cache"
    plot._loaded_model_timestamp = payload["saved_at_utc"]
    plot.pipeline_cache_status = "loaded"
    plot.pipeline_cache_reason = None

    for name, value in (
        ("inference_source", "cache"),
        ("fallback_used", True),
        ("fallback_reason", "training_disabled_cache"),
    ):
        if hasattr(result, name):
            setattr(result, name, value)

    return (current_tension, threshold_timestamp, predictions), None


def record_unavailable(plot, reason):
    """Clear stale state and expose why no worker result is available."""
    # Do not leave an older result looking current after cache validation fails.
    plot.pipeline_result = None
    plot._inference_source = "unavailable"
    plot.pipeline_cache_status = "unavailable"
    plot.pipeline_cache_reason = str(reason)
