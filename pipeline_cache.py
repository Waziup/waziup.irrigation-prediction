"""Persistent cache for the latest unified pipeline output per plot."""

from datetime import datetime, timezone
from pathlib import Path
import pickle


# Increment whenever the persisted payload shape changes.
CACHE_VERSION = 2


def cache_path(plot_id, root="data/cache") -> Path:
    return Path(root) / f"saved_variables_plot_{plot_id}.pkl"


def save_cycle(plot, current_tension, threshold_timestamp, predictions, result=None, root="data/cache") -> Path:
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
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Publish only a fully written payload.
    temporary_path.replace(path)
    return path


def load_cycle(plot_id, root="data/cache"):
    """Load and validate a cache payload, returning (payload, reason)."""
    path = cache_path(plot_id, root)
    if not path.is_file():
        return None, "missing"
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
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
        return max(0.0, (current - saved_at.astimezone(timezone.utc)).total_seconds() / 3600.0)
    except (KeyError, TypeError, ValueError):
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
