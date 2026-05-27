"""Incremental online calibration for production forecasts.

This module learns a small residual model from the latest prediction cycle
so the edge deployment can adapt between full batch retrains.

The calibrator is intentionally conservative: it only learns one residual
sample per cycle, and it only applies an adjustment once it has enough
history to avoid destabilizing the batch forecast.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ONLINE_DIR = BASE_DIR / "data" / "online_learning"
PENDING_STATE_NAME = "pending_cycle.json"
MODEL_STATE_NAME = "calibration_model.joblib"

FEATURE_NAMES = [
    "issue_tension",
    "forecast_last",
    "forecast_mean",
    "forecast_std",
    "forecast_min",
    "forecast_max",
    "forecast_slope",
    "forecast_span_hours",
    "forecast_count",
    "threshold",
    "issue_hour_sin",
    "issue_hour_cos",
    "issue_day_sin",
    "issue_day_cos",
]

MIN_SAMPLES_TO_APPLY = 3
MAX_RESIDUAL_ABS = 50.0
SAVE_EVERY_N_SAMPLES = 6
SAVE_MAX_INTERVAL_HOURS = 24.0

_PENDING_CACHE: Dict[str, Dict[str, float]] = {}
_SAVE_META: Dict[str, Dict[str, object]] = {}


def _as_timestamp(value: Optional[object]) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(datetime.utcnow())
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    hour = timestamp.hour + timestamp.minute / 60.0
    day_of_year = timestamp.dayofyear
    return {
        "issue_hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "issue_hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "issue_day_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "issue_day_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
    }


def _ensure_predictions_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["smoothed_values"])

    if "smoothed_values" not in predictions.columns:
        raise ValueError("predictions must include a 'smoothed_values' column")

    frame = predictions.copy()
    frame["smoothed_values"] = pd.to_numeric(
        frame["smoothed_values"], errors="coerce")
    frame = frame.dropna(subset=["smoothed_values"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _summarize_predictions(predictions: pd.DataFrame) -> Dict[str, float]:
    frame = _ensure_predictions_frame(predictions)
    if frame.empty:
        raise ValueError("predictions are empty")

    values = frame["smoothed_values"].astype(float)
    first_value = float(values.iloc[0])
    last_value = float(values.iloc[-1])
    span_hours = 0.0
    if len(frame.index) >= 2:
        span_hours = float(
            (frame.index[-1] - frame.index[0]).total_seconds() / 3600.0)

    slope = 0.0 if span_hours <= 0 else float(
        (last_value - first_value) / span_hours)

    return {
        "issue_tension": last_value,
        "forecast_last": last_value,
        "forecast_mean": float(values.mean()),
        "forecast_std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
        "forecast_min": float(values.min()),
        "forecast_max": float(values.max()),
        "forecast_slope": slope,
        "forecast_span_hours": span_hours,
        "forecast_count": float(len(values)),
    }


def _build_feature_row(
    predictions: pd.DataFrame,
    issue_tension: float,
    threshold: float,
    issue_timestamp: Optional[object] = None,
) -> Dict[str, float]:
    timestamp = _as_timestamp(issue_timestamp)
    summary = _summarize_predictions(predictions)
    summary["issue_tension"] = float(issue_tension)
    summary["threshold"] = float(threshold)
    summary.update(_time_features(timestamp))
    return summary


@dataclass
class OnlineCalibrationModel:
    scaler: StandardScaler = field(default_factory=StandardScaler)
    regressor: SGDRegressor = field(
        default_factory=lambda: SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            random_state=42,
        )
    )
    trained_samples: int = 0
    is_fitted: bool = False

    def partial_fit(self, feature_row: Dict[str, float], residual: float) -> None:
        x = np.asarray([[float(feature_row[name])
                       for name in FEATURE_NAMES]], dtype=float)
        y = np.asarray([float(residual)], dtype=float)

        self.scaler.partial_fit(x)
        x_scaled = self.scaler.transform(x)

        if not self.is_fitted:
            self.regressor.partial_fit(x_scaled, y)
            self.is_fitted = True
        else:
            self.regressor.partial_fit(x_scaled, y)

        self.trained_samples += 1

    def predict_residual(self, feature_row: Dict[str, float]) -> float:
        if not self.is_fitted or self.trained_samples < MIN_SAMPLES_TO_APPLY:
            return 0.0

        x = np.asarray([[float(feature_row[name])
                       for name in FEATURE_NAMES]], dtype=float)
        x_scaled = self.scaler.transform(x)
        return float(self.regressor.predict(x_scaled)[0])


def _farm_dir(farm_key: str) -> Path:
    farm_dir = ONLINE_DIR / farm_key
    farm_dir.mkdir(parents=True, exist_ok=True)
    return farm_dir


def _model_path(farm_key: str) -> Path:
    return _farm_dir(farm_key) / MODEL_STATE_NAME


def _state_path(farm_key: str) -> Path:
    return _farm_dir(farm_key) / PENDING_STATE_NAME


def _load_model(farm_key: str) -> OnlineCalibrationModel:
    model_path = _model_path(farm_key)
    if not model_path.exists():
        return OnlineCalibrationModel()

    try:
        model = joblib.load(model_path)
        if isinstance(model, OnlineCalibrationModel):
            return model
    except (OSError, EOFError, ValueError, TypeError, AttributeError) as exc:
        log.warning("Failed to load online model for %s: %s", farm_key, exc)

    return OnlineCalibrationModel()


def _save_model(farm_key: str, model: OnlineCalibrationModel) -> None:
    joblib.dump(model, _model_path(farm_key))


def _load_pending_record(farm_key: str) -> Optional[Dict[str, float]]:
    cached = _PENDING_CACHE.get(farm_key)
    if isinstance(cached, dict):
        return cached

    state_path = _state_path(farm_key)
    if not state_path.exists():
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("Failed to load online state for %s: %s", farm_key, exc)
        return None

    pending = payload.get("pending_cycle")
    if not isinstance(pending, dict):
        return None
    _PENDING_CACHE[farm_key] = pending
    return pending


def _save_pending_record(farm_key: str, record: Dict[str, float], trained_samples: int) -> None:
    state_path = _state_path(farm_key)
    payload = {
        "farm_key": farm_key,
        "trained_samples": int(trained_samples),
        "updated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "pending_cycle": record,
    }
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _should_flush_state(farm_key: str, trained_samples: int) -> bool:
    now = datetime.utcnow()
    meta = _SAVE_META.get(farm_key)
    if meta is None:
        return True

    last_samples = int(meta.get("trained_samples", -1))
    last_saved_at = meta.get("saved_at")
    if isinstance(last_saved_at, datetime):
        elapsed_h = (now - last_saved_at).total_seconds() / 3600.0
    else:
        elapsed_h = SAVE_MAX_INTERVAL_HOURS + 1.0

    if trained_samples != last_samples and trained_samples > 0:
        if trained_samples % SAVE_EVERY_N_SAMPLES == 0:
            return True

    return elapsed_h >= SAVE_MAX_INTERVAL_HOURS


def _flush_state(
    farm_key: str,
    model: OnlineCalibrationModel,
    record: Dict[str, float],
) -> None:
    _save_model(farm_key, model)
    _save_pending_record(farm_key, record, model.trained_samples)
    _SAVE_META[farm_key] = {
        "trained_samples": int(model.trained_samples),
        "saved_at": datetime.utcnow(),
    }


def _apply_residual_correction(predictions: pd.DataFrame, correction: float) -> pd.DataFrame:
    if predictions.empty or correction == 0.0:
        return predictions

    adjusted = predictions.copy()
    adjusted["smoothed_values"] = (
        adjusted["smoothed_values"].astype(
            float) + float(np.clip(correction, -MAX_RESIDUAL_ABS, MAX_RESIDUAL_ABS))
    )
    return adjusted


def update_predictions(
    farm_key: str,
    predictions: pd.DataFrame,
    observed_tension: float,
    threshold: float,
    issue_timestamp: Optional[object] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Update the online residual model and return corrected predictions.

    The model is trained on the most recent pending cycle once a new observed
    tension arrives, then the updated model is used to calibrate the current
    forecast.
    """
    frame = _ensure_predictions_frame(predictions)
    if frame.empty:
        return frame, {"applied_correction": 0.0, "trained_samples": 0}

    model = _load_model(farm_key)
    previous_pending = _load_pending_record(farm_key)

    if previous_pending is not None:
        try:
            residual = float(observed_tension) - \
                float(previous_pending["forecast_last"])
            if np.isfinite(residual):
                model.partial_fit(previous_pending, residual)
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("Skipping online update for %s: %s", farm_key, exc)

    current_record = _build_feature_row(
        frame,
        issue_tension=observed_tension,
        threshold=threshold,
        issue_timestamp=issue_timestamp,
    )
    _PENDING_CACHE[farm_key] = current_record

    correction = model.predict_residual(current_record)
    adjusted = _apply_residual_correction(frame, correction)

    if _should_flush_state(farm_key, model.trained_samples):
        _flush_state(farm_key, model, current_record)

    return adjusted, {
        "applied_correction": float(np.clip(correction, -MAX_RESIDUAL_ABS, MAX_RESIDUAL_ABS)),
        "trained_samples": int(model.trained_samples),
    }
