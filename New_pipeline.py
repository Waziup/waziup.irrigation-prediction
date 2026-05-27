"""
New_pipeline.py -- Orchestration layer connecting the tension ML model
(create_model.py) with the crop phenology system.

Responsibilities:
  1. Update plot.threshold to the crop-model-aware value when enabled,
     so create_model.calc_threshold() uses the right number.
  2. Delegate all ML training and prediction to create_model.py.
  3. Provide a stale-prediction fallback.
  4. Expose the two entry points the threads call.

NOTE: This file was partially ported from a separate farm-pipeline project
that uses train_farm_models.py, build_farm_datasets.py, and manifest-based
model loading. None of those are used in this project. create_model.py is
the ML backend here.
"""

import logging
from typing import Tuple

import pandas as pd

import runtime_config

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _apply_dynamic_threshold(plot) -> None:
    """
    If use_dynamic_threshold is enabled, overwrite plot.threshold with the
    crop-model stress threshold so create_model.calc_threshold() and the UI
    display both use the phenology-aware value.

    Falls back silently to the configured static threshold on any error.
    """
    if not getattr(plot, 'use_dynamic_threshold', False):
        return
    try:
        from actuation import compute_runtime_dynamic_threshold

        dyn = compute_runtime_dynamic_threshold(plot)
        if dyn and dyn != plot.threshold:
            log.info(
                "[%s] Dynamic threshold updated: %.1f cbar (was %.1f cbar)",
                plot.user_given_name, dyn, plot.threshold,
            )
            plot.threshold = dyn
    except Exception as exc:
        log.warning(
            "[%s] Dynamic threshold computation failed, keeping static value: %s",
            plot.user_given_name, exc,
        )


def _fallback_predictions(plot) -> pd.DataFrame:
    """Return the last cached predictions if they are within the staleness limit."""
    existing = getattr(plot, "predictions", None)
    if (
        isinstance(existing, pd.DataFrame)
        and "smoothed_values" in existing.columns
        and not existing.empty
    ):
        loaded_at = getattr(plot, "_loaded_model_timestamp", None)
        max_age_h = float(
            getattr(runtime_config, "Max_fallback_prediction_age_hours", 24)
        )
        if loaded_at is not None:
            age_h = (
                pd.Timestamp.utcnow() - pd.Timestamp(loaded_at)
            ).total_seconds() / 3600.0
            if age_h > max_age_h:
                log.error(
                    "Fallback predictions stale for %s (age=%.1fh, max=%.1fh)",
                    plot.user_given_name, age_h, max_age_h,
                )
                return pd.DataFrame(columns=["smoothed_values"])
        return existing.copy()
    return pd.DataFrame(columns=["smoothed_values"])


def _last_sensor_reading(plot, predictions: pd.DataFrame) -> float:
    """Best-effort: return the most recent soil tension value."""
    try:
        if isinstance(plot.data, pd.DataFrame) and not plot.data.empty:
            col = "rolling_mean_grouped_soil"
            if col in plot.data.columns:
                return float(plot.data[col].iloc[-1])
    except Exception:
        pass
    if not predictions.empty and "smoothed_values" in predictions.columns:
        return float(predictions["smoothed_values"].iloc[-1])
    return 0.0


# -----------------------------------------------------------------------------
# Core cycle functions
# -----------------------------------------------------------------------------

def _run_full_training(plot) -> Tuple[float, str, pd.DataFrame]:
    """
    Full train + predict cycle.
    Updates the dynamic threshold, then delegates to create_model.main().
    """
    import create_model

    _apply_dynamic_threshold(plot)

    try:
        tension, threshold_ts, predictions = create_model.main(plot)
    except Exception as exc:
        log.error(
            "create_model.main() failed for %s: %s",
            plot.user_given_name, exc,
        )
        predictions = _fallback_predictions(plot)
        if predictions.empty:
            raise RuntimeError(
                f"Training failed and no fallback available for "
                f"'{plot.user_given_name}'"
            ) from exc
        tension = _last_sensor_reading(plot, predictions)
        threshold_ts = getattr(plot, "threshold_timestamp", "")
        return tension, threshold_ts, predictions

    plot._loaded_model_timestamp = pd.Timestamp.utcnow().isoformat() + "Z"
    plot._inference_source = "live"
    return tension, threshold_ts, predictions


def _run_prediction_only(plot) -> Tuple[float, str, pd.DataFrame]:
    """
    Prediction-only cycle (no retraining).
    Updates the dynamic threshold, then delegates to
    create_model.predict_with_updated_data().
    """
    import create_model

    _apply_dynamic_threshold(plot)

    try:
        tension, threshold_ts, predictions = create_model.predict_with_updated_data(
            plot
        )
    except Exception as exc:
        log.error(
            "predict_with_updated_data() failed for %s: %s",
            plot.user_given_name, exc,
        )
        predictions = _fallback_predictions(plot)
        if predictions.empty:
            raise RuntimeError(
                f"Prediction failed and no fallback available for "
                f"'{plot.user_given_name}'"
            ) from exc
        tension = _last_sensor_reading(plot, predictions)
        threshold_ts = getattr(plot, "threshold_timestamp", "")
        return tension, threshold_ts, predictions

    plot._loaded_model_timestamp = pd.Timestamp.utcnow().isoformat() + "Z"
    plot._inference_source = "live"
    return tension, threshold_ts, predictions


# -----------------------------------------------------------------------------
# Public entry points (called by threads)
# -----------------------------------------------------------------------------

def expert_main(plot) -> Tuple[float, str, pd.DataFrame]:
    """
    Full training cycle.
    Called by training_thread.TrainingThread.run().
    """
    return _run_full_training(plot)


def expert_predict_with_updated_data(plot) -> Tuple[float, str, pd.DataFrame]:
    """
    Hybrid prediction cycle called by prediction_thread.PredictionThread.run().

    Triggers a full retrain every train_period_days (same schedule as before).
    Between retrains uses the lighter predict_with_updated_data() path which
    re-fetches latest sensor data and runs the already-trained model.
    """
    now = pd.Timestamp.utcnow()
    last_training = getattr(plot, "_last_expert_training_at", None)
    retrain_hours = max(1, int(getattr(plot, "train_period_days", 1) * 24))

    should_retrain = (
        last_training is None
        or (now - last_training).total_seconds() >= retrain_hours * 3600
    )

    if should_retrain:
        tension, threshold_ts, predictions = _run_full_training(plot)
        plot._last_expert_training_at = now
    else:
        tension, threshold_ts, predictions = _run_prediction_only(plot)

    plot.threshold_timestamp = threshold_ts
    plot.predictions = predictions
    return tension, threshold_ts, predictions
