"""
Minimal runtime configuration for training and prediction threads.

Extracted from the legacy create_model.py to eliminate unnecessary imports.
"""

from dataclasses import dataclass

# Backoff time used after thread errors before retrying work.
Resource_wait_time_seconds = 1800

DEFAULT_SENSOR_SAMPLING_INTERVAL_MINUTES = 60
DEFAULT_FORECAST_HORIZON_DAYS = 5.0
DEFAULT_PREDICTION_INTERVAL_HOURS = 3.0
DEFAULT_IRRIGATION_CONFIRMATION_SECONDS = 10800


@dataclass(frozen=True)
class TimingConfig:
    """Resolved timing contract shared by model, workers, and actuation."""

    sensor_sampling_interval_minutes: int = DEFAULT_SENSOR_SAMPLING_INTERVAL_MINUTES
    prediction_interval_hours: float = DEFAULT_PREDICTION_INTERVAL_HOURS
    forecast_horizon_days: float = DEFAULT_FORECAST_HORIZON_DAYS
    retraining_interval_days: float = 1.0
    irrigation_confirmation_seconds: int = DEFAULT_IRRIGATION_CONFIRMATION_SECONDS
    error_retry_seconds: int = 1800


def get_timing_config(plot=None) -> TimingConfig:
    """Resolve timing from plot configuration with production-safe defaults."""
    # This is the single runtime timing adapter used by models and workers.
    values = TimingConfig(
        sensor_sampling_interval_minutes=int(getattr(
            plot, "forecast_interval_minutes",
            DEFAULT_SENSOR_SAMPLING_INTERVAL_MINUTES)),
        prediction_interval_hours=float(getattr(
            plot, "predict_period_hours", DEFAULT_PREDICTION_INTERVAL_HOURS)),
        forecast_horizon_days=float(getattr(
            plot, "forecast_horizon_days", DEFAULT_FORECAST_HORIZON_DAYS)),
        retraining_interval_days=float(getattr(
            plot, "retrain_interval_days", 1.0)),
        irrigation_confirmation_seconds=int(getattr(
            plot, "irrigation_confirmation_seconds",
            DEFAULT_IRRIGATION_CONFIRMATION_SECONDS)),
        error_retry_seconds=int(getattr(
            plot, "error_retry_seconds", Resource_wait_time_seconds)),
    )
    if values.sensor_sampling_interval_minutes <= 0:
        raise ValueError("sensor sampling interval must be positive")
    if values.prediction_interval_hours <= 0:
        raise ValueError("prediction interval must be positive")
    if values.forecast_horizon_days <= 0:
        raise ValueError("forecast horizon must be positive")
    if values.retraining_interval_days <= 0:
        raise ValueError("retraining interval must be positive")
    if values.irrigation_confirmation_seconds <= 0:
        raise ValueError("irrigation confirmation interval must be positive")
    if values.error_retry_seconds <= 0:
        raise ValueError("error retry interval must be positive")
    return values


# Fallback global base-model key used for unseen farms.
Global_base_model_key = "global_base"

# Enable periodic per-farm retraining when local farm datasets exist.
# Keep False for pure global-base inference deployments.
Enable_farm_specific_training = False

# Maximum age for cached fallback predictions before they are considered stale.
Max_fallback_prediction_age_hours = 24
