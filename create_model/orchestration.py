"""Unified training, prediction, crop-state, and decision orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import pycaret
from pycaret.regression import *
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from api_contract import json_safe
from utils import TimeUtils
from sensor_quality import evaluate_tension_sensor_safety

from . import state
from .constants import *
from .evaluation import eval_approach_mix, evaluate_against_testset, evaluate_against_testset_nn, evaluate_against_validation, evaluate_against_validation_nn, evaluate_results_and_choose_top_n
from .features import prepare_data, split_by_ratio
from .nn_architectures import adapt_X_for_model
from .nn_ensemble import EnsemblePredictor, compare_nn_ensembles
from .nn_training import init_nn_subprocess_tuning_and_ensemble, prepare_data_for_cnn2, prepare_future_values, save_models_nn, train_nn_models, tune_model_nn
from .prediction import align_with_latest_sensor_values, calc_threshold, compare_train_predictions_cols, create_future_values, generate_predictions, generate_predictions_nn
from .pycaret_models import create_and_compare_ensemble, create_and_compare_model_reg, init_pycaret_subprocess_tuning_and_ensemble, save_models, train_best, tune_models
from .runtime import free_memory
from .soil import add_volumetric_col_to_df


@dataclass
class PipelineCycleResult:
    """Complete backend result for one model, crop-state, and decision cycle."""

    # Stable contract consumed by the API, workers, cache, and actuator.
    current_tension: float
    threshold_timestamp: object
    predictions: pd.DataFrame
    tension_forecast: dict
    forecast_timestamps: list
    forecast_interval_hours: float | None
    forecast_horizon_hours: float | None
    stress_threshold_cbar: float | None
    stress_threshold_forecast: dict
    stress_threshold_timestamps: dict
    crop_state: object | None
    recommendation: object | None
    data_freshness: dict
    data_sources: dict
    model_status: str
    inference_source: str
    calculated_at: str | None = None
    previous_calculated_at: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Return JSON-oriented metadata while retaining predictions separately."""
        crop_state = None
        if self.crop_state is not None:
            crop_state = asdict(self.crop_state)
            for name in (
                "satellite_validation",
                "weather_data_summary",
                "satellite_data_summary",
            ):
                if hasattr(self.crop_state, name):
                    crop_state[name] = getattr(self.crop_state, name)

        recommendation = None
        if self.recommendation is not None:
            recommendation = asdict(self.recommendation)
            for key, value in recommendation.items():
                if isinstance(value, (datetime, pd.Timestamp)):
                    recommendation[key] = value.isoformat()

        return json_safe({
            "current_tension": self.current_tension,
            "threshold_timestamp": (
                self.threshold_timestamp.isoformat()
                if hasattr(self.threshold_timestamp, "isoformat")
                else self.threshold_timestamp
            ),
            "tension_forecast": dict(self.tension_forecast),
            "forecast_timestamps": list(self.forecast_timestamps),
            "forecast_interval_hours": self.forecast_interval_hours,
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "stress_threshold_cbar": self.stress_threshold_cbar,
            "stress_threshold_forecast": dict(self.stress_threshold_forecast),
            "stress_threshold_timestamps": dict(self.stress_threshold_timestamps),
            "crop_state": crop_state,
            "recommendation": recommendation,
            "data_freshness": dict(self.data_freshness),
            "data_sources": dict(self.data_sources),
            "model_status": self.model_status,
            "inference_source": self.inference_source,
            "calculated_at": self.calculated_at,
            "previous_calculated_at": self.previous_calculated_at,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
        })


def _predict_with_updated_data_unlocked(plot):
    timezone_name = TimeUtils.for_plot(plot)
    # Run data pipeline to obtain latest data. training=False: reuse the scaler the frozen
    # model was trained under instead of refitting it on the newly accumulated data.
    train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler = data_pipeline(
        plot, training=False)
    # Create future value set to feed new data to model
    future_features = create_future_values(plot.data, plot)
    # Compare dataframes cols to be sure that they match, otherwise drop
    future_features = compare_train_predictions_cols(train, future_features)
    # NN
    if not plot.use_pycaret:
        Z, Z_scaled, Z_cnn = prepare_future_values(
            scaler, future_features, X_train.columns)
        plot.predictions = generate_predictions_nn(
            plot.best_model,
            Z_scaled,
            future_features.index[0],
            future_features.index[-1],
            interval_minutes=int(
                getattr(plot, "forecast_interval_minutes", 60)),
        )
    else:
        plot.predictions = generate_predictions(
            plot.best_model, plot.best_exp, future_features)

    # Cut passed time from predictions
    if not plot.load_data_from_csv:
        plot.predictions = plot.predictions.loc[pd.Timestamp((datetime.now()).replace(
            microsecond=0, second=0, minute=0)).tz_localize(timezone_name):]

    # Align predictions with historical data
    align_with_latest_sensor_values(plot)

    # Calculate when threshold will be meet
    plot.threshold_timestamp = calc_threshold(
        plot.predictions, 'smoothed_values', plot)

    # Add volumetric water content
    if plot.sensor_kind == 'tension':
        plot.predictions = add_volumetric_col_to_df(
            plot.predictions, "smoothed_values", plot)

    # Runs every 3h on the RPi: return the freed frames to the OS before idling.
    # No clear_keras - plot.best_model must stay usable for the next cycle.
    free_memory(label="prediction run")

    # Return last accumulated reading and threshold timestamp currentSoilTension, threshold_timestamp, predictions
    return plot.data['rolling_mean_grouped_soil'][-1], plot.threshold_timestamp, plot.predictions


def predict_with_updated_data(plot):
    """Run prediction while holding the process-wide model resource lock."""
    with state.model_operation():
        return _predict_with_updated_data_unlocked(plot)


def data_pipeline(plot, training=True):
    # Data preparation pipeline, calls other subfunction to perform the task
    # Classical regression
    plot.data = prepare_data(plot)

    # Search for gaps in data again (quick fix) => tackle problem with latest data "nan", in case of irrigations saved
    if plot.data.isna().any().any():
        # Data.drop(Data.index[-1], inplace=True)
        plot.data.dropna(inplace=True)

    # Split dataset
    # here a split is done to rule out the models that are overfitting
    train, val, test = split_by_ratio(plot.data)

    # NN
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler = prepare_data_for_cnn2(
        plot, train, val, test, 'rolling_mean_grouped_soil', training=training)

    return train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler


def _train_model_cycle_unlocked(plot) -> int:
    timezone_name = TimeUtils.for_plot(plot)
    print("Check version of pycaret:", pycaret.__version__, "should be >= 3.0")
    plot.config = plot.read_config()

    # ---------------------------
    # DATA PIPELINE
    # ---------------------------
    train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, \
        X_train_scaled, X_val_scaled, X_test_scaled, \
        X_train_cnn, X_val_cnn, X_test_cnn, scaler = data_pipeline(plot)

    # ---------------------------
    # TRAIN BASE MODELS (TRAIN ONLY)
    # ---------------------------
    exp, best_pycaret = create_and_compare_model_reg(train)

    nn_models = train_nn_models(
        X_train, X_val,
        y_train, y_val,
        X_train_scaled, X_val_scaled,
        X_train_cnn, X_val_cnn,
        plot.user_given_name
    )

    # keras models are still live - do not clear the session here
    free_memory(label="base model training")

    # ---------------------------
    # VALIDATION EVALUATION (NOT TEST!)
    # ---------------------------
    results_pycaret = evaluate_against_validation(
        exp,
        best_pycaret,
        val
    )

    results_nn = evaluate_against_validation_nn(
        nn_models,
        X_val_scaled,
        y_val
    )

    # ---------------------------
    # MODEL SELECTION (VALIDATION ONLY)
    # ---------------------------
    _, plot.use_pycaret = eval_approach_mix(
        results_pycaret,
        results_nn
    )

    # DEBUG: Force pycaret or nn usage for testing purposes
    # plot.use_pycaret = False

    # ---------------------------
    # TUNING + ENSEMBLE (NO TEST!)
    # ---------------------------
    if plot.use_pycaret:
        # NN lost the selection: drop all 5 keras models and the TF session state
        # before the memory-heavy tuning stage (safe - no keras model is used again)
        del nn_models
        free_memory(clear_keras=True, label="discarding NN branch")

        # only use best 3 models for tuning and ensemble creation
        best_pycaret = evaluate_results_and_choose_top_n(
            results_pycaret, best_pycaret, 3, pycaret_format=True)
        if not best_pycaret:
            raise ValueError("No successfully evaluated classical candidates")
        if state.Use_subprocess:
            plot.best_model = init_pycaret_subprocess_tuning_and_ensemble(
                plot.user_given_name,
                exp,
                best_pycaret,
                plot.ensemble
            )
        else:
            plot.best_model = tune_models(exp, best_pycaret)

            if plot.ensemble:
                plot.best_model = create_and_compare_ensemble(
                    plot.user_given_name,
                    exp,
                    plot.best_model
                )

    else:
        # Throw the worst architectures away before the expensive part: only the 3 best
        # NN models by validation R2 go into Hyperband tuning and ensemble creation.
        top_nn_models = evaluate_results_and_choose_top_n(
            results_nn, nn_models, 3, pycaret_format=False
        )

        # pycaret lost the selection: release its experiment (holds the dataset plus
        # every candidate pipeline) and the 2 losing keras models. NO clear_keras here -
        # the top 3 keras models must stay usable for tuning/ensembling.
        del exp, best_pycaret, nn_models
        free_memory(label="discarding pycaret branch")

        if state.Use_subprocess and plot.ensemble:
            plot.best_model = init_nn_subprocess_tuning_and_ensemble(
                plot.user_given_name,
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                top_nn_models
            )
        else:
            tuned_models = []
            tuned_hps = []

            for m in top_nn_models:
                tuned, hp = tune_model_nn(
                    X_train_scaled, y_train,
                    X_val_scaled, y_val,
                    m
                )
                tuned_models.append(tuned)
                tuned_hps.append(hp)

            if plot.ensemble:
                results_ensemble = compare_nn_ensembles(
                    tuned_models,
                    tuned_hps,
                    X_train_scaled,
                    y_train,
                    X_val_scaled,
                    y_val
                )
                plot.best_model = results_ensemble["best_predictor"]
            else:
                plot.best_model = tuned_models[0]

    # plot.best_model may be a live keras model/ensemble - no clear_keras
    free_memory(label="tuning + ensemble")

    # ---------------------------
    # FINAL TRAINING (TRAIN + VAL ONLY)
    # ---------------------------
    print("[INFO] Retraining best model on train + val...")

    if plot.use_pycaret:
        # Use the original `val` (not X_val/y_val) - X_val already had To_be_dropped columns
        # and the target stripped out by prepare_data_for_cnn2, so concatenating it with the
        # still-full `train` produced a column mismatch: every val-derived row ended up with
        # NaN in those 9 ignored columns (visible as "Rows with missing values: 20.0%" in the
        # setup summary). Harmless in practice since those columns are ignore_features anyway,
        # but `val` already has the same columns as `train` and avoids it outright.
        full_data = pd.concat([train, val], axis=0)

        plot.best_model, plot.best_exp = train_best(plot.best_model, full_data)

    else:
        X_full = np.concatenate([X_train_scaled, X_val_scaled])
        y_full = np.concatenate([y_train, y_val])

        # The tuned model already converged against a real validation set during tuning
        # (Hyperband + EarlyStopping). Continuing to fit it for a flat 50 unmonitored epochs
        # on train+val overfits it right past that optimum - validation R2 looks fine but the
        # final test evaluation tanks. Hold the trailing ~10% of train+val out of the refit
        # purely as an early-stopping monitor: chronological slice (no shuffle), so the
        # monitor stays a true "future" segment relative to the refit data.
        monitor_size = max(1, int(len(X_full) * 0.1))
        X_refit, X_monitor = X_full[:-monitor_size], X_full[-monitor_size:]
        y_refit, y_monitor = y_full[:-monitor_size], y_full[-monitor_size:]

        def refit_nn(m):
            m.fit(
                adapt_X_for_model(m, X_refit),
                y_refit,
                validation_data=(adapt_X_for_model(m, X_monitor), y_monitor),
                epochs=50,
                batch_size=32,
                callbacks=[EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True)],
                verbose=state.Verbose_logging
            )

        if isinstance(plot.best_model, EnsemblePredictor) and plot.best_model.method == "stacking":
            # Do NOT refit the fold base models in place here: meta_model (Ridge) was fit on
            # their out-of-fold predictions from CV on X_train. Refitting the base models on
            # train+val without regenerating those out-of-fold predictions leaves meta_model
            # calibrated against a distribution the base models no longer produce (in-sample-ish,
            # overconfident predictions instead of honest out-of-fold ones) - this silently wrecks
            # accuracy rather than improving it. Properly redoing this would mean rerunning the
            # K-fold OOF + meta_model fit from compare_nn_ensembles on the combined data, which
            # isn't exposed here. Leave the already-validly-trained stacking ensemble as-is instead.
            print(
                "[INFO] Skipping train+val refit for stacking ensemble (would decalibrate meta_model).")

        elif isinstance(plot.best_model, EnsemblePredictor):
            # average/bagging have no meta-model calibration to invalidate - refitting each
            # base model in place is safe. predict() reads self.base_models at call time, so
            # this "retrains the ensemble" without needing to rebuild the wrapper.
            for m in plot.best_model.base_models:
                refit_nn(m)
        else:
            refit_nn(plot.best_model)

    # plot.best_model may be a live keras model/ensemble - no clear_keras
    free_memory(label="final training")

    # ---------------------------
    # FINAL TEST EVALUATION (ONLY ONCE!)
    # ---------------------------
    print("[INFO] Final evaluation on TEST set...")

    if plot.use_pycaret:
        final_eval, _ = evaluate_against_testset(
            plot,
            test,
            plot.best_exp,
            plot.best_model
        )
    else:
        final_eval, _ = evaluate_against_testset_nn(
            plot,
            plot.best_model,
            X_test_scaled,
            y_test
        )

    # Persist the exact final model used for inference. Publishing the manifest
    # retains the previous generation as an explicit rollback target.
    # Every promotion gets immutable filenames so the manifest's previous
    # generation remains usable after a later training run.
    generation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final_prefix = (
        f"models/{plot.user_given_name}/generations/{generation_id}/"
    )
    if plot.use_pycaret:
        model_artifacts = save_models(
            plot.user_given_name,
            plot.best_exp,
            plot.best_model,
            final_prefix + "soil_tension_prediction_",
        )
    else:
        model_artifacts = save_models_nn(
            plot.user_given_name,
            plot.best_model,
            final_prefix + "soil_tension_prediction_",
        )
    if not model_artifacts:
        raise RuntimeError("Final model could not be persisted safely")
    plot.model_artifacts = model_artifacts

    # ---------------------------
    # FUTURE PREDICTIONS
    # ---------------------------
    future_features = create_future_values(plot.data, plot)
    future_features = compare_train_predictions_cols(train, future_features)

    if not plot.use_pycaret:
        Z, Z_scaled, Z_cnn = prepare_future_values(
            scaler, future_features, X_train.columns
        )

    if plot.use_pycaret:
        preds = generate_predictions(
            plot.best_model,
            plot.best_exp,
            future_features.reset_index(drop=True)
        )
        # generate_predictions doesn't reorder/filter rows, so the original datetime index
        # (dropped above so predict_model() gets a plain-indexed frame) maps back 1:1 by position.
        preds.index = future_features.index
    else:
        preds = generate_predictions_nn(
            plot.best_model,
            Z_scaled,
            future_features.index[0],
            future_features.index[-1],
            interval_minutes=int(
                getattr(plot, "forecast_interval_minutes", 60)),
        )

    plot.predictions = preds

    # ---------------------------
    # FINAL CLEANUP
    # ---------------------------

    # Ensure the index of plot.predictions is datetime with the same timezone
    if plot.predictions.index.tz is None:
        plot.predictions.index = pd.to_datetime(
            plot.predictions.index).tz_localize(timezone_name)
    else:
        plot.predictions.index = plot.predictions.index.tz_convert(
            timezone_name)

    # Create a Timestamp from the current date and time (without microseconds, seconds, and minutes)
    current_time = pd.Timestamp.now(tz=timezone_name).floor('H')

    # Now, slice the predictions DataFrame based on the timestamp
    # Cut passed time from predictions
    if not plot.load_data_from_csv:
        plot.predictions = plot.predictions.loc[current_time:]

    # Align predictions with historical data -> TODO: dodgy fix, only trigger in case of bad performance? DEBUG
    align_with_latest_sensor_values(plot)
    # plot.predictions['smoothed_values'] = plot.predictions['prediction_label']

    # Calculate when threshold will be meet
    plot.threshold_timestamp = calc_threshold(
        plot.predictions, 'smoothed_values', plot)

    # Add volumetric water content
    if plot.sensor_kind == 'tension':
        plot.predictions = add_volumetric_col_to_df(
            plot.predictions, "smoothed_values", plot)

    return (
        plot.data['rolling_mean_grouped_soil'][-1],
        plot.threshold_timestamp,
        plot.predictions
    )


def main(plot) -> int:
    """Train and forecast while holding the process-wide model resource lock."""
    with state.model_operation():
        return _train_model_cycle_unlocked(plot)


def _build_tension_forecast(
    predictions: pd.DataFrame,
    current_timestamp=None,
) -> dict:
    """Convert timestamped model output into decision-engine horizon labels."""
    from decision_engine import build_timestamp_forecast

    return build_timestamp_forecast(predictions, current_timestamp)


def _build_forecast_metadata(
    predictions: pd.DataFrame,
    current_timestamp=None,
) -> tuple:
    """Return forecast values plus explicit timestamps, cadence, and horizon."""
    forecast = _build_tension_forecast(predictions, current_timestamp)
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return forecast, [], None, None

    index = pd.DatetimeIndex(predictions.index).sort_values().unique()
    now = pd.Timestamp(current_timestamp) if current_timestamp is not None else (
        pd.Timestamp.now(tz=index.tz)
        if index.tz is not None else pd.Timestamp.now())
    if index.tz is None and now.tzinfo is not None:
        now = now.tz_localize(None)
    elif index.tz is not None and now.tzinfo is None:
        now = now.tz_localize(index.tz)
    elif index.tz is not None and now.tzinfo is not None:
        now = now.tz_convert(index.tz)
    future = index[index > now]
    timestamps = [timestamp.isoformat() for timestamp in future]
    if len(future) == 0:
        return forecast, timestamps, None, None

    horizon_hours = (future[-1] - now).total_seconds() / 3600.0
    interval_hours = None
    if len(future) > 1:
        deltas = pd.Series(future[1:] - future[:-1])
        interval_hours = round(
            float(deltas.dt.total_seconds().median() / 3600.0), 3)
    return (
        forecast,
        timestamps,
        interval_hours,
        round(max(0.0, horizon_hours), 3),
    )


def _runtime_metadata(runtime_state, plot=None, current_tension=None) -> tuple:
    """Flatten runtime-state freshness and source details for the API contract."""
    freshness = {
        "weather_age_hours": getattr(runtime_state, "weather_data_age_hours", None),
        "satellite_age_hours": getattr(runtime_state, "sat_data_age_hours", None),
        "satellite_ndvi_age_hours": getattr(runtime_state, "sat_ndvi_age_hours", None),
    }
    if plot is not None:
        sensor_quality = evaluate_tension_sensor_safety(
            plot, current_value=current_tension)
        freshness["soil_tension_age_hours"] = sensor_quality.get(
            "latest_age_hours")
        freshness["soil_tension_safe_for_automatic"] = sensor_quality.get(
            "safe_for_automatic", False)
        freshness["soil_tension_quality"] = sensor_quality
    sources = {
        "soil_tension": "plot sensor/model",
        "weather": getattr(runtime_state, "weather_provider", "unknown"),
        "satellite": getattr(runtime_state, "satellite_provider", "unavailable"),
        "crop_state": (
            "configured"
            if runtime_state is not None
            else getattr(plot, "runtime_crop_state_error", "unavailable")
        ),
    }
    return freshness, sources


def run_complete_cycle(plot, training: bool = False) -> PipelineCycleResult:
    """Run model inference, crop-state computation, and decision evaluation."""
    import actuation
    from crop_model import build_forecast_thresholds
    from decision_engine import (
        error_recommendation,
        evaluate_forecast,
        inactive_season_recommendation,
    )

    previous_result = getattr(plot, "pipeline_result", None)
    if isinstance(previous_result, dict):
        previous_calculated_at = previous_result.get("calculated_at")
    else:
        previous_calculated_at = getattr(
            previous_result, "calculated_at", None)

    # Resolve the current crop state first; forecast thresholds are then
    # projected independently at every model horizon.
    runtime_state = actuation.get_runtime_crop_state(plot)
    if runtime_state is not None:
        plot.threshold = float(runtime_state.stress_threshold_cbar)

    try:
        model_result = main(
            plot) if training else predict_with_updated_data(plot)
        current_tension, threshold_timestamp, predictions = model_result
        decision_now = pd.Timestamp.now(tz="UTC")
        (
            forecast,
            forecast_timestamps,
            forecast_interval_hours,
            forecast_horizon_hours,
        ) = _build_forecast_metadata(predictions, decision_now)
        data_freshness, data_sources = _runtime_metadata(
            runtime_state, plot, current_tension)
        threshold = (
            float(runtime_state.stress_threshold_cbar)
            if runtime_state is not None
            else float(getattr(plot, "threshold", 0.0))
        )
        threshold_forecast = {}
        threshold_timestamps = {}
        threshold_forecast_error = None
        if runtime_state is not None and getattr(
                runtime_state, "season_active", True):
            try:
                forecast_farm = actuation._build_runtime_farm_config(plot)
                forecast_farm.planting_date = getattr(
                    runtime_state, "phenology_reference_date", None
                ) or getattr(plot, "planting_date", "")
                threshold_forecast = build_forecast_thresholds(
                    forecast_farm,
                    runtime_state.gdd_cumulative,
                    forecast,
                    getattr(runtime_state, "weather_forecast_frame", None),
                    decision_now,
                )
                ordered_labels = sorted(
                    threshold_forecast,
                    key=lambda label: float(str(label).lower().removesuffix("h")),
                )
                available_timestamps = [
                    pd.Timestamp(value) for value in forecast_timestamps]
                threshold_timestamps = {}
                for label in ordered_labels:
                    target = decision_now + pd.Timedelta(hours=float(
                        str(label).lower().removesuffix("h")))
                    if available_timestamps:
                        distances = []
                        for value in available_timestamps:
                            comparable_target = target
                            if value.tzinfo is None:
                                comparable_target = target.tz_localize(None)
                            elif comparable_target.tzinfo is None:
                                comparable_target = comparable_target.tz_localize(
                                    value.tzinfo)
                            else:
                                comparable_target = comparable_target.tz_convert(
                                    value.tzinfo)
                            distances.append(abs(
                                (value - comparable_target).total_seconds()))
                        timestamp = available_timestamps[int(np.argmin(distances))]
                        threshold_timestamps[timestamp.isoformat()] = (
                            threshold_forecast[label])
            except (TypeError, ValueError) as exc:
                threshold_forecast_error = str(exc)
        sensor_quality = data_freshness.get("soil_tension_quality", {})
        if runtime_state is None:
            recommendation = error_recommendation(
                current_tension=float(current_tension),
                tension_forecast=forecast,
                stress_threshold=threshold,
                message=(
                    "The tension forecast is available, but crop-specific "
                    "irrigation advice is disabled because crop state or "
                    "threshold calibration is unavailable: "
                    + str(getattr(
                        plot, "runtime_crop_state_error", "unavailable"))
                ),
            )
        elif not getattr(runtime_state, "season_active", True):
            recommendation = inactive_season_recommendation(
                current_tension=float(current_tension),
                tension_forecast=forecast,
                stress_threshold=threshold,
                message=(
                    "Irrigation recommendations are disabled because the crop "
                    "season is inactive: "
                    + str(getattr(
                        runtime_state, "season_end_reason", "outside_season"))
                ),
            )
        elif threshold_forecast_error is not None:
            recommendation = error_recommendation(
                current_tension=float(current_tension),
                tension_forecast=forecast,
                stress_threshold=threshold,
                message=(
                    "Dynamic forecast thresholds are unavailable: "
                    + threshold_forecast_error
                ),
            )
        elif sensor_quality.get("safe_for_advisory", False):
            # Decision horizons come from the model timestamps built above.
            recommendation = evaluate_forecast(
                current_tension=float(current_tension),
                tension_forecast=forecast,
                stress_threshold=threshold,
                stress_thresholds=threshold_forecast,
                current_timestamp=decision_now.to_pydatetime(),
                advise_horizon_hours=float(
                    getattr(plot, "look_ahead_time", 24) or 24),
                watch_horizon_hours=float(
                    getattr(plot, "watch_horizon_time", 72) or 72),
            )
        else:
            reasons = sensor_quality.get("advisory_reasons") or [
                "soil_tension_evidence_unavailable"]
            recommendation = error_recommendation(
                current_tension=float(current_tension),
                tension_forecast=forecast,
                stress_threshold=threshold,
                message=(
                    "No irrigation alert was evaluated because current raw "
                    "soil-tension evidence is unsafe for advisory use: "
                    + ", ".join(reasons)
                ),
            )
        threshold_timestamp = (
            recommendation.first_breach_timestamp
            if recommendation.first_breach_timestamp is not None else False
        )
        plot.threshold_timestamp = threshold_timestamp
        result = PipelineCycleResult(
            current_tension=float(current_tension),
            threshold_timestamp=threshold_timestamp,
            predictions=predictions,
            tension_forecast=forecast,
            forecast_timestamps=forecast_timestamps,
            forecast_interval_hours=forecast_interval_hours,
            forecast_horizon_hours=forecast_horizon_hours,
            stress_threshold_cbar=threshold,
            stress_threshold_forecast=threshold_forecast,
            stress_threshold_timestamps=threshold_timestamps,
            crop_state=runtime_state,
            recommendation=recommendation,
            data_freshness=data_freshness,
            data_sources=data_sources,
            model_status="trained" if training else "prediction_only",
            inference_source=getattr(plot, "_inference_source", "live"),
            calculated_at=decision_now.isoformat(),
            previous_calculated_at=previous_calculated_at,
        )
    except Exception as exc:
        # Keep a structured failure available to the API before retrying.
        result = PipelineCycleResult(
            current_tension=float(getattr(plot, "threshold", 0.0)),
            threshold_timestamp=getattr(plot, "threshold_timestamp", ""),
            predictions=getattr(plot, "predictions", pd.DataFrame()),
            tension_forecast={},
            forecast_timestamps=[],
            forecast_interval_hours=None,
            forecast_horizon_hours=None,
            stress_threshold_cbar=(
                float(runtime_state.stress_threshold_cbar)
                if runtime_state is not None else None
            ),
            stress_threshold_forecast={},
            stress_threshold_timestamps={},
            crop_state=runtime_state,
            recommendation=None,
            data_freshness={},
            data_sources={},
            model_status="failed",
            inference_source=getattr(plot, "_inference_source", "error"),
            # A failed cycle does not advance the water-calculation checkpoint.
            calculated_at=previous_calculated_at,
            previous_calculated_at=(
                previous_result.get("previous_calculated_at")
                if isinstance(previous_result, dict)
                else getattr(previous_result, "previous_calculated_at", None)
            ),
            fallback_used=False,
            fallback_reason=None,
            error=str(exc),
        )
        plot.pipeline_result = result
        raise

    plot.pipeline_result = result
    return result


def run_training_cycle(plot):
    """Run the complete model training and prediction cycle for one plot."""
    result = run_complete_cycle(plot, training=True)
    return result.current_tension, result.threshold_timestamp, result.predictions


def run_prediction_cycle(plot):
    """Run the prediction-only cycle using the plot's loaded model state."""
    result = run_complete_cycle(plot, training=False)
    return result.current_tension, result.threshold_timestamp, result.predictions
