"""Top-level pipelines: training run (main) and periodic re-prediction.

Split out of the original create_model.py - function bodies are verbatim
(only module-flag references now go through create_model.state).
"""
import csv
import ctypes
from datetime import timedelta, datetime
import gc
import json
import logging
import os
import pickle
import shutil
from dateutil import parser
import subprocess
import joblib
import pycaret
from pycaret.regression import *
from pycaret.internal.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pytz
import traceback
import psutil
from pathlib import Path

import tensorflow
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Input, Reshape
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, get as get_optimizer
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.backend import floatx
import tensorflow.keras.models as keras_models
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from keras_tuner import Hyperband, HyperParameters
import time

from utils import NetworkUtils, TimeUtils
from tune_grids import PYCARET_REGRESSION_TUNE_GRIDS
from subprocess_manager import run_tuning_and_ensemble_nn_with_subprocess, run_tuning_and_ensemble_with_subprocess, run_tuning_with_subprocess, run_ensemble_with_subprocess

from . import state
from .constants import *
from .evaluation import eval_approach_mix, evaluate_against_testset, evaluate_against_testset_nn, evaluate_against_validation, evaluate_against_validation_nn, evaluate_results_and_choose_top_n
from .features import prepare_data, split_by_ratio
from .nn_architectures import adapt_X_for_model
from .nn_ensemble import EnsemblePredictor, compare_nn_ensembles
from .nn_training import init_nn_subprocess_tuning_and_ensemble, prepare_data_for_cnn2, prepare_future_values, train_nn_models, tune_model_nn
from .prediction import align_with_latest_sensor_values, calc_threshold, compare_train_predictions_cols, create_future_values, generate_predictions, generate_predictions_nn
from .pycaret_models import create_and_compare_ensemble, create_and_compare_model_reg, init_pycaret_subprocess_tuning_and_ensemble, train_best, tune_models
from .runtime import free_memory
from .soil import add_volumetric_col_to_df


def predict_with_updated_data(plot):
# (flag now lives in state module)

    # Prevents multiple training or prediction at the same time
    while state.Currently_active:
        print(f"[{plot.user_given_name}] Waiting for resources to be released. Another training or prediction is already running.")
        time.sleep(Resource_wait_time_seconds)
    # Before training starts, lock the resource    
    state.Currently_active = True
    
    # Run data pipeline to obtain latest data. training=False: reuse the scaler the frozen
    # model was trained under instead of refitting it on the newly accumulated data.
    train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler = data_pipeline(plot, training=False)
    # Create future value set to feed new data to model
    future_features = create_future_values(plot.data, plot)
    # Compare dataframes cols to be sure that they match, otherwise drop
    future_features = compare_train_predictions_cols(train, future_features)
    # NN
    if not plot.use_pycaret:
        Z, Z_scaled, Z_cnn = prepare_future_values(scaler, future_features, X_train.columns)
        plot.predictions = generate_predictions_nn(plot.best_model, Z_scaled, future_features.index[0], future_features.index[-1])
    else:
        plot.predictions = generate_predictions(plot.best_model, plot.best_exp, future_features)
    
    # Cut passed time from predictions
    if not plot.load_data_from_csv:
        plot.predictions = plot.predictions.loc[pd.Timestamp((datetime.now()).replace(microsecond=0, second=0, minute=0)).tz_localize(TimeUtils.Timezone):]
        
    # Align predictions with historical data
    align_with_latest_sensor_values(plot)
    
    # Calculate when threshold will be meet
    plot.threshold_timestamp = calc_threshold(plot.predictions, 'smoothed_values', plot)

    # Add volumetric water content
    if plot.sensor_kind == 'tension':
        plot.predictions = add_volumetric_col_to_df(plot.predictions, "smoothed_values", plot)

    # Runs every 3h on the RPi: return the freed frames to the OS before idling.
    # No clear_keras - plot.best_model must stay usable for the next cycle.
    free_memory(label="prediction run")

    # After finished job set active to false
    state.Currently_active = False

    # Return last accumulated reading and threshold timestamp currentSoilTension, threshold_timestamp, predictions
    return plot.data['rolling_mean_grouped_soil'][-1], plot.threshold_timestamp, plot.predictions


def data_pipeline(plot, training=True):
    # Data preparation pipeline, calls other subfunction to perform the task
    # Classical regression
    plot.data = prepare_data(plot)

    # Search for gaps in data again (quick fix) => tackle problem with latest data "nan", in case of irrigations saved
    if plot.data.isna().any().any():
        #Data.drop(Data.index[-1], inplace=True)
        plot.data.dropna(inplace=True)

    # Split dataset
    train, val, test = split_by_ratio(plot.data) # here a split is done to rule out the models that are overfitting

    # NN
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler = prepare_data_for_cnn2(plot, train, val, test, 'rolling_mean_grouped_soil', training=training)

    return train, val, test, X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled, X_train_cnn, X_val_cnn, X_test_cnn, scaler 


def main(plot) -> int:
# (flag now lives in state module)

    while state.Currently_active:
        print(f"[{plot.user_given_name}] Waiting for resources...")
        time.sleep(Resource_wait_time_seconds)

    state.Currently_active = True

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
    index, plot.use_pycaret = eval_approach_mix(
        results_pycaret,
        results_nn
    )

    # DEBUG: Force pycaret or nn usage for testing purposes
    #plot.use_pycaret = False

    # ---------------------------
    # TUNING + ENSEMBLE (NO TEST!)
    # ---------------------------
    if plot.use_pycaret:
        # NN lost the selection: drop all 5 keras models and the TF session state
        # before the memory-heavy tuning stage (safe - no keras model is used again)
        del nn_models
        free_memory(clear_keras=True, label="discarding NN branch")

        # only use best 3 models for tuning and ensemble creation
        best_pycaret = best_pycaret[:3]
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
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
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
            print("[INFO] Skipping train+val refit for stacking ensemble (would decalibrate meta_model).")

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
            future_features.index[-1]
        )

    plot.predictions = preds

    # ---------------------------
    # FINAL CLEANUP
    # ---------------------------

    # Ensure the index of plot.predictions is datetime with the same timezone
    if plot.predictions.index.tz is None:
        #plot.predictions.index = pd.to_datetime(plot.predictions.index).tz_localize('UTC').tz_convert(TimeUtils.Timezone)
        plot.predictions.index = pd.to_datetime(plot.predictions.index).tz_localize(TimeUtils.Timezone)
    else:
        plot.predictions.index = plot.predictions.index.tz_convert(TimeUtils.Timezone)

    # Create a Timestamp from the current date and time (without microseconds, seconds, and minutes)
    current_time = pd.Timestamp.now(tz=TimeUtils.Timezone).floor('H')

    # Now, slice the predictions DataFrame based on the timestamp
    # Cut passed time from predictions
    if not plot.load_data_from_csv:
        plot.predictions = plot.predictions.loc[current_time:]

    # Align predictions with historical data -> TODO: dodgy fix, only trigger in case of bad performance? DEBUG
    align_with_latest_sensor_values(plot)
    #plot.predictions['smoothed_values'] = plot.predictions['prediction_label']

    # Calculate when threshold will be meet
    plot.threshold_timestamp = calc_threshold(plot.predictions, 'smoothed_values', plot)

    # Add volumetric water content
    if plot.sensor_kind == 'tension':
        plot.predictions = add_volumetric_col_to_df(plot.predictions, "smoothed_values", plot)

    # After finished job set active to false
    state.Currently_active = False

    return (
        plot.data['rolling_mean_grouped_soil'][-1],
        plot.threshold_timestamp,
        plot.predictions
    )
