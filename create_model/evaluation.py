"""Metric computation, model ranking and approach selection.

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
from .nn_architectures import adapt_X_for_model, prepare_lstm_data


# evaluate performance of prediction against test part of data
def evaluate_target_variable(series1, series2, model_name):
    # Callers must supply the same ordered observations. Never truncate or
    # independently drop missing rows: that compares unrelated measurements.
    if (not series1.index.equals(series2.index)
            or not series1.index.is_unique or not series2.index.is_unique):
        raise ValueError("Evaluation requires matching unique observation indexes")
    paired = series1.notna() & series2.notna()
    values1 = pd.to_numeric(series1.loc[paired], errors='raise').astype(float)
    values2 = pd.to_numeric(series2.loc[paired], errors='raise').astype(float)
    if (len(values1) < 2 or not np.isfinite(values1.to_numpy()).all()
            or not np.isfinite(values2.to_numpy()).all()):
        raise ValueError("Evaluation requires at least two finite observation pairs")

    # test print
    #print(values1)
    #print(values2)
    
    # MAE, RMSE, MPE
    diff = np.abs(values1.values - values2.values)
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    # MPE (Mean Percentage Error) - Avoid division by zero
    non_zero_mask = values1.values != 0
    mpe = np.mean(diff[non_zero_mask] / values1.values[non_zero_mask]) * 100 if np.any(non_zero_mask) else np.nan


    # Use the same finite constant-target policy as get_r2_manual and ensemble
    # scoring: perfect constant predictions score 1, imperfect ones score 0.
    r2 = r2_score(values1.to_numpy(), values2.to_numpy(), force_finite=True)

    # print metrics
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MPE: {mpe:.2f} %")
    print(f"R2 {r2:.2f}",'\n')
    #print("df1 len:",len(values1),"df2 len:",len(values2),'\n')

    metrics = { model_name : ['mae', 'rmse', 'mpe', 'r2'],
                # Only presentation is rounded; selection needs full precision.
                'results'  : [mae, rmse, mpe, r2]
              }
    results = pd.DataFrame(metrics)
    results['results'] = results['results'].astype(float)

    return results


def _failed_candidate_result(name):
    # Keep one result slot per original candidate, even when inference fails.
    return pd.DataFrame({name: ['mae', 'rmse', 'mpe', 'r2'],
                         'results': [np.nan] * 4})


def _rankable_candidates(results, models):
    if len(results) != len(models):
        raise ValueError("Candidate scores must match the original candidate list")
    return [(float(result['results'].iloc[3]), index)
            for index, result in enumerate(results)
            if np.isfinite(result['results'].iloc[3])]


# Sort neural network models in new dataframe according to performance on testset 
def evaluate_results_and_choose_best(results_for_one_df, best_for_one_df, pycaret_format=True):
    # sort according to R2 score -> hence [3]
    eligible = _rankable_candidates(results_for_one_df, best_for_one_df)
    if not eligible:
        raise ValueError("No successfully evaluated candidates")
    max_r2_value = max(eligible)
    
    max_value = max_r2_value[0]
    max_index = max_r2_value[1]

    best_model_for_df = best_for_one_df[max_index]
    
    print("The best model after evaluation is:", best_model_for_df.__module__ if pycaret_format else best_model_for_df.model_name)
    print("Maximum R2 Value:", max_value)
    print('This is the rest of the metrics: mae', results_for_one_df[max_index]['results'][0],  'rmse', results_for_one_df[max_index]['results'][1], 'mpe', results_for_one_df[max_index]['results'][2])
    print("Index of Maximum R2 Value:", max_index,"\n")

    return best_model_for_df


# for ensemble/stacking model return the 3 best models based on R2 score
def evaluate_results_and_choose_top_n(results_for_one_df, best_for_one_df, top_n=3, pycaret_format=True):
    """
    results_for_one_df: list of dicts, each with key 'results' -> sequence [MAE, RMSE, MPE, R2, ...]
    best_for_one_df: list of fitted models corresponding 1:1 to results_for_one_df
    top_n: how many top models to return (default 3)
    """
    # collect (R2, index) pairs
    r2_with_index = _rankable_candidates(results_for_one_df, best_for_one_df)

    # sort by R2 descending
    r2_with_index.sort(key=lambda x: x[0], reverse=True)

    # take top_n (or fewer if not enough models)
    top_n = min(top_n, len(r2_with_index))
    top_indices = r2_with_index[:top_n]

    best_models = []
    print(f"Top {top_n} models by R2:\n")

    for rank, (r2_value, idx) in enumerate(top_indices, start=1):
        model = best_for_one_df[idx]
        res = results_for_one_df[idx]["results"]

        if state.Verbose_logging:
            print(f"Rank {rank}:")
            print("  Model:", model.__module__ if pycaret_format else model.model_name)
            print("  R2:  ", r2_value)
            print("  MAE: ", res[0])
            print("  RMSE:", res[1])
            print("  MPE: ", res[2])
            print("  Index in lists:", idx, "\n")

        best_models.append(model)

    return best_models


# Evaluate PyCaret models on validation set
def evaluate_against_validation(
    exp,
    models,
    val_df
):
    print("[VALIDATION] Evaluating PyCaret models")

    # ---------------------------------------
    # COPY VALIDATION DATA
    # ---------------------------------------
    val_df = val_df.copy().reset_index(drop=True)

    # ---------------------------------------
    # REMOVE UNUSED FEATURES
    # ---------------------------------------
    to_be_dropped = [
        item for item in To_be_dropped
        if item != 'Timestamp'
    ]

    val_df = val_df.drop(
        columns=to_be_dropped,
        errors='ignore'
    )

    # ---------------------------------------
    # SPLIT TARGET / FEATURES
    # ---------------------------------------
    ground_truth = val_df['rolling_mean_grouped_soil']

    val_features = val_df.drop(
        ['rolling_mean_grouped_soil'],
        axis=1
    )

    # ---------------------------------------
    # EVALUATE MODELS
    # ---------------------------------------
    results_for_model = []

    for model in models:

        model_name = model.__module__

        print(f"[VALIDATION] Model: {model_name}")

        try:

            preds = exp.predict_model(
                model,
                data=val_features
            )

            y_true = ground_truth.reset_index(drop=True)

            y_pred = preds[
                'prediction_label'
            ].reset_index(drop=True)

            result = evaluate_target_variable(
                y_true,
                y_pred,
                model_name
            )

            results_for_model.append(result)

        except Exception as e:

            results_for_model.append(_failed_candidate_result(model_name))
            print(
                f"[VALIDATION] Error in "
                f"{model_name}: {e}"
            )

    return results_for_model


# eval model against formerly split validation set for NN models (custom evaluation)
def evaluate_against_validation_nn(
    nn_models,
    X_val_scaled,
    y_val
):
    print("[VALIDATION] Evaluating NN models")

    results_for_model = []

    for model in nn_models:
        try:
            if model.model_name == "lstm_model":
                X_val_lstm = prepare_lstm_data(X_val_scaled)
                pred = model.predict(X_val_lstm)
            else:
                pred = model.predict(adapt_X_for_model(model, X_val_scaled))

            pred = pred.flatten()

            results_for_model.append(
                evaluate_target_variable(
                    y_val.reset_index(drop=True),
                    pd.Series(pred),
                    model.model_name
                )
            )

        except Exception as e:
            results_for_model.append(_failed_candidate_result(model.model_name))
            print(f"[VALIDATION] Error in {model.model_name}: {e}")

    return results_for_model


# eval model against formerly split testset -> TODO: test has duplicates because of irrigation added formerly
def evaluate_against_testset(currentPlot, test, exp, best):
    print("This is the evaluation against the split testset")
    ground_truth = test['rolling_mean_grouped_soil']
    #ground_truth.reset_index(drop=True, inplace=True)
    test_features = test.drop(['rolling_mean_grouped_soil'], axis=1)
    # Create a new list without 'Timestamp', .remove() is inplace and does not return
    to_be_dropped = [item for item in To_be_dropped if item != 'Timestamp']
    #test_features.reset_index(drop=True, inplace=True)
    test_features = test_features.drop(to_be_dropped, axis=1)

    predictions = []
    results_for_model = []

    if not isinstance(best, list):
        best = [best]

    # iterate best models
    for i in range(len(best)):
        model_name = best[i].__module__
        print("Current model: " + model_name)

        # Create predictions
        try:
            prediction = exp.predict_model(best[i], data=test_features)
            result = evaluate_target_variable(ground_truth, prediction['prediction_label'], model_name)
        except Exception as exc:
            print(f'[TEST] Error in {model_name}: {exc}')
            result = _failed_candidate_result(model_name)
        results_for_model.append(result)

    if not _rankable_candidates(results_for_model, best):
        raise ValueError('No successfully evaluated held-out candidates')
    
    if currentPlot.ensemble == True:
        # For ensemble/stacking return top 3 models
        best_eval = evaluate_results_and_choose_top_n(results_for_model, best, 3, pycaret_format=True)
    else:
        # Sort models in new dataframe according to performance on testset
        best_eval = evaluate_results_and_choose_best(results_for_model, best, pycaret_format=True)
    
    return best_eval, results_for_model


# Perform a evaluation of the models against the testset(X_test), slit before 
def evaluate_against_testset_nn(currentPlot, nn_models, X_test_scaled, y_test):
    results_for_model = []

    if not isinstance(nn_models, list):
        nn_models = [nn_models]
        
    for i in range(len(nn_models)):
    #     # Evaluate the model on the test set
    #     try:
    #         loss = nn_models[i].evaluate(X_test_scaled, y_test.to_numpy()[...,np.newaxis])
    #         print(f'Model: {i}  Test Loss: {loss}')
    #     except Exception as e:
    #         print(f"Evaluate is not available for the model. {e}")
        # Make predictions
        try:
            if nn_models[i].model_name == "lstm_model":
                # Prepare data for LSTM input explicitly
                X_test_lstm = prepare_lstm_data(X_test_scaled)
                prediction = nn_models[i].predict(X_test_lstm)
            else:
                prediction = nn_models[i].predict(adapt_X_for_model(nn_models[i], X_test_scaled))
            result = evaluate_target_variable(
                y_test.reset_index(drop=True), pd.Series(prediction.flatten()), nn_models[i].model_name)
        except Exception as e:
            print(f"There was an error in predict() for the model {nn_models[i].model_name}.\n Error: {e}")
            result = _failed_candidate_result(nn_models[i].model_name)

        # evaluate predictions against testset 
        results_for_model.append(result)

    if not _rankable_candidates(results_for_model, nn_models):
        raise ValueError('No successfully evaluated held-out candidates')
    
    if currentPlot.ensemble == True:
        # For ensemble/stacking return top 3 models
        best_eval = evaluate_results_and_choose_top_n(results_for_model, nn_models, 3, pycaret_format=False)
    else:
        # Sort models in new dataframe according to performance on testset
        best_eval = evaluate_results_and_choose_best(results_for_model, nn_models, pycaret_format=False)

    return best_eval, results_for_model


# Decide which approach is the winning approach based on a weighted mixture of normalized MAE, MPE, and R2.
def eval_approach_mix(results_pycaret, results_nn, weights=None, top_k=3):
    """
    Compare the pycaret and NN approaches using a weighted mixture of normalized
    MAE, |MPE| and R2 (scaled across ALL models so each metric has comparable influence).

    The approaches are compared on the MEAN of their top_k combined scores, not on
    their single best model: the winning side proceeds as a *group* into tuning and
    ensembling (top 3), and a single lucky standout in the much larger pycaret pool
    would otherwise win disproportionately often (a max over ~19 noisy candidates
    beats a max over ~3 even when both families are equally good).

    Normalization is winsorized at the 5th/95th percentile per metric, so one broken
    candidate (e.g. an R2 of -9999) cannot stretch the scale and wash out the real
    differences between the healthy models. Candidates must have finite enabled
    metrics; zero-weight metrics do not affect eligibility or scores.

    Returns (best_model_index_within_winning_group, use_pycaret). Ties go to pycaret.
    """

    if weights is None:
        weights = {"mae": 1/3, "mpe": 1/3, "r2": 1/3}
    try:
        weights = {name: float(weights[name]) for name in ('mae', 'mpe', 'r2')}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Metric weights must specify finite nonnegative mae, mpe and r2 values') from exc
    if (not all(np.isfinite(value) and value >= 0 for value in weights.values())
            or not any(value > 0 for value in weights.values())):
        raise ValueError('Metric weights must be finite and nonnegative, with at least one positive weight')

    # Merge all results temporarily for normalization
    all_results = results_pycaret + results_nn
    if not all_results:
        raise ValueError("eval_approach_mix: both result lists are empty")

    # Extract metrics into arrays; negate lower-is-better ones so higher is always better
    active_metrics = {}
    for name, index in (('mae', 0), ('mpe', 2), ('r2', 3)):
        if weights[name] == 0:
            continue  # A disabled metric must not affect eligibility or scores.
        values = np.array([r['results'][index] for r in all_results], dtype=float)
        active_metrics[name] = values if name == 'r2' else (-np.abs(values) if name == 'mpe' else -values)
    eligible = np.logical_and.reduce([np.isfinite(values) for values in active_metrics.values()])
    if not eligible.any():
        raise ValueError("No successfully evaluated candidates")

    # Clip each metric to its 5th..95th percentile before scaling: a single broken
    # model must lose, but must not distort how all the others compare to each other
    def winsorize(x):
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return x
        lo, hi = np.percentile(finite, 5), np.percentile(finite, 95)
        return np.clip(x, lo, hi)

    # Min-max normalize each metric to [0..1] (NaN-tolerant)
    def minmax(x):
        mn, mx = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(mx - mn) or mx == mn:
            return np.full_like(x, 0.5, dtype=float)
        return (x - mn) / (mx - mn)

    # Normalize only eligible candidates and enabled metrics. In particular,
    # multiplying a NaN metric by zero would still contaminate the score.
    combined_scores = np.zeros(len(all_results), dtype=float)
    for name, values in active_metrics.items():
        combined_scores[eligible] += weights[name] * minmax(winsorize(values[eligible]))
    combined_scores[~eligible] = -np.inf

    # Split back to pycaret and nn ranges
    n_pycaret = len(results_pycaret)

    scores_pycaret = combined_scores[:n_pycaret]
    scores_nn      = combined_scores[n_pycaret:]

    # Group strength: mean of the top_k scores (an empty side loses outright)
    def group_strength(scores):
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return -np.inf
        k = min(top_k, len(scores))
        return float(np.sort(scores)[-k:].mean())

    strength_pycaret = group_strength(scores_pycaret)
    strength_nn      = group_strength(scores_nn)

    if state.Verbose_logging:
        print(f"[SELECTION] pycaret: top-{top_k} mean={strength_pycaret:.3f} (n={n_pycaret}) | "
              f"nn: top-{top_k} mean={strength_nn:.3f} (n={len(scores_nn)})")

    # Compare approaches
    if strength_pycaret >= strength_nn:
        return int(np.argmax(scores_pycaret)), True      # use pycaret
    else:
        return int(np.argmax(scores_nn)), False          # use NN


def get_r2_manual(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)
    except Exception as e:
        print(f"Warning: failed to compute R2 for {model}: {e}")
        return -9999
