"""NN ensemble predictor wrapper and ensemble technique comparison.

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
from .evaluation import get_r2_manual
from .nn_architectures import adapt_X_for_model, fresh_optimizer_from_model_or_hp, hp_get


# Predictor object for ensemble models
class EnsemblePredictor:
    def __init__(self, base_models, meta_model=None, method="stacking", name=None):
        self.base_models = base_models   # list[list[model]] for stacking
        self.meta_model = meta_model
        self.method = method
        self.model_name = name or f"ensemble_{method}"

    def predict(self, X):
        X = np.asarray(X)

        if self.method == "stacking":
            # base_models is list of lists (fold models per base model)
            base_preds = []

            for model_group in self.base_models:
                preds = []
                for m in model_group:
                    Xm = adapt_X_for_model(m, X)
                    preds.append(m.predict(Xm, verbose=state.Verbose_logging).ravel())
                base_preds.append(np.mean(preds, axis=0))

            base_preds = np.column_stack(base_preds)
            return self.meta_model.predict(base_preds)

        else:
            preds = []
            for m in self.base_models:
                Xm = adapt_X_for_model(m, X)
                preds.append(m.predict(Xm, verbose=state.Verbose_logging).ravel())

            preds = np.column_stack(preds)

            if self.method == "average":
                return np.mean(preds, axis=1, keepdims=True)
            elif self.method == "bagging":
                return np.median(preds, axis=1, keepdims=True)


# create and compare different nn ensemble techniques
def compare_nn_ensembles(
    tuned_models,
    tuned_hps,
    X_train,
    y_train,
    X_val,
    y_val,
    metric="r2",
    bagging_rounds=5,   # DEBUG, was 5
    stacking_folds=5,   # DEBUG, was 5
    verbose=state.Verbose_logging
):
    """
    Compare:
    - best single model
    - averaging
    - bagging
    - stacking

    Returns best predictor + diagnostics
    """
    try:
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val   = np.asarray(X_val)
        y_val   = np.asarray(y_val)

        def score(y_true, y_pred):
            if metric == "r2":
                return r2_score(y_true, y_pred)
            elif metric == "mae":
                return -mean_absolute_error(y_true, y_pred)
            else:
                raise ValueError("Unsupported metric")

        results = {}

        # SINGLE BEST MODEL
        print("Evaluating single best models...")
        single_scores = []
        for i, m in enumerate(tuned_models):
            Xv = adapt_X_for_model(m, X_val)
            pred = m.predict(Xv, verbose=state.Verbose_logging).ravel()
            s = score(y_val, pred)
            single_scores.append((s, i))

        best_single_score, best_single_idx = max(single_scores)
        results["single"] = (best_single_score, tuned_models[best_single_idx])

        # SIMPLE AVERAGING
        print("Evaluating averaging ensemble...")
        avg_preds = np.column_stack([
            m.predict(adapt_X_for_model(m, X_val), verbose=state.Verbose_logging).ravel()
            for m in tuned_models
        ]).mean(axis=1)

        results["average"] = (
            score(y_val, avg_preds),
            EnsemblePredictor(
                base_models=tuned_models,
                method="average",
                name="average_ensemble"
            )
        )


        # BAGGING
        print("Evaluating bagging ensemble...")
        bagged_models = []

        for i, base_model in enumerate(tuned_models):
            for b in range(bagging_rounds):
                idx = np.random.choice(len(X_train), len(X_train), replace=True)

                model = clone_model(base_model)
                model.set_weights(base_model.get_weights())

                model.compile(
                    optimizer=fresh_optimizer_from_model_or_hp(base_model, tuned_hps[i]),
                    loss=base_model.loss
                )

                Xt = adapt_X_for_model(model, X_train[idx])
                model.fit(
                    Xt,
                    y_train[idx],
                    epochs=hp_get(tuned_hps[i], "tuner/epochs", 50),
                    batch_size=hp_get(tuned_hps[i], "batch_size", 32),
                    verbose=state.Verbose_logging
                )

                bagged_models.append(model)

        bag_preds = np.column_stack([
            m.predict(adapt_X_for_model(m, X_val), verbose=state.Verbose_logging).ravel()
            for m in bagged_models
        ]).mean(axis=1)

        results["bagging"] = (
            score(y_val, bag_preds),
            EnsemblePredictor(
                base_models=bagged_models,
                method="bagging",
                name="bagging_ensemble"
            )
        )


        # STACKING
        kf = KFold(n_splits=stacking_folds, shuffle=True, random_state=123)

        n_models = len(tuned_models)
        oof = np.zeros((len(X_train), n_models))

        # Store trained fold models per base model
        stack_models = [[] for _ in range(n_models)]

        for fold, (tr, va) in enumerate(kf.split(X_train)):
            if verbose:
                print(f"  Fold {fold+1}/{stacking_folds}")

            for i, base_model in enumerate(tuned_models):
                model = clone_model(base_model)
                model.set_weights(base_model.get_weights())

                model.compile(
                    optimizer=fresh_optimizer_from_model_or_hp(base_model, tuned_hps[i]),
                    loss=base_model.loss
                )

                Xt = adapt_X_for_model(model, X_train[tr])
                model.fit(
                    Xt,
                    y_train[tr],
                    epochs=hp_get(tuned_hps[i], "tuner/epochs", 50),
                    batch_size=hp_get(tuned_hps[i], "batch_size", 32),
                    verbose=state.Verbose_logging
                )

                # Out-of-fold prediction
                Xva = adapt_X_for_model(model, X_train[va])
                oof[va, i] = model.predict(Xva, verbose=state.Verbose_logging).ravel()

                # Save model for inference-time stacking
                model.trainable = False
                stack_models[i].append(model)

        # Tain meta-learner Ridge
        meta = Ridge(alpha=1.0)
        meta.fit(oof, y_train)

        # Validation prediction
        stack_val_base_preds = []

        for i in range(n_models):
            preds = [
                m.predict(adapt_X_for_model(m, X_val), verbose=state.Verbose_logging).ravel()
                for m in stack_models[i]
            ]
            stack_val_base_preds.append(np.mean(preds, axis=0))

        stack_val_preds = meta.predict(np.column_stack(stack_val_base_preds))

        results["stacking"] = (
            score(y_val, stack_val_preds),
            EnsemblePredictor(
                base_models=stack_models, 
                meta_model=meta,
                method="stacking",
                name="stacking_ensemble"
            )
        )

        # SELECT BEST
        best_name, (best_score, best_predictor) = max(
            results.items(), key=lambda x: x[1][0]
        )

        if verbose:
            print("\n=== Ensemble comparison ===")
            for k, (s, _) in results.items():
                print(f"{k:10s}: {s:.4f}")
            print(f"\nBest approach: {best_name} ({metric}={best_score:.4f})")
            best_predictor.model_name = best_name

        return {
            "best_name": best_name,
            "best_predictor": best_predictor,
            "scores": {k: v[0] for k, v in results.items()}
        }

    except Exception as e:
        print(f"There was an error creating ensemble NN models. {e} Fallback to best tuned model and perform prediction.")
        return {
            "best_name": tuned_models[0].model_name,
            "best_predictor": tuned_models[0],
            "scores": {"single": get_r2_manual(tuned_models[0], adapt_X_for_model(tuned_models[0], X_val), y_val)}
        }
