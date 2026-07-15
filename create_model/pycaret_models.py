"""PyCaret regression: compare/tune/ensemble/persist classical models.

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
from .runtime import custom_exception_hook


# Create and compare models
def create_and_compare_model_reg(train):
# (flag now lives in state module)

    # Disable logging to a file
    #logging.basicConfig(filename=None, level=logging.INFO)
    logging.basicConfig(filename="logs.log", level=logging.INFO)

    # create regression exp
    re_exp = pycaret.regression.RegressionExperiment()

    # to rangeindex => do not use timestamps!
    train.reset_index(drop=False, inplace=True)
    train.rename(columns={'index': 'Timestamp'}, inplace=True)

    # old: to_be_dropped = ['minute', 'Timestamp','gradient','grouped_soil','grouped_resistance','grouped_soil_temp']

    print("Those are the remaining features after dropping:", list(set(train.columns.tolist()) - set(To_be_dropped)))

    # Run the following code with a custom exception hook
    sys.excepthook = custom_exception_hook

    # Build a CV strategy whose gap matches the forecast horizon, so folds are scored on
    # genuinely forecasting Forecast_horizon_periods ahead rather than the very next timestamp
    # (which is far easier and would overstate model quality via short-term autocorrelation).
    # n_splits is clamped to whatever stays feasible for the amount of data this plot has.
    train_size = 0.9
    cv_rows = int(len(train) * train_size)
    gap = Forecast_horizon_periods
    max_feasible_folds = max(2, (cv_rows - gap) // (gap + 5))
    fold_count = min(10, max_feasible_folds)
    if fold_count < 10:
        print(f"[create_and_compare_model_reg] Only {cv_rows} rows available after the "
              f"{train_size:.0%} train split; reduced CV folds from 10 to {fold_count} to "
              f"keep the {gap}-period forecast-horizon gap feasible.")
    cv_strategy = TimeSeriesSplit(n_splits=fold_count, gap=gap)

    # Run pycarets setup
    # normalize=True: z-score inside pycaret's own pipeline (fit on the train split only,
    # so leakage-safe). Irrelevant for the tree/boosting models but needed by the
    # scale-sensitive ones (knn, svm, mlp, regularized linear) once the full model zoo runs.
    s = re_exp.setup(train,
              target = 'rolling_mean_grouped_soil',
              session_id = 123,
              verbose = state.Verbose_logging,
              ignore_features = To_be_dropped,
              train_size = train_size,
              normalize = True,
              fold_strategy=cv_strategy,
              data_split_shuffle=False,
              fold_shuffle=False,
              n_jobs = None
    )
    
    # Print available models
    print("Available models: ", re_exp.models())

    # Save pycarets setup state.Config to be persistant
    state.Config = {
        "target": re_exp.target_param,
        "session_id": 123,
        "ignore_features": re_exp._fxs.get("Ignore", []),
        "numeric_features": re_exp._fxs.get("Numeric", []),
        "categorical_features": re_exp._fxs.get("Categorical", []),
        "train_size": re_exp.train_size_param if hasattr(re_exp, "train_size_param") else 0.8,
        "normalize": True,
        "verbose": state.Verbose_logging
    }
    
    # Run compare_models function TODO: configure setup accordingly
    best_re = re_exp.compare_models(
        n_select = 19,
        fold = fold_count, # matches the gap-aware TimeSeriesSplit built above
        sort = 'R2',
        verbose = state.Verbose_logging,           
        errors="raise",   # DEBUG: raise, ignore, warn
        exclude=['lar', 'dummy', 'lightgbm', 'lr', 'par'], # excluded those that do not perform well (bad R2 on testset)
        #include=['xgboost', 'catboost'] #DEBUG
    )

    return re_exp, best_re


# Save the best models
def save_models(plot_name, exp, best, path_to_save):
    try:
        # save pipeline
        model_names = []

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

        # type check for array -> convert
        if not isinstance(best, list):
            best = [best]

        for i in range(len(best)):
            full_path = path_to_save + str(i) + "_" + best[i].__module__ + "_" + plot_name
            exp.save_model(best[i], full_path)
            model_names.append(full_path)
            
        return model_names
    except Exception as e:
        print("An error occurred while saving the models:", str(e))
        return []


# TODO: model_names will not work if it was not saved before, adjust paths accordingly        
# Load the best models        
def load_models(model_names):
    # load pipeline
    loaded_best_pipeline = []
    for i in range(len(model_names)): # TODO: model_names will not work if it was not saved before
        loaded_best_pipeline.append(load_model(model_names[i]))
    
    return loaded_best_pipeline


# train the best model after eval with fulln dataset
def train_best(best_model, data):
# (flag now lives in state module)

    # create regression exp
    re_exp = pycaret.regression.RegressionExperiment()

    # to rangeindex => do not use timestamps!
    data = data.reset_index(drop=True, inplace=False) #TODO: maybe problems here with live data from gw
    data = data.rename(columns={'index': 'Timestamp'}, inplace=False)

    # old: to_be_dropped = ['minute', 'Timestamp','gradient','grouped_soil','grouped_resistance','grouped_soil_temp']
    print("Those are the remaining features after dropping:", list(set(data.columns.tolist()) - set(To_be_dropped)))

    # Run the following code with a custom exception hook => maybe only relevant in VSCode => TODO: test without in production
    sys.excepthook = custom_exception_hook

    # Build a CV strategy whose gap matches the forecast horizon, so folds are scored on
    # genuinely forecasting Forecast_horizon_periods ahead rather than the very next timestamp
    # (which is far easier and would overstate model quality via short-term autocorrelation).
    # n_splits is clamped to whatever stays feasible for the amount of data this plot has.
    train_size = 0.9
    cv_rows = int(len(data) * train_size)
    gap = Forecast_horizon_periods
    max_feasible_folds = max(2, (cv_rows - gap) // (gap + 5))
    fold_count = min(10, max_feasible_folds)
    if fold_count < 10:
        print(f"[create_and_compare_model_reg] Only {cv_rows} rows available after the "
              f"{train_size:.0%} train split; reduced CV folds from 10 to {fold_count} to "
              f"keep the {gap}-period forecast-horizon gap feasible.")
    cv_strategy = TimeSeriesSplit(n_splits=fold_count, gap=gap)

    # Run pycarets setup
    s = re_exp.setup(data,
              target = 'rolling_mean_grouped_soil',
              session_id = 123,
              verbose = state.Verbose_logging,
              ignore_features = To_be_dropped,
              train_size = train_size,
              normalize = True,
              fold_strategy=cv_strategy,
              data_split_shuffle=False,
              fold_shuffle=False,
              n_jobs = None 
    )

    # Save pycarets setup state.Config to be persistant
    state.Config = {
        "target": re_exp.target_param,
        "session_id": 123,
        "ignore_features": re_exp._fxs.get("Ignore", []),
        "numeric_features": re_exp._fxs.get("Numeric", []),
        "categorical_features": re_exp._fxs.get("Categorical", []),
        "train_size": re_exp.train_size_param if hasattr(re_exp, "train_size_param") else 0.8,
        "normalize": True,
        "verbose": state.Verbose_logging
    }
    
    # Pass the already-tuned/ensembled estimator object straight to create_model() instead of
    # looking its type up in Model_mapping. pycaret clones whatever estimator object it's given
    # (sklearn's clone(), preserving its hyperparameters via get_params()) and fits that clone on
    # this setup's training data - Model_mapping only covers pycaret's atomic base model IDs, so
    # it raised KeyError for compound estimators (Pipeline, StackingRegressor, VotingRegressor from
    # stack_models()/blend_models()). Worse, even for atomic models it was silently rebuilding a
    # fresh DEFAULT-hyperparameter model of the same class instead of refitting the tuned one -
    # discarding all tuning whenever it didn't outright crash.
    if not isinstance(best_model, list):
        # Create one model
        model = re_exp.create_model(best_model)

        return model, re_exp
    else:
        # Create multiple models, for ensemble
        model = []
        for m in best_model:
            # Create model
            model.append(re_exp.create_model(m))

        return model, re_exp


# Extract model from pipeline
def unwrap_model(m):
    if isinstance(m, Pipeline):
        return m.steps[-1][1]
    return m


def init_pycaret_subprocess_tuning(plot_name, exp, best_models):
    #global state.Config

    try:
        # save data, model and exp to disk TODO: evaluate tmp folder-> overflow
        temp_dir = Path("./tmp") / f"tuning_{plot_name}_{datetime.now().timestamp()}" #TODO: later cleanup old temp folders
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save full dataset
        exp.dataset.to_csv(temp_dir / "data.csv", index=False)
        # Save test set (optional but precise)
        #plot.best_exp.X_test.to_csv(temp_dir / "X_test.csv", index=False)
        #plot.best_exp.y_test.to_csv(temp_dir / "y_test.csv", index=False)
        #plot.best_exp.save_experiment(str(temp_dir / "experiment.pkl")) # does not work

        # # Save config -> TODO: init later globally 
        # state.Config = {
        #     "target": exp.target_param,
        #     "session_id": 123,  # you defined this manually
        #     "ignore_features": exp._fxs.get("Ignore", []),
        #     "numeric_features": exp._fxs.get("Numeric", []),
        #     "categorical_features": exp._fxs.get("Categorical", []),
        #     "train_size": exp.train_size_param if hasattr(exp, "train_size_param") else 0.8,
        # }
        with open(temp_dir / "config.pkl", "wb") as f:
            pickle.dump(state.Config, f)

        # Save models to be available in subprocess   
        list_of_model_names = save_models(plot_name, exp, best_models, str(temp_dir / "best_model_before_tuning_"))
        best_tuned_models = []
        for name in list_of_model_names:
            print(f"Saved model {name} for tuning in {temp_dir / 'best_model.pkl'}")
            # Run tuning in subprocess
            best_model_path = run_tuning_with_subprocess(
                str(temp_dir), 
                name,
                plot_name
            )
            # Load models from disk after tuning
            m = load_model(best_model_path)
            # Make sure to load the actual model if it was wrapped in a pipeline
            m = unwrap_model(m)
            # Add to list of best tuned models
            best_tuned_models.append(m)
        # If only one model, return it directly instead of a list
        best_tuned_models = best_tuned_models if len(best_tuned_models) > 1 else best_tuned_models[0]

        # Save best tuned pycaret model
        model_names = save_models(plot_name, exp, best_models, f'models/{plot_name}/tuned_models/pycaret/best_soil_tension_prediction_')

        return best_tuned_models

    except Exception as e:
        print(f"[{plot_name}] Error during tuning: {e}, using the original model.")
        return best_models

    finally:
        # Cleanup method
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Deleted temp dir after successful tuning: {temp_dir}")         


# Tune hyperparameters of one model
def tune_one_model(exp, best):
    try: # double try, except
        # fallback: infer from class name
        model_id = Model_mapping[best.__class__.__name__]
        # Load grid
        grid = PYCARET_REGRESSION_TUNE_GRIDS.get(model_id)

        if not grid:
            print(f"No grid for model_id='{model_id}', skipping tuning.")
            return best
    
        best = exp.tune_model(best, choose_better = True, custom_grid=grid) # Throws ERROR: grid missing
        return best
    except Exception as e:
        print(f"There was an error tuning the model. {e}")
        return best


# Tune hyperparameters of several models -> used for ensemble learning, not implemented yet
def tune_models(exp, best):
    # tune hyperparameters of dt
    tuned_best_models = []
    for i in range(len(best)):
        print("This is for the",i+1,"model:",best[i])
        # get model id and grid
        model_id = Model_mapping[best[i].__class__.__name__]
        # Load grid
        grid = PYCARET_REGRESSION_TUNE_GRIDS.get(model_id)
        # tune model and append
        tuned_best_models.append(
            exp.tune_model(best[i], choose_better = True)#, custom_grid=grid)
        ) # check grid again: xgboost, !!!catboost!!! is VERY SLOW
        
    return tuned_best_models


def safe_make(label: str, fn):
    """
    Safely executes a function (fn). 
    If it fails, prints an error with the given label and returns None.
    """
    try:
        result = fn()  # IMPORTANT: fn must be callable
        if result is None:
            raise ValueError(f"{label} returned None")
        return result
    except Exception as e:
        print(f"{label} failed: {e}")
        traceback.print_exc()
        return None


def log_scores_csv(scores, filename="model_scores.csv"):

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if file exists to write header
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        # Header only once
        if not file_exists:
            writer.writerow(["timestamp", "type", "r2", "model"])
        
        ts = datetime.now().isoformat()

        for key, (r2, model) in scores.items():
            writer.writerow([ts, key, r2, type(model).__name__ if model else "None"])


def init_pycaret_subprocess_ensemble(plot_name, exp, tuned_best_models):
    try:
        # save data, model and exp to disk TODO: evaluate tmp folder-> overflow
        temp_dir = Path("./tmp") / f"tuning_{plot_name}_{datetime.now().timestamp()}" #TODO: later cleanup old temp folders
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save full dataset
        exp.dataset.to_csv(temp_dir / "data.csv", index=False)

        # Save config to file
        with open(temp_dir / "config.pkl", "wb") as f:
            pickle.dump(state.Config, f)
        
        # Save models to be available in subprocess   
        list_of_model_names = save_models(plot_name, exp, tuned_best_models, str(temp_dir / "best_model_before_ensemble_"))

        # Extract model
        tuned_best_models = [unwrap_model(m) for m in tuned_best_models]
    
        # Run ensemble creation in subprocess
        best_ensemble_model_path = run_ensemble_with_subprocess(
            str(temp_dir),
            list_of_model_names,
            plot_name
        )

        # Load ensemble model from disk after creation
        best_ensemble_model = unwrap_model(load_model(best_ensemble_model_path))

         # Save best tuned pycaret model
        model_names = save_models(plot_name, exp, best_ensemble_model, f'models/{plot_name}/ensemble_models/pycaret/best_soil_tension_prediction_')

        return best_ensemble_model

    except Exception as e:
        print(f"[{plot_name}] Error during tuning: {e}, using the original model.")
        return tuned_best_models


def init_pycaret_subprocess_tuning_and_ensemble(plot_name, exp, tuned_best_models, ensemble=True):
    try:
        # save data, model and exp to disk TODO: evaluate tmp folder-> overflow
        temp_dir = Path("./tmp") / f"tuning_{plot_name}_{datetime.now().timestamp()}" #TODO: later cleanup old temp folders
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save full dataset
        exp.dataset.to_csv(temp_dir / "data.csv", index=False)

        # Save config to file
        with open(temp_dir / "config.pkl", "wb") as f:
            pickle.dump(state.Config, f)
        
        # Save models to be available in subprocess   
        list_of_model_names = save_models(plot_name, exp, tuned_best_models, str(temp_dir / "best_model_before_tuning and_ensemble_"))

        # Extract model
        tuned_best_models = [unwrap_model(m) for m in tuned_best_models]
    
        # Run ensemble creation in subprocess
        best_ensemble_model_path = run_tuning_and_ensemble_with_subprocess(
            str(temp_dir),
            list_of_model_names,
            plot_name
        )

        # Load ensemble model from disk after creation
        best_ensemble_model = unwrap_model(load_model(best_ensemble_model_path))

         # Save best ensemble pycaret model
        model_names = save_models(plot_name, exp, best_ensemble_model, f'models/{plot_name}/ensemble_models/pycaret/best_soil_tension_prediction_')

        return best_ensemble_model

    except Exception as e:
        print(f"[{plot_name}] Error during tuning: {e}, using the original model.")
        return tuned_best_models


# Create and compare different ensemble model techniques for pycaret models
def create_and_compare_ensemble(plot_name, exp, tuned_best_models):
    try:
        print(f"Creating ensemble models from {len(tuned_best_models)} tuned base models.")
        # Base model is the tuned one
        if isinstance(tuned_best_models, list):
            base_models = tuned_best_models
        else:
            base_models = [tuned_best_models]

        # Safety: keep only models that have a fit() method
        base_models = [m for m in base_models if hasattr(m, "fit")]
        if not base_models:
            raise ValueError("No valid base models with a .fit() method were provided.")

        # Build ensemble techniques independently
        stacked_model = safe_make(
            "stacked_model",
            lambda: exp.stack_models(
                estimator_list=base_models,
                choose_better=False
            )
        )

        blended_model = safe_make(
            "blended_model",
            lambda: exp.blend_models(
                estimator_list=base_models,
                choose_better=False
            )
        )

        bagged_model = safe_make(
            "bagged_model",
            lambda: exp.ensemble_model(
                base_models[0],
                method="Bagging",
                choose_better=False
            )
        )

        boosted_model = safe_make(
            "boosted_model",
            lambda: exp.ensemble_model(
                base_models[0],
                method="Boosting",
                choose_better=False
            )
        )

        # Evaluate all candidates
        # tuned: just take the *best* single base model by R2 as baseline
        r2_base = [(get_r2_manual(m, exp.X_test, exp.y_test), m) for m in base_models]
        r2_tuned, best_single_tuned = max(r2_base, key=lambda x: x[0])

        r2_stacked = get_r2_manual(stacked_model, exp.X_test, exp.y_test)
        r2_blended = get_r2_manual(blended_model, exp.X_test, exp.y_test)
        r2_bagged  = get_r2_manual(bagged_model, exp.X_test, exp.y_test)
        r2_boosted = get_r2_manual(boosted_model, exp.X_test, exp.y_test)

        # Pick the best by R2
        scores = {
            "tuned":   (r2_tuned,   best_single_tuned),
            "stacked": (r2_stacked, stacked_model),
            "blended": (r2_blended, blended_model),
            "bagged":  (r2_bagged,  bagged_model),
            "boosted": (r2_boosted, boosted_model),
        }

        best_name, (best_r2, best_model) = max(scores.items(), key=lambda x: x[1][0])

        print(f"Best ensemble strategy: {best_name} with R2 = {best_r2:.4f}")

        if state.Verbose_logging:
            log_scores_csv(scores, filename=f"models/{plot_name}/best_models/pycaret/ensemble_scores.csv")

        return best_model
    
    except Exception as e:
        print(f"There was an error creating ensemble models. {e} "
              f"Fallback to best tuned model and perform prediction.")
        return tuned_best_models[0]
