"""System/runtime utilities: memory management and training guard callbacks.

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


# Restrict time to training
class TimeLimitCallback(Callback):
    def __init__(self, max_time_seconds):
        super(TimeLimitCallback, self).__init__()
        self.max_time_seconds = max_time_seconds
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):  # Changed to on_epoch_end
        elapsed_time = time.time() - self.start_time
        if state.Verbose_logging:
            print(f"\nEpoch {epoch}: Elapsed time {elapsed_time:.2f} seconds")
        if elapsed_time > self.max_time_seconds:
            self.model.stop_training = True
            print(f"\nTraining stopped after {self.max_time_seconds} seconds") # will continue after epoch


# Custom exception for memory limit reached
class MemoryLimitReachedError(Exception):
    """Custom exception to stop the entire tuning process."""
    pass


class MemoryLimitCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if psutil.virtual_memory().percent > Memory_limit_percent:
            print(f"!!! Memory limit reached (>{Memory_limit_percent}%). Terminating all tuning trials. !!!")
            # Raising an error here breaks out of tuner.search() completely
            raise MemoryLimitReachedError("System memory exhausted during tuning")


class HardCleanupCallback(tensorflow.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        import gc
        tensorflow.keras.backend.clear_session()
        gc.collect()


# Release RAM back to the OS after memory-intensive stages - essential on 4GB RPi.
# gc.collect() alone is not enough: Python/NumPy return freed memory to glibc's
# allocator, but glibc keeps the pages in its arenas, so process RSS stays high.
# malloc_trim(0) is what actually hands the freed pages back to the kernel.
def free_memory(clear_keras=False, label=""):
    """
    clear_keras=True additionally resets the global Keras/TF state (graphs, cached
    tf.functions). Only pass it when NO already-built Keras model is needed afterwards
    - models built before clear_session() are not reliably usable for further
    fit/clone calls. Weights of unreferenced models are freed by gc either way.
    """
    if clear_keras:
        tensorflow.keras.backend.clear_session()

    gc.collect()

    try:
        # glibc only; harmless no-op guard elsewhere (e.g. musl-based images)
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    if state.Verbose_logging:
        rss_mb = psutil.Process().memory_info().rss / 1024 ** 2
        print(f"[MEM] freed after {label or 'cleanup'}: RSS={rss_mb:.0f}MB, system={psutil.virtual_memory().percent:.0f}%")


# Custom exception hook is need to debug in VSCode
def custom_exception_hook(exctype, value, traceback):
    # Your custom exception handling code here
    print(f"Exception Type: {exctype}\nValue: {value}")
    print(f"Trace:{traceback}")
