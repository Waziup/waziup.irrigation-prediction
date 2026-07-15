"""Generic dataframe cleaning: gap filling, resampling, dtype conversion.

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


# Impute missing data & apply rolling mean (imputation & cleaning)
def fill_gaps(data):
    # Show if there are any missing values inside the data
    if state.Verbose_logging:
        print("This is before interpolate: \n",data.isna().any())

    data = data.interpolate(method='linear')

    # Show if there are any missing values inside the data
    if state.Verbose_logging:
        print("This is afterwards interpolate: \n",data.isna().any())

    return data


# Remove larger gaps in data as imputing them is not accurate    
def remove_large_gaps(df, col, gap_threshold = 6):
    # Detect NaN gaps in the specified column
    is_nan = df[col].isna()
    
    # Group consecutive NaN and non-NaN values using cumsum on not-NaN
    group = (~is_nan).cumsum()
    
    # Count consecutive NaNs within each group
    consecutive_gaps = is_nan.groupby(group).cumsum()
    
    # Filter out rows where consecutive NaN values exceed the threshold
    df_filtered = df[consecutive_gaps <= gap_threshold].copy()
    
    return df_filtered


# convert to float64
def convert_cols(data):
    obj_dtype = 'object'
    
    for col in data.columns:
        if data[col].dtype == obj_dtype:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        print(col, "has the following dtype: ", data[col].dtype)

    return data


# Resample data
def resample(d):
    d_resample = d.resample(str(Sample_rate)+'T').mean()
    return d_resample
