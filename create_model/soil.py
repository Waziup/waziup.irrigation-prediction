"""Soil water retention: tension <-> volumetric water content conversion.

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


# VWC with splines scale
def soil_tension_to_volumetric_water_content_spline(soil_tension, soil_water_retention_curve):
    """
    Convert soil tension (kPa) to volumetric water content (fraction) using cubic spline interpolation.
    
    Parameters:
        soil_tension (float): Soil tension value in kPa.
        soil_water_retention_curve (list of tuples): A list of tuples containing points on the soil-water retention curve.
            Each tuple contains two elements: (soil_tension_value, volumetric_water_content_value).
    
    Returns:
        float: Volumetric water content as a fraction (between 0 and 1).
    """
    # Extract tension and water content values from the curve
    tensions, water_contents = zip(*soil_water_retention_curve)
    
    # Create a cubic spline interpolator
    spline = CubicSpline(tensions, water_contents, bc_type='natural')

    # Evaluate the spline at the given soil tension
    interpolated_water_content = spline(soil_tension)
    
    # Clip the result to ensure it remains within the valid range [0, 1]
    return np.clip(interpolated_water_content, 0, 1)


def add_volumetric_col_to_df(df, col_name, plot):
    # Iterate over the rows of the dataframe and calculate volumetric water content
    soil_water_retention_tupel_list = [(float(dct['Soil tension']), float(dct['VWC'])) for dct in plot.config['Soil_water_retention_curve']]
    # Sort the soil-water retention curve points by soil tension in ascending order => TODO: NOT EFFICIENT HERE, move out
    sorted_curve = sorted(soil_water_retention_tupel_list, key=lambda x: x[0])
    for index, row in df.iterrows():
        soil_tension = row[col_name]
        # Calculate volumetric water content
        volumetric_water_content = soil_tension_to_volumetric_water_content_spline(soil_tension, sorted_curve)
        # Assign the calculated value to a new column in the dataframe
        df.at[index, col_name + '_vol'] = round(volumetric_water_content, 4)

    return df


# Calculate a single soil tension value to VWC 
def calc_volumetric_water_content_single_value(soil_tension_value, currentPlot):
    # Check config being loaded, otherwise read it
    if not currentPlot.config:
        currentPlot.config = currentPlot.read_config() # this is just for the case of returning to index, after settings was created/changed
    # Iterate over the rows of the dataframe and calculate volumetric water content
    soil_water_retention_tupel_list = [(float(dct['Soil tension']), float(dct['VWC'])) for dct in currentPlot.config['Soil_water_retention_curve']]
    # Sort the soil-water retention curve points by soil tension in ascending order => TODO: NOT EFFICIENT HERE, move out
    sorted_curve = sorted(soil_water_retention_tupel_list, key=lambda x: x[0])
    # Calculate volumetric water content
    volumetric_water_content = soil_tension_to_volumetric_water_content_spline(soil_tension_value, sorted_curve)

    return volumetric_water_content
