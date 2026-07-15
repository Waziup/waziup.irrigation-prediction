"""Forecast feature assembly and prediction generation/postprocessing.

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
from .features import Weather_derived_feature_cols, add_weather_derived_features
from .nn_architectures import adapt_X_for_model, safe_model_name
from .weather import get_weather_forecast_api, only_get_historical_weather_api


# Create future value testset for prediction
def create_future_values(data, plot):
    # Create ranges and dataframe with timestamps 
    start = data.index[0]
    print("start: ", start)
    train_end = data.index[-1] #+ timedelta(minutes=sample_rate)
    print("train end before adding: ", train_end)
    end = train_end+pd.Timedelta(days=Forecast_horizon)
    print("end after adding: ", end,"\n")
    all_dates = pd.date_range(start=train_end, end=end, freq=str(Sample_rate)+'T')    
    print("all dates: ", all_dates,"\n")

    # Fetch data from weather API
    if plot.load_data_from_csv:
        data_weather_api_cut = only_get_historical_weather_api(train_end, end, plot)
    else:
        data_weather_api_cut = get_weather_forecast_api(train_end, end, plot, data)
    data_weather_api_cut.rename_axis('Timestamp', inplace=True)

    # Create features and merge data from weather API
    new_data = (pd.DataFrame())

    # weather forecast 
    new_data.index = all_dates
    new_data = pd.concat([new_data, data_weather_api_cut], axis=1)
    new_data.reset_index(inplace=True)  # Reset the index
    new_data.rename(columns={'index': 'Timestamp'}, inplace=True)

    # dates
    new_data['hour'] = [i.hour for i in new_data['Timestamp']]
    new_data['minute'] = [i.minute for i in new_data['Timestamp']] #minute is not important
    new_data['date'] = [i.day for i in new_data['Timestamp']]
    new_data['month'] = [i.month for i in new_data['Timestamp']]
    new_data['day_of_year'] = [i.dayofyear for i in new_data['Timestamp']]

    # make up some other data from weatherAPI
    #new_data['rolling_mean_grouped_soil_vol'] = new_data['Soil_moisture_0-7'] #Approach is not any more used
    new_data['rolling_mean_grouped_soil_temp'] = new_data['Soil_temperature_7-28'] # TODO: calculate/calibrate diviation for better alignment

    # also include pump_state or irrigation_amount depending on config, set to zero as we want to assume the behavior without watering
    if "DeviceAndSensorIdsFlow" in plot.config:
        new_data = new_data.assign(irrigation_amount=0)
    else:
        new_data = new_data.assign(pump_state=0)

    # Compute the same weather-derived features as in training (create_features).
    # The rolling windows need antecedent history: prepend the last 7 days of observed
    # weather from the training data so the windows at the forecast start see the real
    # recent conditions instead of an empty warmup, then cut the history rows off again.
    feature_inputs = ['Temperature', 'Humidity', 'Rain', 'Et0_evapotranspiration', 'hour']
    # Give water_balance_7d the measured irrigation history where a flow meter exists
    # (forecast rows themselves carry irrigation_amount=0, assigned above)
    if 'irrigation_amount' in new_data.columns and 'irrigation_amount' in data.columns:
        feature_inputs.append('irrigation_amount')
    first_forecast_ts = new_data['Timestamp'].iloc[0]
    history = data.loc[data.index < first_forecast_ts, feature_inputs].tail(int(168 * 60 / Sample_rate))
    combined = pd.concat(
        [history.reset_index(drop=True), new_data[feature_inputs].reset_index(drop=True)],
        axis=0, ignore_index=True
    )
    combined = add_weather_derived_features(combined)
    new_data[Weather_derived_feature_cols] = combined[Weather_derived_feature_cols].tail(len(new_data)).to_numpy()

    return new_data


# Compare dataframes cols to be sure that they match, otherwise drop
def compare_train_predictions_cols(train, future_features):
    # Identify missing columns in the prediction data
    missing_columns = set(train.columns) - set(future_features.columns) #data.columns
    missing_columns.remove('rolling_mean_grouped_soil') # use array from setup function
    missing_columns.remove('gradient')
    missing_columns.remove('grouped_soil')
    missing_columns.remove('grouped_soil_temp')

    print(missing_columns)
    print(To_be_dropped)

    # drop missing
    for col in missing_columns:
        future_features.drop(columns = col, inplace=True)

    # set_index again on timestamp
    future_features.set_index('Timestamp', inplace=True)
    future_features.head()

    return future_features


# Generate prediction with best_model and impute generated future_values
def generate_predictions(best, exp, features):
    # Generate predictions
    predictions = exp.predict_model(best, data=features)

    # Clip neg predictions to zero
    predictions.loc[predictions['prediction_label'] < 0, 'prediction_label'] = 0

    return predictions


# Generating predictions with neural network model
def generate_predictions_nn(best_model_nn, features, start, end):
    print("Generating predictions with NN model:", safe_model_name(best_model_nn))

    # Ensure features is a numpy array
    X_pred = np.asarray(features)

    # Adapt features to model input shape
    X_pred = adapt_X_for_model(best_model_nn, X_pred)

    # Generate predictions
    predictions = best_model_nn.predict(X_pred)
    
    # Clip neg predictions to zero
    predictions = np.maximum(predictions, 0)

    # Convert the numpy array to a pandas DataFrame and name the column
    predictions = pd.DataFrame(predictions, columns=['prediction_label'])

    # Add timestamps to the dataframe -> create a DatetimeIndex
    date_range = pd.date_range(start=start, end=end, freq=str(Sample_rate)+'T')

    # Ensure the length of date_range matches the DataFrame
    if len(date_range) != len(predictions):
        raise ValueError("Length of date_range does not match length of DataFrame")

    # Replace the index with the new DatetimeIndex
    predictions.index = date_range

    return predictions


# Quadratic weighting function, to be used in align_with_latest_sensor_values -> obsolete
def quadratic_weights(length):
    """
    Generate quadratic weights for blending.
    The weights start high (1.0) and decrease quadratically. (aggressive)
    """
    x = np.linspace(0, 1, length)   # Normalized positions
    weights = (1 - x**2)            # Quadratic decay

    return weights


# Exponential weighting function, to be used in align_with_latest_sensor_values
def exponential_weights(length):
    """
    Generate exponential weights for blending.
    The weights start high (.5) and decrease exponentially to (.1). (moderate)
    """
    x = np.linspace(0, 1, length)   # Normalized positions
    weights = np.exp(-.5*x) -.5     # Exponential decay

    return weights


# align prediction according to latest sensor values
def align_with_latest_sensor_values(plot):
    # Extract the last actual value
    last_actual_value = plot.data['rolling_mean_grouped_soil'].iloc[-1]

    # Generate weights for the prediction range
    weights = exponential_weights(len(plot.predictions))

    # Step 3: Blend the historical and predicted values
    plot.predictions['smoothed_values'] = (
        weights * last_actual_value + (1 - weights) * plot.predictions['prediction_label']
    )


# Calculates the time when threshold will be meet, according to predictions
def calc_threshold(predictions, col, plot):
    threshold = plot.threshold
    strategy = plot.sensor_kind

    # Define comparison logic based on strategy
    comparison_fn = (lambda value, threshold: value > threshold) if strategy == "tension" else (
        lambda value, threshold: value < threshold
    )

    # calculate next occurance
    for i in range(len(predictions)):
        if comparison_fn(predictions[col][i], threshold):
            print("Threshold will be reached on", predictions.index[i], "With a value of:", predictions[col][i])
            return predictions.index[i]

    return ""
