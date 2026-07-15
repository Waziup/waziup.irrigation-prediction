"""Open-Meteo API access: historical (ERA5) and forecast weather.

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
from .cleaning import convert_cols


# Get historical values from open-meteo TODO: include timezone service: https://stackoverflow.com/questions/16086962/how-to-get-a-time-zone-from-a-location-using-latitude-and-longitude-coordinates
def get_historical_weather_api(data, plot):
    # TODO: remove that one day: overlapping data
    if plot.data_w.empty:
        first_day_minus_one_str = (data.index[0] - timedelta(days = 1)).strftime("%Y-%m-%d")
    else:
        first_day_minus_one_str = (plot.data_w.index[-1] - timedelta(days = 1)).strftime("%Y-%m-%d")

    #plot.data_w_former_end = plot.data_w.index[-1]

    # need to add one day, overlapping have to be cut off later => useless because data is not available to fetch via api
    last_date = data.index[-1]
    last_date_str = last_date.strftime("%Y-%m-%d")

    # TODO: save in instance of specific plot as [] not string :|
    lat = plot.gps_info['lattitude']
    long = plot.gps_info['longitude']

    url = (
        f'https://archive-api.open-meteo.com/v1/era5'
        f'?latitude={lat}'
        f'&longitude={long}'
        f'&start_date={first_day_minus_one_str}'
        f'&end_date={last_date_str}'
        f'&hourly=temperature_2m,relativehumidity_2m,rain,cloudcover,shortwave_radiation,windspeed_10m,winddirection_10m,soil_temperature_7_to_28cm,soil_moisture_0_to_7cm,et0_fao_evapotranspiration'
        f'&timezone={TimeUtils.Timezone}'
    )

    dct = subprocess.check_output(['curl', url]).decode()
    dct = json.loads(dct)

    # Also convert it to a pandas dataframe
    data_w_fetched = (pd.DataFrame([dct['hourly']['temperature_2m'], 
                          dct['hourly']['relativehumidity_2m'], 
                          dct['hourly']['rain'], 
                          dct['hourly']['cloudcover'], 
                          dct['hourly']['shortwave_radiation'],
                          dct['hourly']['windspeed_10m'], 
                          dct['hourly']['winddirection_10m'], 
                          dct['hourly']['soil_temperature_7_to_28cm'], 
                          dct['hourly']['soil_moisture_0_to_7cm'],  
                          dct['hourly']['et0_fao_evapotranspiration'],
                          dct['hourly']['time']], 
                         index = ['Temperature', 'Humidity', 'Rain', 'Cloudcover', 'Shortwave_Radiation', 'Windspeed', 'Winddirection', 'Soil_temperature_7-28', 'Soil_moisture_0-7', 'Et0_evapotranspiration', 'Timestamp'])
            .T
            .assign(Timestamp = lambda x : pd.to_datetime(x.Timestamp, format='%Y-%m-%dT%H:%M'))
            .set_index(['Timestamp'])
            .dropna())

    # Add timezone information without converting 
    data_w_fetched.index = data_w_fetched.index.map(lambda x: x.replace(tzinfo=pytz.timezone(TimeUtils.Timezone)))
    #data_w.index = pd.to_datetime(data_w.index) + pd.DateOffset(hours=get_timezone_offset(timezone))
    
    # convert cols to float64
    data_w_fetched = convert_cols(data_w_fetched)

    # If empty
    if plot.data_w.empty:
        # Set as global
        plot.data_w = data_w_fetched
    # If it is the same
    elif data_w_fetched.index[-1] == plot.data_w.index[-1]:
        return plot.data_w
    # Concat data
    else:
        # Merge former historical weather data and fetched one into one dataframe
        plot.data_w = pd.concat([plot.data_w.loc[plot.data_w.index[0]:],
                            data_w_fetched.loc[plot.data_w.index[-1] + 
                                            timedelta(minutes=Sample_rate) 
                                            : data.index[-1]]
                                            ])

    return plot.data_w


# For debug the weather data has to be retrieved without saving and other stuff from open-meteo 
def only_get_historical_weather_api(start, end, plot):

    start_str = start.strftime("%Y-%m-%d")
    last_date_str = end.strftime("%Y-%m-%d")

    # TODO: save in instance of specific plot as [] not string :|
    lat = plot.gps_info['lattitude']
    long = plot.gps_info['longitude']

    url = (
        f'https://archive-api.open-meteo.com/v1/era5'
        f'?latitude={lat}'
        f'&longitude={long}'
        f'&start_date={start_str}'
        f'&end_date={last_date_str}'
        f'&hourly=temperature_2m,relativehumidity_2m,rain,cloudcover,shortwave_radiation,windspeed_10m,winddirection_10m,soil_temperature_7_to_28cm,soil_moisture_0_to_7cm,et0_fao_evapotranspiration'
        f'&timezone={TimeUtils.Timezone}'
    )

    dct = subprocess.check_output(['curl', url]).decode()
    dct = json.loads(dct)

    # Also convert it to a pandas dataframe
    data_w_fetched = (pd.DataFrame([dct['hourly']['temperature_2m'], 
                          dct['hourly']['relativehumidity_2m'], 
                          dct['hourly']['rain'], 
                          dct['hourly']['cloudcover'], 
                          dct['hourly']['shortwave_radiation'],
                          dct['hourly']['windspeed_10m'], 
                          dct['hourly']['winddirection_10m'], 
                          dct['hourly']['soil_temperature_7_to_28cm'], 
                          dct['hourly']['soil_moisture_0_to_7cm'],  
                          dct['hourly']['et0_fao_evapotranspiration'],
                          dct['hourly']['time']], 
                         index = ['Temperature', 'Humidity', 'Rain', 'Cloudcover', 'Shortwave_Radiation', 'Windspeed', 'Winddirection', 'Soil_temperature_7-28', 'Soil_moisture_0-7', 'Et0_evapotranspiration', 'Timestamp'])
            .T
            .assign(Timestamp = lambda x : pd.to_datetime(x.Timestamp, format='%Y-%m-%dT%H:%M'))
            .set_index(['Timestamp'])
            .dropna())

    # Add timezone information without converting 
    data_w_fetched.index = data_w_fetched.index.map(lambda x: x.replace(tzinfo=pytz.timezone(TimeUtils.Timezone)))
    #data_w.index = pd.to_datetime(data_w.index) + pd.DateOffset(hours=get_timezone_offset(timezone))
    
    # convert cols to float64
    data_w_fetched = convert_cols(data_w_fetched)

    return data_w_fetched


# Get weather forecast from open-meteo
def get_weather_forecast_api(start_date, end_date, plot, data):
    # Compare if forecast date is in the past, important for DEBUG and past data
    daysOpenMeteoHistoricalDataAvailable = 2 # open meteo will have the historical data from the day before yesterday
    local_tz = pytz.timezone(TimeUtils.Timezone)
    now = datetime.now()
    localized_now = local_tz.localize(now)  # Now has timezone info

    if end_date < localized_now - timedelta(days=daysOpenMeteoHistoricalDataAvailable): 
        print("Forecast date is in the past. Using historical weather data instead. Start: ", start_date, " End: ", end_date)
        # Use historical weather data
        return get_historical_weather_api(data, plot)


    # Geo_location
    lat = plot.gps_info['lattitude']
    long = plot.gps_info['longitude']

    # Define the API URL for weather forecast
    url = (
        f'https://api.open-meteo.com/v1/forecast'
        f'?latitude={lat}'
        f'&longitude={long}'
        f'&hourly=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,et0_fao_evapotranspiration,wind_speed_10m,wind_direction_10m,soil_temperature_18cm,soil_moisture_3_to_9cm,shortwave_radiation'
        f'&start_date={start_date.strftime("%Y-%m-%d")}'
        f'&end_date={end_date.strftime("%Y-%m-%d")}'
        f'&timezone={TimeUtils.Timezone}'
    )

    # Use subprocess to run the curl command and decode the output
    dct = subprocess.check_output(['curl', url]).decode()
    dct = json.loads(dct)

    # Convert API response to a pandas dataframe
    data_forecast = (pd.DataFrame([dct['hourly']['temperature_2m'], 
                          dct['hourly']['relative_humidity_2m'], 
                          dct['hourly']['precipitation'], 
                          dct['hourly']['cloud_cover'], 
                          dct['hourly']['shortwave_radiation'],
                          dct['hourly']['wind_speed_10m'], 
                          dct['hourly']['wind_direction_10m'], 
                          dct['hourly']['soil_temperature_18cm'], 
                          dct['hourly']['soil_moisture_3_to_9cm'], 
                          dct['hourly']['et0_fao_evapotranspiration'],
                          dct['hourly']['time']], 
                         index = ['Temperature', 'Humidity', 'Rain', 'Cloudcover', 'Shortwave_Radiation', 'Windspeed', 'Winddirection', 'Soil_temperature_7-28', 'Soil_moisture_0-7', 'Et0_evapotranspiration', 'Timestamp'])
            .T
            .assign(Timestamp = lambda x : pd.to_datetime(x.Timestamp, format='%Y-%m-%dT%H:%M'))
            .set_index(['Timestamp'])
            .dropna())
    
    # Add timezone information without converting 
    #data_forecast.index = data_forecast.index.tz_localize('UTC').tz_convert(timezone)
    #data_forecast.index = data_forecast.index.tz_localize(timezone, utc=True)
    #data_forecast.index = pd.DatetimeIndex(data_forecast.index).tz_localize('UTC').tz_convert('Europe/Berlin')
    data_forecast.index = data_forecast.index.map(lambda x: x.replace(tzinfo=pytz.timezone(TimeUtils.Timezone)))
    #data_forecast.index = pd.to_datetime(data_forecast.index) + pd.DateOffset(hours=get_timezone_offset(timezone)) + pd.DateOffset(hours=1.0)
    
    # convert cols to float64
    data_forecast = convert_cols(data_forecast)

    return data_forecast
