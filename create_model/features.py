"""Data preparation and feature engineering for the training dataset.

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
from .cleaning import convert_cols, fill_gaps, remove_large_gaps, resample
from .weather import get_historical_weather_api, get_weather_forecast_api


# TODO: more sophisticated approach needed: needs to learn from former => introduce model, is now excluded when flow meter is installed
def add_pump_state(data, plot):
    slope = plot.slope
    # for index, row in data.iterrows():
    #     if row['gradient'] < slope:
    #         #print(index, row['rolling_mean_grouped_soil'], row['gradient'], row['Rain'])
    #         # only add if there was no rain in the previous hours
    #         if row['Rain'] == 0.0:
    #             data.at[index, 'pump_state'] = 1

    for i in range(1,len(data)):
        # look for two consecutive occurances of high negative slope
        if data.iloc[i-1]['gradient'] < slope and data.iloc[i]['gradient'] < slope:
            # and if it rained now and previously
            if data.iloc[i-1]['Rain'] == 0.0 and data.iloc[i]['Rain'] == 0.0:
                data.loc[data.index[i], 'pump_state'] = 1
                data.loc[data.index[i-1], 'pump_state'] = 1

    return data


# Function to ensure the JSON file exists or create it if missing
def ensure_json_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump({"irrigations": []}, file)
        print(f"Created new JSON file at: {file_path}")


# include the (on device saved) amount of irrigation given
def include_irrigation_amount(df, plot):
    irrigation_file = plot.irrigations_from_json
    # Check and ensure the JSON file exists
    ensure_json_file(irrigation_file)

    if plot.load_irrigations_from_file:
        # Load JSON data from file
        with open(irrigation_file, 'r') as file:
            irrigations_json = json.load(file)

        print("Loaded JSON data:")
        print(irrigations_json)

        # Convert the 'irrigations' list to a pandas DataFrame
        irrigations_df = pd.DataFrame(irrigations_json['irrigations'])

        print("\nDataFrame from JSON:")
        print(irrigations_df.head())  # Check the first few rows to ensure data is loaded correctly

        # Convert timestamp to datetime
        irrigations_df['timestamp'] = pd.to_datetime(irrigations_df['timestamp'])

        # Set timestamp as index
        irrigations_df.set_index('timestamp', inplace=True)

        print("\nDataFrame after timestamp conversion and setting index:")
        print(irrigations_df.head())  # Check again to ensure timestamps are converted correctly

        # Resample the irrigations dataframe to hourly intervals, summing the amounts
        irrigations_resampled = irrigations_df.resample('H').sum()

        # Reindex the irrigations dataframe to match the main dataframe's index, filling missing values with zero
        irrigations_reindexed = irrigations_resampled.reindex(df.index, fill_value=0)

        print("\nReindexed DataFrame:")
        print(irrigations_reindexed.head())  # Check reindexed DataFrame to see if it aligns with df's index

        # Add the reindexed irrigation amounts to the main dataframe
        df['irrigation_amount'] = irrigations_reindexed['amount']
    else:
        if (len(plot.device_and_sensor_ids_flow) >  0):
            # Load data and create the dataframe
            data_irrigation = plot.load_data_api(plot.device_and_sensor_ids_flow[0], "actuators", plot.start_date)
            df_irrigation = pd.DataFrame(data_irrigation)
            
            # Check if the dataframe is empty
            if df_irrigation.empty:
                print("No irrigation data found for the specified device and sensor IDs.")
                return df
            
            # Rename columns for consistency
            df_irrigation.rename(columns={'time': 'Timestamp'}, inplace=True)

            # Convert the 'Timestamp' column to datetime, ensuring it is in UTC
            df_irrigation['Timestamp'] = pd.to_datetime(df_irrigation['Timestamp'], utc=True)
            df_irrigation['Timestamp'] = df_irrigation['Timestamp'].dt.tz_convert(TimeUtils.Timezone)  # Replace with the correct timezone

            df_irrigation.rename(columns={'value': 'irrigation_amount'}, inplace=True)

            # Convert irrigation_amount to numeric, forcing errors to NaN
            df_irrigation['irrigation_amount'] = pd.to_numeric(df_irrigation['irrigation_amount'], errors='coerce')

            # Round the timestamps to the nearest hour and aggregate the irrigation amounts
            df_irrigation['Timestamp'] = df_irrigation['Timestamp'].dt.round('H')
            df_irrigation = df_irrigation.groupby('Timestamp').agg({'irrigation_amount': 'sum'}).reset_index()

            # subtract one hour, like wtf!!!!
            #df_irrigation['Timestamp'] = df_irrigation['Timestamp'] - pd.to_timedelta(1, unit='h')

            # Set the Timestamp as the index
            df_irrigation.set_index('Timestamp', inplace=True)

            # Timezone has to be set for df_irrigation
            df_irrigation = df_irrigation.tz_convert(TimeUtils.Timezone)

            # Merge the dataframes
            df = pd.merge(df, df_irrigation, left_index=True, right_index=True, how='outer', suffixes=('_main', '_irrigation'))

            # Fill missing irrigation_amount with 0
            df['irrigation_amount'] = df['irrigation_amount'].fillna(0)

    return df


# Weather-derived features for irrigation timing. Soil tension has physical memory:
# today's value depends on how wet the last days were, not just on the current hour.
# Every input is a weather or clock column (never the target), so the same function can
# run on the training frame (create_features) and on the forecast frame
# (create_future_values) without leaking target information, and all features stay
# computable at prediction time from the weather forecast alone.
def add_weather_derived_features(df):
    # Window sizes in rows (rows are resampled to Sample_rate minutes).
    # min_periods=1 keeps partial sums at the start of the frame instead of NaN rows.
    def rows(hours):
        return max(1, int(hours * 60 / Sample_rate))

    # Antecedent water supply and atmospheric demand
    df['rain_sum_24h'] = df['Rain'].rolling(rows(24), min_periods=1).sum()
    df['rain_sum_72h'] = df['Rain'].rolling(rows(72), min_periods=1).sum()
    df['rain_sum_7d'] = df['Rain'].rolling(rows(168), min_periods=1).sum()
    df['et0_sum_72h'] = df['Et0_evapotranspiration'].rolling(rows(72), min_periods=1).sum()

    # Simplified running soil water balance over the last 7 days: supply - demand.
    # Supply includes measured irrigation when a flow meter provides it (the forecast
    # frame assigns irrigation_amount=0, i.e. "behavior without watering", which is the
    # scenario the prediction is asking about).
    water_in = df['Rain']
    if 'irrigation_amount' in df.columns:
        water_in = water_in + df['irrigation_amount'].fillna(0)
    et0_sum_7d = df['Et0_evapotranspiration'].rolling(rows(168), min_periods=1).sum()
    df['water_balance_7d'] = water_in.rolling(rows(168), min_periods=1).sum() - et0_sum_7d

    # Vapour pressure deficit (kPa, Tetens formula): drives transpiration more directly
    # than temperature or humidity alone
    saturation_vp = 0.6108 * np.exp(17.27 * df['Temperature'] / (df['Temperature'] + 237.3))
    df['vpd'] = saturation_vp * (1 - df['Humidity'] / 100)

    # Hours since last significant rain (>0.5mm), capped at 7 days (also the fill value
    # for leading rows before the first observed rain)
    positions = np.arange(len(df))
    rained = df['Rain'].to_numpy(dtype=float) > 0.5
    last_rain_pos = pd.Series(np.where(rained, positions, np.nan)).ffill().to_numpy()
    hours_since = (positions - last_rain_pos) * Sample_rate / 60
    df['hours_since_rain'] = pd.Series(hours_since, index=df.index).fillna(168.0).clip(upper=168.0)

    # Cumulative atmospheric demand since the soil was last wetted by rain - the classic
    # dry-down signal: tension rises roughly with accumulated ET0 after wetting. Capped at
    # the 7-day ET0 total so the value stays identical whether it is computed on the full
    # training history or on the 7-day warmup prepended in create_future_values.
    wetting_group = pd.Series(np.cumsum(rained), index=df.index)
    et0_since = df['Et0_evapotranspiration'].groupby(wetting_group).cumsum()
    df['et0_since_rain'] = np.minimum(et0_since, et0_sum_7d)

    # Cyclical encoding of hour, so 23:00 and 00:00 are neighbours (mainly for the NNs;
    # the raw 'hour' column stays available for the tree models)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df



# Column names produced by add_weather_derived_features - single source of truth so the
# training and forecast frames cannot drift apart (compare_train_predictions_cols crashes
# on a column mismatch between them)
Weather_derived_feature_cols = [
    'rain_sum_24h', 'rain_sum_72h', 'rain_sum_7d', 'et0_sum_72h',
    'water_balance_7d', 'vpd', 'hours_since_rain', 'et0_since_rain',
    'hour_sin', 'hour_cos'
]


# Augment the dataset creating new features
def create_features(data, plot):
    # Create average cols
    data['grouped_soil'] = data[plot.device_and_sensor_ids_moisture].mean(axis=1)
    data['grouped_soil_temp'] = data[plot.device_and_sensor_ids_temp].mean(axis=1)
    
    # Create rolling mean: introduces NaN again -> later just cut off
    data['rolling_mean_grouped_soil'] = data['grouped_soil'].rolling(window=RollingMeanWindowGrouped, win_type='gaussian').mean(std=RollingMeanWindowGrouped)
    data['rolling_mean_grouped_soil_temp'] = data['grouped_soil_temp'].rolling(window=RollingMeanWindowGrouped, win_type='gaussian').mean(std=RollingMeanWindowGrouped)
    
    # Drop those values without rolling_mean /// was 18 before 
    data = data[4:]

    # Resample data
    data = resample(data)

    # Create time related features
    data['hour'] = data.index.hour#.astype("float64")
    data['minute'] = data.index.minute#.astype("float64")
    data['date'] = data.index.day#.astype("float64")
    data['month'] = data.index.month#.astype("float64")
    data['day_of_year'] = data.index.dayofyear#.astype("float64")

    # Save the length of sensordata to var in days -> to dynamically adjust train interval
    plot.train_period_days = (data.index[-1] - data.index[0]).days

    # Get weather from weather meteo
    data_weather = get_historical_weather_api(data, plot)

    # Resample weatherdata before merge => takes a long time
    data_weather = resample(data_weather)

    # historical weather data is not available for the latest two days, use forecast to account for that!
    if not plot.load_data_from_csv:
        data_weather_endtime = data_weather.index[-1]
        data_endtime = data.index[-1]

        # Get forecast for the ~last two days
        data_weather_recent_forecast = get_weather_forecast_api(data_weather_endtime, data_endtime, plot, data)

        # Merge weather data to one dataframe
        data_weather_merged = pd.concat([data_weather.loc[data.index[0]:], 
                                        data_weather_recent_forecast.loc[data_weather_endtime + 
                                                                        timedelta(minutes=Sample_rate) 
                                                                        : data_endtime]
                                                                        ])
    else:
        data_weather_merged = data_weather.loc[data.index[0]:data.index[-1]]

    # Merge data_weather_merged into data
    data = pd.merge(data, data_weather_merged, left_index=True, right_index=True, how='outer')

    # # Calculate and add volumetric water content => do not use this approach, does not yield better results
    # data = add_volumetric_col_to_df(data, 'rolling_mean_grouped_soil')
    # # align soil water retention curve with data from API => do not use this approach, does not yield better results
    # corrected_water_retention_curve = (data, data_weather)
    # # Drop not aligned curve
    # data = data.drop(columns=['rolling_mean_grouped_soil_vol'])
    # # Calculate and add CORRECTED volumetric water content
    # data = add_volumetric_col_to_df(data, 'rolling_mean_grouped_soil', corrected_water_retention_curve)

    # Omit rows when there is no data from physical sensors for over six hours => below: interpolate
    data = remove_large_gaps(data, 'rolling_mean_grouped_soil', 6)

    # Check gaps => TODO: not every col should interpolated (month?), some data is lost here
    data = fill_gaps(data)

    # Add calculated pump state or actual irrigation amount
    # Calc gradient
    f = data.rolling_mean_grouped_soil
    data['gradient'] = np.gradient(f)
    # Skip the pump state if there is a flow meter where the artificial irrigation amount is measured
    if "DeviceAndSensorIdsFlow" in plot.config:
        data = include_irrigation_amount(data, plot)
    else:
        data['pump_state'] = int(0)
        data = add_pump_state(data, plot)

    # Add weather-derived features (antecedent rain/ET0 windows, water balance, VPD,
    # hours since rain, ET0 since rain, cyclical hour) - after fill_gaps so the windows
    # see complete series, and after the irrigation block so water_balance_7d can include
    # measured irrigation amounts where a flow meter exists
    data = add_weather_derived_features(data)

    # also add hours since last irrigation => TODO: check later, still an error, !!!!!questionable whether it is useful!!!!!
    #data = hours_since_pump_was_turned_on(data)
    
    return data


# Split dataset into train and test set by ratio     
def split_by_ratio(data):
    # Calculate the number of rows for the test set
    n = len(data)

    train_end = int(n * 0.64) # hard coded, TODO: dynamically would be better
    val_end   = int(n * 0.80)

    train_df = data.iloc[:train_end]
    val_df   = data.iloc[train_end:val_end]
    test_df  = data.iloc[val_end:]

    return train_df, val_df, test_df


# Data preparation pipeline, calls other subfunction to perform the task
def prepare_data(plot):
    # Load data from local wazigate api -> each sensor individually
    data_moisture = []
    data_temp = []

    # start date is in UTC, but user expects it in his timezone
    start_date = plot.start_date
    lat = plot.gps_info['lattitude']
    long = plot.gps_info['longitude']
    TimeUtils.Timezone = TimeUtils.get_timezone(lat, long)
    start_date = parser.parse(start_date)
    start_date = start_date.replace(tzinfo=pytz.timezone(TimeUtils.Timezone))

    if plot.load_data_from_csv:
        # Load from CSV
        data = pd.read_csv(plot.data_from_csv, header=0)
        data.rename(columns={'timestamp': 'Time'}, inplace=True)
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)
        # Correct timestamp for timezone
        # Add timezone information without converting 
        data.index = data.index.map(lambda x: x.replace(tzinfo=pytz.timezone(TimeUtils.Timezone)))
        #data.index = pd.to_datetime(data.index) + pd.DateOffset(hours=get_timezone_offset(Timezone))
    else:
        # Load data from API
        for moisture in plot.device_and_sensor_ids_moisture:
            data_moisture.append(plot.load_data_api(moisture, "sensors", start_date))
        for temp in plot.device_and_sensor_ids_temp:
            data_temp.append(plot.load_data_api(temp, "sensors", start_date))
    
        # Save JSON data to one dataframe for further processing
        # Create first dataframe with first moisture sensor -> dfs have to be of same length, same timestamps
        data = pd.DataFrame(data_moisture[0])
        data.rename(columns={'time': 'Time'}, inplace=True)
        data.rename(columns={'value': plot.device_and_sensor_ids_moisture[0]}, inplace=True)
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)
        
        # Append the other cols and match timestamps
        for i in range(len(plot.device_and_sensor_ids_moisture)):
            if i==0:
                continue
            else:
                d = pd.DataFrame(data_moisture[i])
                d.rename(columns={'time': 'Time'}, inplace=True)
                d.rename(columns={'value': plot.device_and_sensor_ids_moisture[i]}, inplace=True)
                d['Time'] = pd.to_datetime(d['Time'])
                d.set_index('Time', inplace=True)
                data = pd.merge(data, d, left_index=True, right_index=True, how='outer')
                
        for i in range(len(plot.device_and_sensor_ids_temp)):
            d = pd.DataFrame(data_temp[i])
            d.rename(columns={'time': 'Time'}, inplace=True)
            d.rename(columns={'value': plot.device_and_sensor_ids_temp[i]}, inplace=True)
            d['Time'] = pd.to_datetime(d['Time'])
            d.set_index('Time', inplace=True)
            data = pd.merge(data, d, left_index=True, right_index=True, how='outer')

    # Rename index
    data.rename_axis('Timestamp', inplace=True)

    # Convert index
    data.index = pd.to_datetime(data.index, utc=True)
    data.index = data.index.tz_convert(TimeUtils.Timezone)
        
    # Impute gaps in data
    data = fill_gaps(data)

    print(data.index.dtype)
    
    # resample using timespan of long intervals => TODO: switch on
    # if ActualSamplingRate != StdSamplingRate:
    #     data_re = data.resample(str(ActualSamplingRate)+'T').mean()# median() ...was median
    #     data = data_re
    
    # create additional features
    data = create_features(data, plot)

    # Drop the raw values -> better without raw values-> overfitting
    data.drop(columns = plot.device_and_sensor_ids_moisture + plot.device_and_sensor_ids_temp, errors='ignore', inplace=True)
    
    # Convert datatype of cols to float64 -> otherwise json parse will parse negative values as object
    #data = data.apply(pd.to_numeric, errors='coerce')
    #data = data.astype(float)
    data = convert_cols(data)

    # Normalization => Is done before training TODO: also for NN?
    #data = normalize(data)

    print(data.iloc[0])
    
    print(data.head(0))
    
    return data#, data_plot, df_comb, cut_sub_dfs
