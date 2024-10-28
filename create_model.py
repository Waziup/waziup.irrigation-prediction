# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:54:49 2023

@author: felix markwordt
"""

#TODO: clear imports

from datetime import timedelta, datetime
import datetime
import json
import logging
import os
from dateutil import parser
import subprocess
from dotenv import load_dotenv
import pycaret 
#from pycaret.time_series import *
from pycaret.regression import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import missingno as msno 
import sys
import pytz
import requests
import urllib
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

# new imports nn
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import floatx
#from transformers import TFAutoModel, AutoTokenizer
from scikeras.wrappers import KerasRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#import kerastuner as kt
from keras_tuner import Hyperband, HyperParameters
import time
from keras.callbacks import Callback

# local
import main


# URL of API to retrive devices
ApiUrl = ""#"http://wazigate/" # Production mode

Token = None

# Initialize an empty dictionary to store the current config
Current_config = {}

# Current timezone
Timezone = ''

# Extracted variables from Current_config
DeviceAndSensorIdsMoisture = []
DeviceAndSensorIdsTemp = []
DeviceAndSensorIdsFlow = []

# Rolling mean window
RollingMeanWindowData = 15
RollingMeanWindowGrouped = 5

# Sampling rate of training dataset
Sample_rate = 60

# Forecast horizon TODO: add config or adjust automa
Forcast_horizon = 5 #days

# Created features that are dropped later
#To_be_dropped = ['Timestamp', 'minute', 'grouped_soil', 'grouped_soil_temp', 'gradient']
To_be_dropped = ['minute', 'Timestamp','gradient','grouped_soil','grouped_soil_temp','Winddirection','month','day_of_year','date']


# Mapping to identify models TODO: check for correctness
Model_mapping = {
    'LinearRegression': 'lr',
    'Lasso': 'lasso',
    'Ridge': 'ridge',
    'ElasticNet': 'en',
    'Lars': 'lar',
    'LassoLars': 'llar',
    'OrthogonalMatchingPursuit': 'omp',
    'BayesianRidge': 'br',
    'ARDRegression': 'ard',
    'PassiveAggressiveRegressor': 'par',
    'RANSACRegressor': 'ransac',
    'TheilSenRegressor': 'tr',
    'HuberRegressor': 'huber',
    'KernelRidge': 'kr',
    'SVR': 'svm',
    'KNeighborsRegressor': 'knn',
    'DecisionTreeRegressor': 'dt',
    'RandomForestRegressor': 'rf',
    'ExtraTreesRegressor': 'et',
    'AdaBoostRegressor': 'ada',
    'GradientBoostingRegressor': 'gbr',
    'MLPRegressor': 'mlp',
    'XGBRegressor': 'xgboost',
    'LGBMRegressor': 'lightgbm',
    'CatBoostRegressor': 'catboost',
    'DummyRegressor': 'dummy'
}

# predictions 
Data = pd.DataFrame
Data_w = pd.DataFrame
Predictions = pd.DataFrame
Threshold_timestamp = ""
Use_pycaret = True
Tuned_best = None
Best_exp = None

# Load data from CSV, is set if there is a file in the root directory
CSVFile = "binned_removed_new_for_app_ww.csv"
LoadDataFromCSV = False

# Load former irrigations from file "data/irrigations.json"
Load_irrigations_from_file = False

# Restrict time to training
class TimeLimitCallback(Callback):
    def __init__(self, max_time_seconds):
        super(TimeLimitCallback, self).__init__()
        self.max_time_seconds = max_time_seconds
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):  # Changed to on_epoch_end
        elapsed_time = time.time() - self.start_time
        print(f"Epoch {epoch}: Elapsed time {elapsed_time:.2f} seconds")
        if elapsed_time > self.max_time_seconds:
            self.model.stop_training = True
            print(f"Training stopped after {self.max_time_seconds} seconds")


def read_config():
    global DeviceAndSensorIdsMoisture
    global DeviceAndSensorIdsTemp
    global DeviceAndSensorIdsFlow
    global LoadDataFromCSV

    # Specify the path to the JSON file you want to read
    json_file_path = 'config/current_config.json'

    try:
        with open(CSVFile, "r") as file:
            # Perform operations on the file
            data = pd.read_csv(file, header=0)
            DeviceAndSensorIdsMoisture = []
            DeviceAndSensorIdsTemp = []
            DeviceAndSensorIdsFlow = []
            
            # create array with sensors strings
            for col in data.columns:
                if col.startswith("tension"):
                    DeviceAndSensorIdsMoisture.append(col)
                elif col.startswith("soil_temp"):
                    DeviceAndSensorIdsTemp.append(col)
            LoadDataFromCSV = True
            # This is not implemented: DeviceAndSensorIdsFlow = config["DeviceAndSensorIdsFlow"]
    except FileNotFoundError:
        # Read the JSON data from the file
        with open(json_file_path, 'r') as json_file:
            config = json.load(json_file)

        DeviceAndSensorIdsMoisture = config["DeviceAndSensorIdsMoisture"]
        DeviceAndSensorIdsTemp = config["DeviceAndSensorIdsTemp"]
        if "DeviceAndSensorIdsFlow" in config:
            DeviceAndSensorIdsFlow = config["DeviceAndSensorIdsFlow"]
    except Exception as e:
        print("An error occurred: No devices are set in settings, there is also no local config file.", e)

    return config

# not ready
def get_token():
    global Token
    # Generate token to fetch data from another gateway
    if ApiUrl.startswith('http://wazigate/'):
        print('There is no token needed, fetch data from local gateway.')
    # Get token, important for non localhost devices
    else:
        # curl -X POST "http://192.168.189.2/auth/token" -H "accept: application/json" -d "{\"username\": \"admin\", \"password\": \"loragateway\"}"
        token_url = ApiUrl + "auth/token"
        
        # Parse the URL
        parsed_token_url = urllib.parse.urlsplit(token_url)
        
        # Encode the query parameters
        encoded_query = urllib.parse.quote(parsed_token_url.query, safe='=&')
        
        # Reconstruct the URL with the encoded query
        encoded_url = urllib.parse.urlunsplit((parsed_token_url.scheme, 
                                            parsed_token_url.netloc, 
                                            parsed_token_url.path, 
                                            encoded_query, 
                                            parsed_token_url.fragment))
        
        # Define headers for the POST request
        headers = {
            'accept': 'application/json',
            #'Content-Type': 'application/json',  # Make sure to set Content-Type
        }
        
        # Define data for the GET request
        data = {
            'username': 'admin',
            'password': 'loragateway',
        }

        try:
            # Send a GET request to the API
            response = requests.post(encoded_url, headers=headers, json=data)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # The response content contains the data from the API
                Token = response.json()
            else:
                print("Request failed with status code:", response.status_code)
        except requests.exceptions.RequestException as e:
            # Handle request exceptions (e.g., connection errors)
            print("Request error:", e)
            
            return "", e #TODO: intruduce error handling!
        
def load_latest_data_api(sensor_name, type):#, token)
    global ApiUrl

    # Token
    load_dotenv()
    ApiUrl = os.getenv('API_URL')

    if ApiUrl.startswith('http://wazigate/'):
        print('There is no token needed, fetch data from local gateway.')
    elif Token != None:
        print('There is no token needed, already present.')
    # Get token, important for non localhost devices
    else:
        get_token()

    # Create URL for API call e.g.:curl -X GET "http://192.168.189.15/devices/669780aa68f319066a12444a/sensors/6697875968f319066a12444d/value" -H "accept: application/json"
    api_url = ApiUrl + "devices/" + sensor_name.split('/')[0] + "/" + type + "/" + sensor_name.split('/')[1] + "/value"
    # Parse the URL
    parsed_url = urllib.parse.urlsplit(api_url)

    # Encode the query parameters
    encoded_query = urllib.parse.quote(parsed_url.query, safe='=&')

    # Reconstruct the URL with the encoded query
    encoded_url = urllib.parse.urlunsplit((parsed_url.scheme, 
                                            parsed_url.netloc, 
                                            parsed_url.path, 
                                            encoded_query, 
                                            parsed_url.fragment))
    
    # Define headers for the GET request
    headers = {
        'Authorization': f'Bearer {Token}',
    }

    try:
        # Send a GET request to the API
        response = requests.get(encoded_url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # The response content contains the data from the API
            response_ok = response.json()
        else:
            print("Request failed with status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        # Handle request exceptions (e.g., connection errors)
        print("Request error:", e)
        return "", e #TODO: intruduce error handling!
    
    return response_ok

# Load from CSV file
def load_data(path):
    # creating a data frame
    data = pd.read_csv("binned_removed.csv",header=0)
    print(data.head())
    return data

# Load from wazigate API
def load_data_api(sensor_name, type, from_timestamp):#, token)
    global ApiUrl
    global Timezone
    global Current_config

    # Load config to obtain GPS coordinates
    Current_config = read_config()

    # Token
    load_dotenv()
    ApiUrl = os.getenv('API_URL')

    # Convert timestamp
    if not isinstance(from_timestamp, str):
        from_timestamp = from_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # Get timezone if no information avalable
    if Timezone == '':
        Timezone = get_timezone(Current_config["Gps_info"]["lattitude"], Current_config["Gps_info"]["longitude"])

    # Correct timestamp for timezone => TODO: here is an ERROR, timezone var is not available in first start
    from_timestamp = (datetime.datetime.strptime(from_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ") - timedelta(hours=get_timezone_offset(Timezone))).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    if ApiUrl.startswith('http://wazigate/'):
        print('There is no token needed, fetch data from local gateway.')
    elif Token != None:
        print('There is no token needed, already present.')
    # Get token, important for non localhost devices
    else:
        get_token()


    # Create URL for API call
    api_url = ApiUrl + "devices/" + sensor_name.split('/')[0] + "/" + type + "/" + sensor_name.split('/')[1] + "/values" + "?from=" + from_timestamp
    # Parse the URL
    parsed_url = urllib.parse.urlsplit(api_url)

    # Encode the query parameters
    encoded_query = urllib.parse.quote(parsed_url.query, safe='=&')

    # Reconstruct the URL with the encoded query
    encoded_url = urllib.parse.urlunsplit((parsed_url.scheme, 
                                            parsed_url.netloc, 
                                            parsed_url.path, 
                                            encoded_query, 
                                            parsed_url.fragment))
    
    # Define headers for the GET request
    headers = {
        'Authorization': f'Bearer {Token}',
    }

    try:
        # Send a GET request to the API
        response = requests.get(encoded_url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # The response content contains the data from the API
            response_ok = response.json()
        else:
            print("Request failed with status code:", response.status_code) 
    except requests.exceptions.RequestException as e:
        # Handle request exceptions (e.g., connection errors)
        print("Request error:", e)
        return "", e #TODO: intruduce error handling!
    
    return response_ok

# Resample and interpolate
def check_gaps(data):
    mask = data.isna().any()
    print(mask)
    if (mask.any()):
        data.dropna(inplace=True)
        data = resample(data)
        return data.interpolate(method='linear')
    else:
        return data
    
# Impute missing data & apply rolling mean (imputation & cleaning)
def fill_gaps(data):
    # Show if there are any missing values inside the data
    print("This is before: \n",data.isna().any())

    data = data.interpolate(method='linear')

    # Show if there are any missing values inside the data
    print("This is afterwards: \n",data.isna().any())

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

# Get offset to UTC time
def get_timezone_offset(timezone_str):
    timezone = pytz.timezone(timezone_str)
    current_time = datetime.datetime.now(tz=timezone)

    utc_offset = current_time.utcoffset().total_seconds() / 3600.0

    return utc_offset

# Get timezone string
def get_timezone(latitude_str, longitude_str):
    # Convert to floats
    latitude = float(latitude_str)
    longitude = float(longitude_str)

    geolocator = Nominatim(user_agent="timezone_finder")
    location = geolocator.reverse((latitude, longitude), language="en")
    
    # Extract timezone using timezonefinder
    timezone_finder = TimezoneFinder()
    timezone_str = timezone_finder.timezone_at(lng=longitude, lat=latitude)
    
    return timezone_str

# Get historical values from open-meteo TODO: include timezone service: https://stackoverflow.com/questions/16086962/how-to-get-a-time-zone-from-a-location-using-latitude-and-longitude-coordinates
def get_historical_weather_api(data):
    global Data_w

    # TODO: remove that one day overlapping data
    if Data_w.empty:
        first_day_minus_one_str = (data.index[0] - timedelta(days = 1)).strftime("%Y-%m-%d")
    else:
        first_day_minus_one_str = (Data_w.index[-1] - timedelta(days = 1)).strftime("%Y-%m-%d")

    #data_w_former_end = Data_w.index[-1]

    # need to add one day, overlapping have to be cut off later => useless because data is not available to fetch via api
    last_date = data.index[-1]
    last_date_str = last_date.strftime("%Y-%m-%d")

    lat = Current_config["Gps_info"]["lattitude"]
    long = Current_config["Gps_info"]["longitude"]

    url = (
        f'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={lat}'
        f'&longitude={long}'
        f'&start_date={first_day_minus_one_str}'
        f'&end_date={last_date_str}'
        f'&hourly=temperature_2m,relativehumidity_2m,rain,cloudcover,shortwave_radiation,windspeed_10m,winddirection_10m,soil_temperature_7_to_28cm,soil_moisture_0_to_7cm,et0_fao_evapotranspiration'
        f'&timezone={Timezone}'
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
    data_w_fetched.index = data_w_fetched.index.map(lambda x: x.replace(tzinfo=pytz.timezone(Timezone)))
    #data_w.index = pd.to_datetime(data_w.index) + pd.DateOffset(hours=get_timezone_offset(timezone))
    
    # convert cols to float64
    data_w_fetched = convert_cols(data_w_fetched)

    if Data_w.empty:
        # Set as global
        Data_w = data_w_fetched
    elif data_w_fetched.index[-1] == Data_w.index[-1]:
        return Data_w
    else:
        # Merge former historical weather data and fetched one into one dataframe
        Data_w = pd.concat([Data_w.loc[Data_w.index[0]:],
                            data_w_fetched.loc[Data_w.index[-1] + 
                                            timedelta(minutes=Sample_rate) 
                                            : data.index[-1]]
                                            ])

    return Data_w

# Get weather forecast from open-meteo
def get_weather_forecast_api(start_date, end_date):

    # Timezone and geo_location
    lat = Current_config["Gps_info"]["lattitude"]
    long = Current_config["Gps_info"]["longitude"]

    # Define the API URL for weather forecast
    url = (
        f'https://api.open-meteo.com/v1/forecast'
        f'?latitude={lat}'
        f'&longitude={long}'
        f'&hourly=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,et0_fao_evapotranspiration,wind_speed_10m,wind_direction_10m,soil_temperature_18cm,soil_moisture_3_to_9cm,shortwave_radiation'
        f'&start_date={start_date.strftime("%Y-%m-%d")}'
        f'&end_date={end_date.strftime("%Y-%m-%d")}'
        f'&timezone={Timezone}'
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
    data_forecast.index = data_forecast.index.map(lambda x: x.replace(tzinfo=pytz.timezone(Timezone)))
    #data_forecast.index = pd.to_datetime(data_forecast.index) + pd.DateOffset(hours=get_timezone_offset(timezone)) + pd.DateOffset(hours=1.0)
    
    # convert cols to float64
    data_forecast = convert_cols(data_forecast)

    return data_forecast

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

# Volumetric water content => Not called
def soil_tension_to_volumetric_water_content(soil_tension, soil_water_retention_curve):
    """
    Convert soil tension (kPa) to volumetric water content (fraction) using a given soil-water retention curve.
    
    Parameters:
        soil_tension (float): Soil tension value in kPa.
        soil_water_retention_curve (list of tuples): A list of tuples containing points on the soil-water retention curve.
            Each tuple contains two elements: (soil_tension_value, volumetric_water_content_value).
    
    Returns:
        float: Volumetric water content as a fraction (between 0 and 1).
    """

    # Find the two points on the curve that bound the given soil tension value
    lower_point, upper_point = None, None
    for tension, water_content in soil_water_retention_curve:
        if tension <= soil_tension:
            lower_point = (tension, water_content)
        else:
            upper_point = (tension, water_content)
            break
    
    # If the soil tension is lower than the first point on the curve, use the first point
    if lower_point is None:
        return soil_water_retention_curve[0][1]
    
    # If the soil tension is higher than the last point on the curve, use the last point
    if upper_point is None:
        return soil_water_retention_curve[-1][1]
    
    # Interpolate to find the volumetric water content at the given soil tension
    tension_diff = upper_point[0] - lower_point[0]
    water_content_diff = upper_point[1] - lower_point[1]
    interpolated_water_content = lower_point[1] + ((soil_tension - lower_point[0]) / tension_diff) * water_content_diff
    
    return interpolated_water_content
# VWC with log scale => Not called
def soil_tension_to_volumetric_water_content_log(soil_tension, soil_water_retention_curve):
    # Transform the tension and content values to logarithmic space
    tensions_log = np.log10([point[0] for point in soil_water_retention_curve])
    content_log = np.log10([point[1] for point in soil_water_retention_curve])

    # Interpolate in logarithmic space
    interpolated_content_log = np.interp(np.log10(soil_tension), tensions_log, content_log)

    # Transform back to linear space
    interpolated_content = 10 ** interpolated_content_log
    
    return interpolated_content

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

def add_volumetric_col_to_df(df, col_name):
    # Iterate over the rows of the dataframe and calculate volumetric water content
    soil_water_retention_tupel_list = [(float(dct['Soil tension']), float(dct['VWC'])) for dct in Current_config['Soil_water_retention_curve']]
    # Sort the soil-water retention curve points by soil tension in ascending order => TODO: NOT EFFICIENT HERE, move out
    sorted_curve = sorted(soil_water_retention_tupel_list, key=lambda x: x[0])
    for index, row in df.iterrows():
        soil_tension = row[col_name]
        # Calculate volumetric water content
        volumetric_water_content = soil_tension_to_volumetric_water_content_spline(soil_tension, sorted_curve)
        # Assign the calculated value to a new column in the dataframe
        df.at[index, col_name + '_vol'] = round(volumetric_water_content, 4)

    return df

def align_retention_curve_with_api(data, data_weather_api):
    soil_water_retention_tupel_list = [(float(dct['Soil tension']), float(dct['VWC'])) for dct in Current_config['Soil_water_retention_curve']]
    # Sort the soil-water retention curve points by soil tension in ascending order => TODO: NOT EFFICIENT HERE, move out
    sorted_curve = sorted(soil_water_retention_tupel_list, key=lambda x: x[0])
    # compare weatherdata from past against messured values more expressive:
    mean_recorded_sensor_values = data["rolling_mean_grouped_soil_vol"].mean()
    mean_open_meteo_past_vol = data_weather_api["Soil_moisture_0-7"].mean()
    factor = mean_open_meteo_past_vol / mean_recorded_sensor_values
    print("mean_recorded_sensor_values: ", mean_recorded_sensor_values, " mean_open_meteo_past_vol: ", mean_open_meteo_past_vol, " factor: ", factor)
    # Multiply the second column by factor
    modified_curve = [(x, y * factor) for x, y in sorted_curve]

    return modified_curve

# TODO: more sophisticated approach needed: needs to learn from former => introduce model, is now excluded when flow meter is installed
def add_pump_state(data):
    slope = float(Current_config["Slope"])
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

# Calculate time since pump was on (time since last irrigation)
# TODO: need to be included, but is not tested yet
def hours_since_pump_was_turned_on(df):    
    # Find the index of rows where pump state is 1
    pump_on_indices = df[df['pump_state'] == 1].index

    # Initialize a new column with NaN values
    df['rows_since_last_pump_on'] = float('nan')

    # Iterate over pump_on_indices and update the new column
    for i in range(len(pump_on_indices)):
        if i == 0:
            # If it's the first occurrence, update with the total rows
            df.loc[:pump_on_indices[i], 'rows_since_last_pump_on'] = len(df)
        else:
            # Update with the difference in rows since the last occurrence
            df.loc[pump_on_indices[i - 1] + 1:pump_on_indices[i], 'rows_since_last_pump_on'] = \
                (pump_on_indices[i] - pump_on_indices[i - 1] - pd.Timedelta(seconds=1)) / pd.Timedelta('1 hour')

    # Fill NaN values with 0 for rows where pump state is 1
    df['rows_since_last_pump_on'] = df['rows_since_last_pump_on'].fillna(0).astype(int)

    return df

# include the (on device saved) amount of irrigation given
def include_irrigation_amount(df):
    if Load_irrigations_from_file:
        # Load JSON data from file
        with open('data/irrigations.json', 'r') as file:
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
        if (len(DeviceAndSensorIdsFlow) >  0):
            # Load data and create the dataframe
            data_irrigation = load_data_api(DeviceAndSensorIdsFlow[0], "actuators", Current_config['Start_date'])
            df_irrigation = pd.DataFrame(data_irrigation)
            df_irrigation.rename(columns={'time': 'Timestamp'}, inplace=True)
            df_irrigation['Timestamp'] = pd.to_datetime(df_irrigation['Timestamp'])
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
            df_irrigation = df_irrigation.tz_convert(Timezone)

            # Merge the dataframes
            df = pd.merge(df, df_irrigation, left_index=True, right_index=True, how='outer', suffixes=('_main', '_irrigation'))

            # Fill missing irrigation_amount with 0
            df['irrigation_amount'] = df['irrigation_amount'].fillna(0)

    return df


# Augment the dataset creating new features
def create_features(data):
    # Create average cols
    data['grouped_soil'] = data[DeviceAndSensorIdsMoisture].mean(axis=1)
    data['grouped_soil_temp'] = data[DeviceAndSensorIdsTemp].mean(axis=1)
    
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

    # Get weather from weather meteo
    data_weather = get_historical_weather_api(data)

    # Resample weatherdata before merge => takes a long time
    data_weather = resample(data_weather)

    # historical weather data is not available for the latest two days, use forecast to account for that!
    data_weather_endtime = data_weather.index[-1]
    data_endtime = data.index[-1]

    # Get forecast for the ~last two days
    data_weather_recent_forecast = get_weather_forecast_api(data_weather_endtime, data_endtime)

    # Merge weather data to one dataframe
    data_weather_merged = pd.concat([data_weather.loc[data.index[0]:], 
                                     data_weather_recent_forecast.loc[data_weather_endtime + 
                                                                      timedelta(minutes=Sample_rate) 
                                                                      : data_endtime]
                                                                      ])

    # Merge data_weather_merged into data
    data = pd.merge(data, data_weather_merged, left_index=True, right_index=True, how='outer')

    # # Calculate and add volumetric water content => do not use this approach, does not yield better results
    # data = add_volumetric_col_to_df(data, 'rolling_mean_grouped_soil')
    # # align soil water retention curve with data from API => do not use this approach, does not yield better results
    # corrected_water_retention_curve = align_retention_curve_with_api(data, data_weather)
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
    # Skip the pump state if there is a flow meter where the artificial irrigation amount is messured
    if "DeviceAndSensorIdsFlow" in Current_config:
        data = include_irrigation_amount(data)
    else:
        data['pump_state'] = int(0)
        data = add_pump_state(data)
        

    # also add hours since last irrigation => TODO: check later, still an error, !!!!!questionable whether it is useful!!!!!
    #data = hours_since_pump_was_turned_on(data)
    
    return data

# Normalize the data in min - max approach from 0 - 1
def normalize(data):
    # feature scaling
    data.describe()
    
    # Min-Max Normalization
    df = data.drop(['Time','rolling_mean_grouped_soil', 'hour', 'minute', 'date', 'month'], axis=1)
    df_norm = (df-df.min())/(df.max()-df.min())
    df_norm = pd.concat([df_norm, data['Time'],data['hour'], data['minute'], data['date'], data['month'], data.rolling_mean_grouped_soil], 1)

    # bring back to order -> not important
    data = data[['Time', 'hour', 'minute', 'date', 'month', 'grouped_soil', 
                 'grouped_resistance', 'grouped_soil_temp', 'rolling_mean_grouped_soil', 
                 'rolling_mean_grouped_soil_temp', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b5; B2_solar_x2_03, Soil_tension', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b7; B2_solar_x2_03, Resistance', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b9; B2_solar_x2_03, Soil_temperature', 
                 '638de56a68f31919048f189e, 63932ce468f319085df375cb; B3_solar_x1_02, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ab; B4_solar_x1_03, Soil_tension', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ad; B4_solar_x1_03, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375af; B4_solar_x1_03, Soil_temperature'
                 ]]

    return df_norm


# Split dataset into train and test set by date
def split_data_by_date(data, split_date):
    return data[data['Time'] <= split_date].copy(), \
           data[data['Time'] >  split_date].copy()

# Split dataset into train and test set by ratio     
def split_by_ratio(data, test_size_percent):
    # Calculate the number of rows for the test set
    test_size = int(len(data) * (test_size_percent / 100))

    # Split the DataFrame
    train_set = data.iloc[:-test_size]
    test_set = data.iloc[-test_size:]

    return train_set, test_set

           
# Delete the ones that are non consecutive
def delete_nonconsecutive_rows(df, column_name, min_consecutive):
    arr = df[column_name].to_numpy()
    i = 0
    while i < len(arr) - 1:
        if arr[i+1] == arr[i] + 1:
            start_index = i
            while i < len(arr) - 1 and arr[i+1] == arr[i] + 1:
                i += 1
            end_index = i
            if end_index - start_index + 1 < min_consecutive:
                df = df.drop(range(start_index, end_index+1))
        i += 1
    return df


# Create visual representation of irrigation times
def highlight(data_plot, ax, neg_slope):
    for index, row in neg_slope.iterrows():
        current_index = int(row['index'])
        #print(current_index)
        ax.axvspan(current_index-10, current_index+10, facecolor='pink', edgecolor='none', alpha=.5)
    
    
# Create ranges to remove from data
def create_split_tuples(df, indices_to_omit):
    # Sort the indices in ascending order
    indices_to_omit = sorted(indices_to_omit)

    # Create a list of index ranges to remove
    ranges_to_remove = []
    start_idx = None
    for idx in indices_to_omit:
        if start_idx is None:
            start_idx = idx
        elif idx == start_idx + 1:
            start_idx = idx
        else:
            ranges_to_remove.append((int(start_idx), int(idx-1)))
            start_idx = idx
    if start_idx is not None:
        ranges_to_remove.append((int(start_idx), df.index.max()))
        
    print("Irrigation times to be omitted: ", ranges_to_remove)
    print("type: ", type(ranges_to_remove[0][0]))

    return ranges_to_remove


# Split data to split dataframes
def split_dataframe(df, index_ranges):
    dfs = []
    for i, (start, end) in enumerate(index_ranges):
        if index_ranges[i][1]-index_ranges[i][0] < 50:
            continue
        else:
            dfs.append(df.iloc[index_ranges[i][0]:index_ranges[i][1]])
            
    return dfs

# Main function to split dataframes
def split_sub_dfs(data, data_plot):
    # calculate slope of "rolling_mean_grouped_soil"
    f = data.rolling_mean_grouped_soil
    data['gradient'] = np.gradient(f)
    
    # create dataframe with downward slope
    neg_slope = pd.DataFrame({"index":[],
                             "rolling_mean_grouped_soil":[],
                             "gradient":[]}
                            )
    
    for index, row in data.iterrows():
        if row['gradient'] < -0.07: 
            #print(index, row['rolling_mean_grouped_soil'], row['gradient'])
            current_series = pd.Series([int(round(index,0)), row['rolling_mean_grouped_soil'],
                                        row['gradient']], index=['index', 
                                                                 'rolling_mean_grouped_soil', 
                                                                 'gradient']).to_frame().T
            neg_slope = neg_slope.append(current_series)
    
    # dont ask, I love pandas^^
    neg_slope_2 = pd.DataFrame({'index':[], 'rolling_mean_grouped_soil':[], 'gradient': []})
    neg_slope_2 = pd.concat([neg_slope_2, neg_slope], ignore_index=True)
    neg_slope = neg_slope_2
    
    # Delete the ones that are non consecutive
    neg_slope = delete_nonconsecutive_rows(neg_slope, 'index', 5)
    with open('output.txt', 'w') as f:
        print(neg_slope, file=f)
    
    # visualize areas with downward slope
    ax = data_plot.drop(['Time'], axis=1).plot()
    highlight(data_plot, ax, neg_slope)
    ax.figure.suptitle("""Irrigation times highlighted\n\n""", fontweight ="bold") 
    ax.figure.savefig('irrigation_times_temp.png', dpi=400)
    
    # convert to numpy array and to int
    neg_slope_indices = neg_slope['index'].to_numpy()
    neg_slope_indices = neg_slope_indices.astype(np.int32)
    
    # Create ranges to remove from data
    tuples_to_remove = create_split_tuples(data, neg_slope_indices)
    
    # Split data to split dataframes
    sub_dfs = split_dataframe(data, tuples_to_remove) 
    
    # print dataframes
    with open('output.txt', 'a') as f:
        print("There are ", len(sub_dfs), " dataframes now.", file=f)
        for sub_df in sub_dfs:
            print(sub_df.head(1), file=f)
            print(len(sub_df), file=f)
            sub_df.drop(['Time', 'hour', 'minute', 'date', 'month'], axis=1).plot()
            
    return data, sub_dfs


# Find global max and min in all sub_dfs and cut them from min to max 
# => train data will start with min and end with max 
def format_begin_end(sub_dfs):
    cut_sub_dfs = []
    for i in range(len(sub_dfs)):
        # reset "new" index
        sub_dfs[i] = sub_dfs[i].reset_index()
        
        # index
        global_min_index = sub_dfs[i]['rolling_mean_grouped_soil'].idxmin()
        global_max_index = sub_dfs[i]['rolling_mean_grouped_soil'].idxmax()
        # value
        global_min = sub_dfs[i]['rolling_mean_grouped_soil'].min()
        global_max = sub_dfs[i]['rolling_mean_grouped_soil'].max()
        
        print(i,": ",global_min_index, "value:", global_min, global_max_index, "value:", global_max, "length:", global_max_index-global_min_index)
        print(i,": ",global_min_index, "value:", sub_dfs[i]['rolling_mean_grouped_soil'][global_min_index], global_max_index, "value:", sub_dfs[i]['rolling_mean_grouped_soil'][global_max_index], "length:", global_max_index-global_min_index)
        
        
        cut_sub_dfs.append(sub_dfs[i].iloc[global_min_index:global_max_index])
    
    # Print them    
    for df in cut_sub_dfs:
        df.drop(['index','Time','hour', 'minute', 'date', 'month'],axis=1).plot()
        
    # Preserve old index and clean
    for i in range(len(cut_sub_dfs)):
        cut_sub_dfs[i] = cut_sub_dfs[i].reset_index()
        # clean dataframe
        cut_sub_dfs[i] = cut_sub_dfs[i].drop(['level_0'], axis=1)
        cut_sub_dfs[i] = cut_sub_dfs[i].rename(columns={'index':'orig_index'})
        
    # Print head of dfs
    i = 1
    with open('output.txt', 'a') as f:
        for df in cut_sub_dfs:
            print("Dataframe: ", i, file=f)
            i+=1
            print(df.iloc[:1], file=f)
        
    return cut_sub_dfs


# Combine them to one dataframe
def combine_dfs(cut_sub_dfs):
    # save all dataframes to one and rename
    df_comb = pd.DataFrame()
    for i in range(len(cut_sub_dfs)):
        # copy elements to one df_comb
        
        df_comb = pd.concat([df_comb, cut_sub_dfs[i]['orig_index']], axis=1)
        df_comb = pd.concat([df_comb, cut_sub_dfs[i]['rolling_mean_grouped_soil']], axis=1)
        
        df_comb = df_comb.rename(columns={'orig_index':'orig_index_' + str(i)})
        df_comb = df_comb.rename(columns={'rolling_mean_grouped_soil':'rolling_mean_grouped_soil_' + str(i)})
    
    # series are not of same length => visualized here!
    df_comb.drop(['orig_index_0', 'orig_index_1', 'orig_index_2', 'orig_index_3'],axis=1).plot()
    
    return df_comb


# Data preparation pipeline, calls other subfunction to perform the task
def prepare_data():
    global Timezone

    # Load data from local wazigate api -> each sensor individually
    data_moisture = []
    data_temp = []

    # start date is in UTC, but user expects it in his timezone
    start_date = Current_config['Start_date']
    lat = Current_config["Gps_info"]["lattitude"]
    long = Current_config["Gps_info"]["longitude"]
    Timezone = get_timezone(lat, long)
    start_date = parser.parse(start_date)
    start_date = start_date.replace(tzinfo=pytz.timezone(Timezone))

    if LoadDataFromCSV:
        # Load from CSV
        data = pd.read_csv(CSVFile, header=0)
        data.rename(columns={'timestamp': 'Time'}, inplace=True)
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)
        # Correct timestamp for timezone
        # Add timezone information without converting 
        data.index = data.index.map(lambda x: x.replace(tzinfo=pytz.timezone(Timezone)))
        #data.index = pd.to_datetime(data.index) + pd.DateOffset(hours=get_timezone_offset(Timezone))
    else:
        # Load data from API
        for moisture in DeviceAndSensorIdsMoisture:
            data_moisture.append(load_data_api(moisture, "sensors", start_date))
        for temp in DeviceAndSensorIdsTemp:
            data_temp.append(load_data_api(temp, "sensors", start_date))
    
        # Save JSON data to one dataframe for further processing
        # Create first dataframe with first moisture sensor -> dfs have to be of same length, same timestamps
        data = pd.DataFrame(data_moisture[0])
        data.rename(columns={'time': 'Time'}, inplace=True)
        data.rename(columns={'value': DeviceAndSensorIdsMoisture[0]}, inplace=True)
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)
        
        # Append the other cols and match timestamps
        for i in range(len(DeviceAndSensorIdsMoisture)):
            if i==0:
                continue
            else:
                d = pd.DataFrame(data_moisture[i])
                d.rename(columns={'time': 'Time'}, inplace=True)
                d.rename(columns={'value': DeviceAndSensorIdsMoisture[i]}, inplace=True)
                d['Time'] = pd.to_datetime(d['Time'])
                d.set_index('Time', inplace=True)
                data = pd.merge(data, d, left_index=True, right_index=True, how='outer')
                
        for i in range(len(DeviceAndSensorIdsTemp)):
            d = pd.DataFrame(data_temp[i])
            d.rename(columns={'time': 'Time'}, inplace=True)
            d.rename(columns={'value': DeviceAndSensorIdsTemp[i]}, inplace=True)
            d['Time'] = pd.to_datetime(d['Time'])
            d.set_index('Time', inplace=True)
            data = pd.merge(data, d, left_index=True, right_index=True, how='outer')

    # Rename index
    data.rename_axis('Timestamp', inplace=True)

    # Convert index
    data.index = pd.to_datetime(data.index, utc=True)
    data.index = data.index.tz_convert(Timezone)
        
    # Impute gaps in data
    data = fill_gaps(data)

    print(data.index.dtype)
    
    # resample using timespan of long intervals => TODO: switch on
    # if ActualSamplingRate != StdSamplingRate:
    #     data_re = data.resample(str(ActualSamplingRate)+'T').mean()# median() ...was median
    #     data = data_re
    
    # create additional features
    data = create_features(data)

    # Drop the raw values -> better without raw values-> overfitting
    data.drop(columns = DeviceAndSensorIdsMoisture + DeviceAndSensorIdsTemp, errors='ignore', inplace=True)
    
    # Convert datatype of cols to float64 -> otherwise json parse will parse negative values as object
    #data = data.apply(pd.to_numeric, errors='coerce')
    #data = data.astype(float)
    data = convert_cols(data)

    # Normalization => Is done before training TODO: also for NN?
    #data = normalize(data)

    print(data.iloc[0])
    
    print(data.head(0))
    
    return data#, data_plot, df_comb, cut_sub_dfs

# Create model in pycaret (time series)
def create_and_compare_model_ts(cut_sub_dfs):    
    # call setup of pycaret
    exp=[]
    for i in range(len(cut_sub_dfs)):
        exp.append(TSForecastingExperiment())
        
        # check the type of exp
        type(exp[i])
        
        # init setup on exp
        exp[i].setup(
            cut_sub_dfs[i], 
            target = 'rolling_mean_grouped_soil', 
            enforce_exogenous = False, 
            fold_strategy='sliding', 
            fh = fh, 
            session_id = 123, 
            fold = 3,
            ignore_features = ['Time', 'orig_index', 'gradient']
            #numeric_imputation_exogenous = 'mean'
        )
    
    with open('output.txt', 'a') as f:
        # check statistical tests on original data
        for i in range(len(cut_sub_dfs)):
            print("This is the", i, "part of the data:", file=f)
            print(exp[i].check_stats(), file=f)
            
    best = []
    for i in range(len(cut_sub_dfs)):
        print("This is for the", i, "part of the dataset: ")
        best.append(exp[i].compare_models(
            n_select = 5, 
            fold = 3, 
            sort = 'R2',
            verbose = 1, 
            #exclude=['lar_cds_dt','auto_arima','arima'],
            include=['lr_cds_dt', 'br_cds_dt', 'ridge_cds_dt', 
                     'huber_cds_dt', 'knn_cds_dt', 'catboost_cds_dt']
        ))
    
    with open('output.txt', 'a') as f:       
        for i in range(len(best)):
            print("\n The best model, for cut_sub_dfs[", i,"] is:", file=f)
            print(best[i][0], file=f)
    
    return exp, best

# Custom exception hook is need to debug in VSCode
def custom_exception_hook(exctype, value, traceback):
    # Your custom exception handling code here
    print(f"Exception Type: {exctype}\nValue: {value}")
    print(f"Trace:{traceback}")

# Create and compare models
def create_and_compare_model_reg(train):
    # Disable logging to a file
    #logging.basicConfig(filename=None, level=logging.INFO)
    logging.basicConfig(filename="logs.log", level=logging.INFO)

    # create regression exp
    re_exp = pycaret.regression.RegressionExperiment()

    # to rangeindex => do not use timestamps!
    train.reset_index(drop=False, inplace=True)
    train.rename(columns={'index': 'Timestamp'}, inplace=True)

    # old: to_be_dropped = ['minute', 'Timestamp','gradient','grouped_soil','grouped_resistance','grouped_soil_temp']

    # Run the following code with a custom exception hook
    sys.excepthook = custom_exception_hook

    # Run pycarets setup
    s = re_exp.setup(train, 
              target = 'rolling_mean_grouped_soil', 
              session_id = 123,
              verbose = True,
              ignore_features = To_be_dropped, 
              train_size = 0.8
    )
    
    # Print available models
    re_exp.models()
    
    # Run compare_models function TODO: configure setup accordingly
    best_re = re_exp.compare_models(
        n_select = 19, 
        fold = 10, 
        sort = 'R2',
        verbose = 1,
        #exclude=['lar']
        include=['xgboost', 'llar', 'catboost'] #debug
    )

    return re_exp, best_re

# Save the best models
def save_models(exp, best, path_to_save):
    # save pipeline
    model_names = []

    # type check for array -> convert
    if not isinstance(best, list):
        best = [best]

    for i in range(len(best)):
        full_path = path_to_save + str(i) + "_" + best[i].__module__
        exp.save_model(best[i], full_path)
        model_names.append(full_path)
        
    return model_names
        
# TODO: model_names will not work if it was not saved before        
# Load the best models        
def load_models(model_names):
    # load pipeline
    loaded_best_pipeline = []
    for i in range(model_names): # TODO: model_names will not work if it was not saved before
        loaded_best_pipeline.append(load_model(model_names[i]))
    
    return loaded_best_pipeline
    
# evaluate performance of prediction against test part of data
def evaluate_target_variable(series1, series2, model_name):
    # drop missing
    values1 = series1.dropna()
    values2 = series2.dropna()

    # calc max length
    min_length = min(len(values1), len(values2))
    values1 = values1[:min_length]
    values2 = values2[:min_length]

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


    # R2
    mean_series1 = np.mean(series1)
    ss_total = np.sum((series1 - mean_series1) ** 2)
    ss_residual = np.sum((series1 - series2) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

    # print metrics
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MPE: {mpe:.2f} %")
    print(f"R2 {r2_score:.2f}",'\n')
    #print("df1 len:",len(values1),"df2 len:",len(values2),'\n')

    metrics = { model_name : ['mae', 'rmse', 'mpe', 'r2'],
                'results'  : [f'{mae:.2f}', f'{rmse:.2f}', f'{mpe:.2f}', f'{r2_score:.2f}']
              }
    results = pd.DataFrame(metrics)
    results['results'] = results['results'].astype(float)

    return results

# Sort models in new dataframe according to performance on testset 
# -> TODO: for ensemble/stacking model return the 3 best models 
def evaluate_results_and_choose_best(results_for_one_df, best_for_one_df):
    # sort according to R2 score -> hence [3]
    max_r2_value = max((df['results'][3].max(), idx) for idx, df in enumerate(results_for_one_df))
    
    max_value = max_r2_value[0]
    max_index = max_r2_value[1]

    best_model_for_df = best_for_one_df[max_index]
    
    print("The best model after evaluation is:", best_model_for_df.__module__)
    print("Maximum R2 Value:", max_value)
    print('This is the rest of the metrics: mae', results_for_one_df[max_index]['results'][0],  'rmse', results_for_one_df[max_index]['results'][1], 'mpe', results_for_one_df[max_index]['results'][2])
    print("Index of Maximum R2 Value:", max_index,"\n")

    return best_model_for_df

# Create future value testset for prediction
def create_future_values(data):
    # Create ranges and dataframe with timestamps 
    start = data.index[0]
    print("start: ", start)
    train_end = data.index[-1] #+ timedelta(minutes=sample_rate)
    print("train end before adding: ", train_end)
    end = train_end+pd.Timedelta(days=Forcast_horizon)
    print("end after adding: ", end,"\n")
    all_dates = pd.date_range(start=train_end, end=end, freq=str(Sample_rate)+'T')    
    print("all dates: ", all_dates,"\n")

    # Fetch data from weather API
    data_weather_api_cut = get_weather_forecast_api(train_end, end)
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

    # also include pump_state, set to zero as we want to assume the behavior without watering
    if "DeviceAndSensorIdsFlow" in Current_config:
        new_data = new_data.assign(irrigation_amount=0)
    else:
        new_data = new_data.assign(pump_state=0)

    return new_data

# eval model against formerly split testset -> TODO: test has duplicates because of irrigation added formerly
def evaluate_against_testset(test, exp, best):
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

    # iterate best models 
    for i in range(len(best)):
        model_name = best[i].__module__
        print("Current model: " + model_name)

        # Create predictions
        predictions.append(exp.predict_model(best[i], data=test_features))

        # evaluate predictions against testset 
        results_for_model.append(evaluate_target_variable(ground_truth, predictions[i]['prediction_label'], model_name))
    
    # Sort models in new dataframe according to performance on testset
    best_eval = evaluate_results_and_choose_best(results_for_model, best)
    
    return best_eval, results_for_model

# train the best model after eval with fulln dataset
def train_best(best_model, data):
    # create regression exp
    re_exp = pycaret.regression.RegressionExperiment()

    # to rangeindex => do not use timestamps!
    data = data.reset_index(drop=True, inplace=False) #TODO: maybe problems here with live data from gw
    data = data.rename(columns={'index': 'Timestamp'}, inplace=False)

    # old: to_be_dropped = ['minute', 'Timestamp','gradient','grouped_soil','grouped_resistance','grouped_soil_temp']

    # Run the following code with a custom exception hook => maybe only relevant in VSCode
    sys.excepthook = custom_exception_hook

    # Run pycarets setup
    s = re_exp.setup(data, 
              target = 'rolling_mean_grouped_soil', 
              session_id = 123,
              verbose = True,
              ignore_features = To_be_dropped, 
              train_size = 0.8
              )
 
    # Create model 
    model = re_exp.create_model(Model_mapping[best_model.__class__.__name__])

    return model, re_exp

# Create NN
# TODO: Save the 
def create_nn_model(hp, shape):
    model = Sequential()

    # Tune the number of units in the first dense layer
    units_hidden1 = hp.Int('units_hidden1', min_value=32, max_value=256, step=32)
    model.add(Dense(units_hidden1, activation='relu', input_shape=shape))

    # Tune the number of units in the second dense layer
    units_hidden2 = hp.Int('units_hidden2', min_value=16, max_value=128, step=16)
    model.add(Dense(units_hidden2, activation='relu'))

    # Output layer
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])

    # Add a custom attribute to identify the model
    model.model_name = "nn_model"
    model.hp = hp
    model.shape = shape

    return model

# Create CNN
def create_cnn_model(hp, shape):
    model = Sequential()

    # Tune the number of filters in the first convolutional layer
    units_hidden1 = hp.Int('units_hidden1', min_value=32, max_value=128, step=16)
    model.add(Conv1D(filters=units_hidden1, kernel_size=3, activation='relu', input_shape=shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # Dense layer
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=32)
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])

    # Add a custom attribute to identify the model
    model.model_name = "cnn_model"
    model.hp = hp
    model.shape = shape

    return model

# Create RNN
def create_rnn_model(hp, shape):
    model = Sequential()

    # Tune the number of units in the LSTM layer
    units_hidden1 = hp.Int('units_hidden1', min_value=32, max_value=256, step=32)
    model.add(LSTM(units=units_hidden1, activation='relu', input_shape=shape))
    
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])

    # Add a custom attribute to identify the model
    model.model_name = "rnn_model"
    model.hp = hp
    model.shape = shape

    return model

# Create GRU
def create_gru_model(hp, shape):
    model = Sequential()

    # Tune the number of units in the GRU layer
    units_hidden1 = hp.Int('units_hidden1', min_value=32, max_value=256, step=32)
    model.add(GRU(units=units_hidden1, activation='relu', input_shape=shape))
    
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])

    # Add a custom attribute to identify the model
    model.model_name = "gru_model"
    model.hp = hp
    model.shape = shape

    return model

# Create bi-LSTM
def create_lstm_model(hp, shape):
    model = Sequential()

    # Tune the number of units in the Bidirectional LSTM layer
    units_hidden1 = hp.Int('units_hidden1', min_value=32, max_value=256, step=32)
    model.add(Bidirectional(LSTM(units=units_hidden1, activation='relu', input_shape=shape)))
    
    model.add(Dense(1))

    # Tune the optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae'])
    
    # Add a custom attribute to identify the model
    model.model_name = "lstm_model"
    model.hp = hp
    model.shape = shape

    return model

# prepare data for (conv) neural nets and other model architechtures
def prepare_data_for_cnn(data, target_variable):
    # to rangeindex => do not use timestamps!
    data = data.reset_index(drop=False, inplace=False)
    #data.rename(columns={'index': 'Timestamp'}, inplace=True)

    # Drop non important
    data_nn = data.drop(columns=To_be_dropped, axis=1, inplace=False) #dropping yields worse results (val_loss in training)

    # Split the dataset into features (X) and target variable (y)
    X = data_nn.drop(target_variable, axis=1)  # Assuming 'target_variable' is the target variable
    y = data_nn[target_variable]

    # Determine the split point (80% training, 20% testing)
    split_point = int(len(X) * 0.8)

    # Split the data into training and testing sets based on the split point
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # numerical_columns = X_test.select_dtypes(include=np.number).columns
    # X_test_numerical = X_test[numerical_columns]

    # # Scale the test features using the same scaler used for training data
    # X_test_scaled = scaler.transform(X_test_numerical)

    # Ensure input data is correctly reshaped for Conv1D
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_test_cnn = X_test_scaled[..., np.newaxis]

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler

# Models being trained on different architectures
def train_models(X_train, y_train, X_train_scaled, X_train_cnn):
    
    # Create an array to store all the models
    nn_models = []

    # # Create neural network # DEBUG

    # # Create a dummy HyperParameters object with fixed values
    # hp = HyperParameters()
    # hp.Fixed('units_hidden1', 64)
    # hp.Fixed('units_hidden2', 32)
    # hp.Fixed('optimizer', 'adam')

    # # Call the model function with the hp object and the input shape
    # input_shape = (X_train.shape[1],)
    # model_nn = create_nn_model(hp, shape=input_shape)

    # # Train the model
    # print('Will now train a Neural net (NN), with the following hyperparameters: ' + str(hp.values))
    # history_nn = model_nn.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
    # # Append for comparison
    # nn_models.append(model_nn)

    # # Create conv neural network

    # # Create a dummy HyperParameters object with fixed values
    # hp = HyperParameters()
    # hp.Fixed('units_hidden1', 64)
    # hp.Fixed('optimizer', 'adam')

    # # Call the model function with the hp object and the input shape
    # input_shape = (X_train_cnn.shape[1], 1)
    # model_cnn = create_cnn_model(hp, shape=input_shape)
    
    # # Train the model
    # print('Will now train a Convolutional neural net (CNN), with the following hyperparameters: ' + str(hp.values))
    # history_cnn = model_cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)
    # # Append for comparison
    # nn_models.append(model_cnn)

    # # Create RNN model

    # # Create a dummy HyperParameters object with fixed values
    # hp = HyperParameters()
    # hp.Fixed('units_hidden1', 50)  # Fixed units for RNN
    # hp.Fixed('optimizer', 'adam')  # Fixed optimizer

    # input_shape = (X_train.shape[1], 1)
    # model_rnn = create_rnn_model(hp, shape=input_shape)

    # # Train the model
    # print('Will now train a Recurrent neural network (RNN), with the following hyperparameters: ' + str(hp.values))
    # history_rnn = model_rnn.fit(X_train_scaled[..., np.newaxis], y_train, epochs=50, batch_size=32, validation_split=0.2)
    # # Append for comparison
    # nn_models.append(model_rnn)

    # # Create GRU model

    # # Create a dummy HyperParameters object with fixed values
    # hp = HyperParameters()
    # hp.Fixed('units_hidden1', 50)  # Fixed units for GRU
    # hp.Fixed('optimizer', 'adam')  # Fixed optimizer

    # input_shape = (X_train.shape[1], 1)
    # model_gru = create_gru_model(hp, shape=input_shape)
    # # Train the model
    # print('Will now train a Gated Recurrent Unit neural network (GRU), with the following hyperparameters: ' + str(hp.values))
    # history_gru = model_gru.fit(X_train_scaled[..., np.newaxis], y_train, epochs=50, batch_size=32, validation_split=0.2)
    # # Append for comparison
    # nn_models.append(model_gru)

    # LSTM architecture

    # Create a dummy HyperParameters object with fixed values
    hp = HyperParameters()
    hp.Fixed('units_hidden1', 50)  # Fixed units for LSTM
    hp.Fixed('optimizer', 'adam')  # Fixed optimizer

    input_shape = (X_train.shape[1], 1)
    model_lstm = create_lstm_model(hp, shape=input_shape)
    # Train the model
    print('Will now train a Long short-term memory neural network (LSTM), with the following hyperparameters: ' + str(hp.values))
    history_bilstm = model_lstm.fit(X_train_scaled[..., np.newaxis], y_train, epochs=50, batch_size=32, validation_split=0.2)
    # Append for comparison
    nn_models.append(model_lstm)

    # # Keras regressor and grid search -> TODO: Kerastuner does not work, package conflict, try optuna hyperopt
    # # Param grid to big -> not supported
    # param_grid = {
    #     'units_hidden1': [32, 64],  # Number of units in the first hidden layer
    #     'units_hidden2': [16, 32],  # Number of units in the second hidden layer
    #     'batch_size': [32, 64],  # Batch size for training
    #     'epochs': [50, 100],  # Number of epochs for training
    #     'optimizer': ['adam', 'rmsprop'],  # Optimizer algorithm
    #     'loss': ['mean_squared_error', 'binary_crossentropy'],  # Loss function
    # }
    # # Define parameter grid => works and improves the result!
    # param_grid = {
    #     'epochs': [50, 100],
    #     'batch_size': [32, 64],
    # }
    # # Wrap the Keras model in a scikit-learn estimator => TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!tune (bi)LSTM or gru instead!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # keras_estimator = KerasRegressor(build_fn=create_nn_model)
    # # Create GridSearchCV
    # grid = GridSearchCV(estimator=keras_estimator, param_grid=param_grid, cv=3)
    # # Fit the model
    # grid_result = grid.fit(X_train_cnn, y_train)
    # # Print best parameters
    # print("Best parameters:", grid_result.best_params_)
    # # Get the best model
    # best_model = grid_result.best_estimator_
    # # Append best model for comparision
    # nn_models.append(best_model)

    return nn_models

# keras Models available functions to train => add new models here!!!
Model_functions = {
    "nn_model" : create_nn_model,
    "cnn_model" : create_cnn_model,
    "rnn_model" : create_rnn_model,
    "gru_model" : create_gru_model,
    "lstm_model" : create_lstm_model
}

def model_builder_with_shape(model_func, shape):
    def build_model(hp):
        return model_func(hp, shape)
    return build_model


# Perform evaluation, create predictions on testset (X_test) and save models
def save_models_nn(nn_models, path_to_save):
    for i in range(len(nn_models)):     
        # # Make predictions, it is only a test, not saved -> change it
        # try:
        #     predictions = nn_models[i].predict(X_test_scaled[..., np.newaxis])
        # except Exception as e:
        #     print(f"Predict is not available for the model.")
            
        # Optionally, you can save the trained model for future use
        try:
            nn_models[i].save(path_to_save + str(i) + '_' + nn_models[i].model_name + '.h5')
        except Exception as e:
            print(f"Save is not available for the model. {nn_models[i].model_name} : {e}")

# Perform a evaluation of the models against the testset(X_test), slit before 
def evaluate_against_testset_nn(nn_models, X_test_scaled, y_test):
    predictions = []
    results_for_model = []
    for i in range(len(nn_models)):
    #     # Evaluate the model on the test set
    #     try:
    #         loss = nn_models[i].evaluate(X_test_scaled, y_test.to_numpy()[...,np.newaxis])
    #         print(f'Model: {i}  Test Loss: {loss}')
    #     except Exception as e:
    #         print(f"Evaluate is not available for the model. {e}")
        # Make predictions
        try:
            predictions.append(nn_models[i].predict(X_test_scaled[..., np.newaxis]))                
        except Exception as e:
            print(f"Predict is not available for the model.")

        # evaluate predictions against testset 
        results_for_model.append(evaluate_target_variable(y_test.reset_index(drop=True), pd.Series(predictions[i].flatten()), ""))
    
    # Sort models in new dataframe according to performance on testset
    best_eval = evaluate_results_and_choose_best(results_for_model, nn_models)

    return best_eval, results_for_model

# Train the best model on the full dataset TODO: metrics bad check again!
def train_best_nn(best_eval, data, scaler):
    # to rangeindex => do not use timestamps!
    data = data.reset_index(drop=False, inplace=False)
    data = data.rename(columns={'index': 'Timestamp'}, inplace=False)

    # Drop non important
    data_nn = data.drop(To_be_dropped, axis=1, inplace=False) #dropping yields worse results (val_loss in training)

    # Split the dataset into features (X) and target variable (y)
    X = data_nn.drop('rolling_mean_grouped_soil', axis=1)  # Assuming 'soil_tension' is the target variable
    y = data_nn['rolling_mean_grouped_soil']

    # Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler.transform(X)

    # numerical_columns = X.select_dtypes(include=np.number).columns
    # X_numerical = X[numerical_columns]

    # # Scale the test features using the same scaler used for training data
    # X_scaled = scaler.transform(X_numerical)

    # Ensure input data is correctly reshaped for Conv1D
    X_cnn = X_scaled[..., np.newaxis]

    function_name = best_eval.model_name

    if function_name in Model_functions: 
        # Create best model TODO: check, bad metrics, compared to other models
        model = Model_functions[function_name](best_eval.hp, best_eval.shape)
        print(f"Will train the '{model.model_name}' as best model for neural nets.")
        # Train the model
        history_rnn = model.fit(X_cnn, y, epochs=50, batch_size=32, validation_split=0.2)
    else:
        print(f"Function '{function_name}' not found")

    return model, X_scaled, y
    
# Create testset, not seen during training=>should be fur
def prepare_future_values(scaler, new_data, X_train_c):
    # Filter the list to include only columns that are present in the DataFrame
    columns_to_drop = [col for col in To_be_dropped if col in new_data.columns]
    # drop not needed features
    new_data_nn = new_data.drop(columns_to_drop, axis=1, inplace=False)
    # Align columns of df1 to match df2 -> in most cases not needed
    new_data_aligned = new_data_nn.reindex(columns=X_train_c)

    # scale testset
    Z = new_data_aligned

    Z_scaled = scaler.transform(Z)

    # numerical_columns = Z.select_dtypes(include=np.number).columns
    # Z_numerical = Z[numerical_columns]

    # # Scale the test features using the SAME scaler used for training data
    # Z_scaled = scaler.transform(Z_numerical)

    Z_cnn = Z_scaled[..., np.newaxis]

    return Z, Z_scaled, Z_cnn

# Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors.
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# Quantile Loss can be defined as a custom loss function, which can be trained to minimize Quantile Loss. If the set percentile values are close to 0 or 1, the training results do not follow the trend of the training data and are relatively flat
def quantile_loss(y_true, y_pred, q):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))

# Calculate mse, rmse, mae, mpe, r2, quantile_loss, SMAPE
def evaluate_target_variable_nd(series1, series2, quantiles=[0.2, 0.4, 0.6, 0.8]):
    # Step 1: Compute the differences between the two series
    differences = series1 - series2
    
    # Step 2: Compute MSE, RMSE, MPE, and R2 score based on the differences
    mse = mean_squared_error(series1, series2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(series1, series2)
    mpe = np.mean(differences / series1) * 100
    r2 = r2_score(series1, series2)
    smape_score = smape(series1, series2)
    quantile_losses = [quantile_loss(series1, series2, q) for q in quantiles]
    # Compute the average quantile loss
    avg_quantile_loss = np.mean(quantile_losses)
    
    # Print the results
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Percentage Error (MPE): {mpe:.2f}%")
    print(f"R-squared (R2) Score: {r2:.2f}")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_score:.2f}%")
    for q, loss in zip(quantiles, quantile_losses):
        print(f"Quantile Loss (q={q}): {loss:.2f}")
    print(f"Average Quantile Loss: {avg_quantile_loss:.2f}")

    return mse, rmse, mae, mpe, r2, smape_score, quantile_losses, avg_quantile_loss

# TODO:wire this
def compare_models_on_test(nn_models, ZZ, Z_cnn):
    best_r2 = -1000
    best_model_index  = -1
    for i in range(len(nn_models)):
        print("This is the " + str(i+1) + ". model in the pipeline")
        # print summary
        if isinstance(nn_models[i], Sequential) or isinstance(nn_models[i], Model):
            nn_models[i].summary()
        else:
            print('For this model there is no summary\n')
            
        try:
            # Make predictions
            predictions = nn_models[i].predict(Z_cnn)
        
            # Evaluate model performance
            print("Metrics:", predictions.shape)
            evaluation_result = evaluate_target_variable_nd(ZZ.values, predictions.reshape(predictions.shape[0], predictions.shape[1]))
            
            # Check if evaluation result is not None
            if evaluation_result is not None:
                mse, rmse, mae, mpe, r2, smape_score, quantile_losses, avg_quantile_loss = evaluation_result
                # select best
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_index = i
            else:
                print("Evaluation result is None. Skipping metrics printing.")

            print("\n*****************************************************************")
        except Exception as e:
            print(f"Is not available for the model. {e}")
            
    if best_model_index != -1:
        print("The best model after evaluation on unseen data during training is:", best_model_index)
        best_model = nn_models[best_model_index]
    else:
        print("No model met the criteria for selection.")   

    return best_model_index

def eval_approach(results, results_nn, metrics):
    if metrics == 'mae':
        best_result = float('inf')
        for i in range(len(results)):
            current = results[i]['results'][0]
            if current < best_result:
                index = i
                best_result = current
                use_pycaret = True
        for i in range(len(results_nn)):
            current = results_nn[i]['results'][0]
            if current < best_result:
                index = i
                best_result = current
                use_pycaret = False
    else:
        print('Unknown metircs specified in eval_approach.')

    return index, use_pycaret


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

def analyze_performance_old(exp, best):
    # plot forecast
    for i in range(len(best)):
        print("This is for model: ",i)
        exp.plot_model(best[i], plot = 'forecast', save = True)
        #.save('Plot_in_testset_'+str(i)+'.png', format='png')
        
        print("After testset: For the dataset:",i)
        exp[i].plot_model(best[i], plot = 'forecast', data_kwargs = {'fh' : 500}, save = True)
        #before.save("Plot_after_testset_"+str(i)+".png", format='png')

# Tune hyperparameters of one models
def tune_model(exp, best):
    try:        
        #print(f"Tuning grid for {best}: {exp.get_tuning_grid(best)}")
        best = exp.tune_model(best, choose_better = True)
        return best
    except Exception as e:
        print(f"There was an error tuning the model. {e}")
        return best
    
# Tune hyperparameters of several models
def tune_models(exp, best):
    # tune hyperparameters of dt
    tuned_best_models = []
    for i in range(len(best)):
        print("This is for the",i,"model:",best[i])
        tuned_best_models.append(exp.tune_model(best[i], choose_better = True))
        
    return tuned_best_models


def tune_model_nn(X_train_scaled, y_train, best_model_nn):
    print("Tuning best NN model (", best_model_nn.model_name, ") after evaluation.")

    # quick workaround for adjusting train shape for tuning a lstm
    if best_model_nn.model_name == 'lstm_model': # TODO: check
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

    hp = HyperParameters()

    tuner = Hyperband(
        model_builder_with_shape(Model_functions[best_model_nn.model_name], best_model_nn.shape),
        objective='val_mae',
        max_epochs=10,              # Tune epochs between 10 and 100 # TODO: was 100
        factor=4,                   # Reduces the number of epochs for each successive run, Defaults to 3.
        hyperband_iterations=1,     # Limits the number full hyperband runs
        directory='hyperband_dir',
        project_name='hyperband_' + best_model_nn.model_name,
        overwrite=True
    )

    # Set the max time in seconds
    max_time_seconds = 60 #3600
    time_limit_callback = TimeLimitCallback(max_time_seconds)

    tuner.search(X_train_scaled, 
                    y_train, 
                    epochs=hp.Int('epochs', 5, 10), #10 100
                    batch_size=32, 
                    validation_split=0.2,
                    callbacks=[time_limit_callback]  # Add the time limit callback here
    )

    # Print the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # Rebuild the model using the best hyperparameters
    final_model = Model_functions[best_model_nn.model_name](best_hps, best_model_nn.shape)

    # Train the final model with the best hyperparameters on the full training data
    final_model.fit(X_train_scaled, y_train, epochs=best_hps.values.get('tuner/epochs', 50), batch_size=32)#, validation_split=0.2)

    # Compare with formerly best model -> since it is trained 
    #final_model = evaluate_against_testset_nn(best_model_nn, X_train_scaled, y_train)

    return final_model

# Generate prediction with best_model and impute generated future_values
def generate_predictions(best, exp, features):
    # Generate predictions
    predictions = exp.predict_model(best, data=features)

    # Clip neg predictions to zero
    predictions.loc[predictions['prediction_label'] < 0, 'prediction_label'] = 0

    return predictions

def generate_predictions_nn(best_model_nn, features, start, end):
    # quick workaround for adjusting train shape for tuning a lstm
    if best_model_nn.model_name == 'lstm_model': # TODO: check
        features = features.reshape((features.shape[0], 1, features.shape[1]))

    # Generate predictions
    predictions = best_model_nn.predict(features[..., np.newaxis])
    
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

# Calculates the time when threshold will be meet, according to predictions
def calc_threshold(Predictions):
    threshold = Current_config["Threshold"]

    # calculate next occurance
    for i in range(len(Predictions)):
        if Predictions['prediction_label'][i] > threshold:
            print("Threshold will be reached on", Predictions.index[i], "With a value of:", Predictions['prediction_label'][i])
            return Predictions.index[i]

    return ""

# Data Getter
def get_Data():
    if Data.empty:
        return False
    else:
        return Data.drop([item for item in To_be_dropped if item != "Timestamp"], axis=1) # This is needed to prevent the timestamp is being omitted

# Predictions Getter
def get_predictions():
    if Predictions.empty:
        return False
    else:
        return Predictions
    
# threshold timestamp Getter
def get_threshold_timestamp():
    if not Threshold_timestamp:
        return False
    else:
        return Threshold_timestamp

def predict_with_updated_data():
    global Predictions
    global Best_exp
    global Threshold_timestamp

    train, test, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler = data_pipeline()
    # Create future value set to feed new data to model
    future_features = create_future_values(Data)
    # Compare dataframes cols to be sure that they match, otherwise drop
    future_features = compare_train_predictions_cols(train, future_features)
    # NN
    if not Use_pycaret:
        Z, Z_scaled, Z_cnn = prepare_future_values(scaler, future_features, X_train.columns)
        Predictions = generate_predictions_nn(Tuned_best, Z_scaled, future_features.index[0], future_features.index[-1])
    else:
        Predictions = generate_predictions(Tuned_best, Best_exp, future_features)
    
    # Cut passed time from predictions
    Predictions = Predictions.loc[pd.Timestamp((datetime.datetime.now()).replace(microsecond=0, second=0, minute=0)).tz_localize(Timezone):]    
    
    # Calculate when threshold will be meet
    Threshold_timestamp = calc_threshold(Predictions)

    # Add volumetric water content
    Predictions = add_volumetric_col_to_df(Predictions, "prediction_label")

    # Return last accumulated reading and threshold timestamp currentSoilTension, threshold_timestamp, predictions 
    return Data['rolling_mean_grouped_soil'][-1], Threshold_timestamp, Predictions


def data_pipeline():
    global Data
    global ApiUrl
    global Current_config

    # Check version of pycaret, should be >= 3.0
    print("Check version of pycaret:", pycaret.__version__, "should be >= 3.0")

    load_dotenv()
    ApiUrl = os.getenv("API_URL")

    # Read user set config and save to Current_config(global)
    Current_config = read_config()

    # Generate token if data is not present on GW
    get_token()
    
    # Data preparation pipeline, calls other subfunction to perform the task
    # Classical regression
    Data = prepare_data()

    # Search for gaps in data again (quick fix) => tackle problem with latest data "nan", in case of irrigations saved
    if Data.isna().any().any():
        #Data.drop(Data.index[-1], inplace=True)
        Data.dropna(inplace=True)
      
    train, test = split_by_ratio(Data, 20) # here a split is done to rule out the models that are overfitting
    # NN
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler = prepare_data_for_cnn(Data, 'rolling_mean_grouped_soil')

    return train, test, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler

# Mighty main fuction ;)
def main() -> int:
    global Predictions
    global Threshold_timestamp
    global Use_pycaret
    global Tuned_best
    global Best_exp

    # Data preparation: get config, fetch, align, clean, sample....
    train, test, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X_train_cnn, X_test_cnn, scaler = data_pipeline()

    # Start training pipeline: setup, train models the best ones to best-array
    # Classical regression
    #exp, best = create_and_compare_model_ts(cut_sub_dfs)
    exp, best = create_and_compare_model_reg(train)
    # NN
    nn_models = train_models(X_train, y_train, X_train_scaled, X_train_cnn)
    
    # Save the best models for further evaluation
    # Classical regression:
    model_names = save_models(exp, best, 'models/intermediate_models/pycaret/soil_tension_prediction_pycaret_model_')
    # NN: (print eval(on X_test) and save to disk)
    save_models_nn(nn_models, 'models/intermediate_models/nn/soil_tension_prediction_nn_model_')
    
    # Load regression model from disk, if there was a magical error => TODO: useless, because it would stop before, surround more with try except
    try:
        best
    except NameError:
        best = load_models(model_names)

    # Evaluate with testset
    # Classical regression
    best_eval, results = evaluate_against_testset(test, exp, best)
    # NN
    best_eval_nn, results_nn = evaluate_against_testset_nn(nn_models, X_test_scaled, y_test)

    # Decide which approach is best
    index, Use_pycaret = eval_approach(results, results_nn, 'mae')

    # TODO: Debug mode
    Use_pycaret = True

    # Train best model on whole dataset (without skipping "test-set")
    if Use_pycaret:
        # Classical regression
        best_model, Best_exp = train_best(best_eval, Data)
    else:
        # NN -> TODO: eval properly -> still error in r2 calc, fallback to mae
        best_model_nn, X_scaled, y = train_best_nn(best_eval_nn, Data, scaler)

    
    # Create future value set to feed new data to model
    future_features = create_future_values(Data)
    # Compare dataframes cols to be sure that they match, otherwise drop
    future_features = compare_train_predictions_cols(train, future_features)
    # NN
    if not Use_pycaret:
        Z, Z_scaled, Z_cnn = prepare_future_values(scaler, future_features, X_train.columns)


    if Use_pycaret:
        # Before tuning
        #best_model_before_tuning = best_exp.compare_models()
        
        # Tune hyperparameters of the 3 best models, see notebook
        try:
            Tuned_best = tune_model(Best_exp, best_model)
        except Exception as e:
            print(f"Error during tuning: {e}, using the original model.")
            Tuned_best = best_model  # Keep original model if tuning fails

        
        # After tuning
        #best_model_after_tuning = best_exp.compare_models()
        
        # Manually compare metrics
        #metrics_before_tuning = best_exp.get_metrics(model=best_model_before_tuning)
        #metrics_after_tuning = best_exp.get_metrics(model=best_model_after_tuning)

        # Compare relevant metrics
        # compare_df = pd.DataFrame({
        #     'Metric': metrics_before_tuning['Metric'],
        #     'Before_Tuning': metrics_before_tuning['Value'],
        #     'After_Tuning': metrics_after_tuning['Value']
        # })

        # print(compare_df)
        
        # Ensemble, Stacking & ... not implemented yet, see notebook

        # Save best pycaret model
        model_names = save_models(exp, [Tuned_best], 'models/best_models/pycaret/best_soil_tension_prediction_pycaret_model_')

        # Create predictions to forecast values
        # Classical regression
        Predictions = generate_predictions(Tuned_best, Best_exp, future_features)
    else:
        # Tune best model
        Tuned_best = tune_model_nn(X_scaled, y, best_model_nn)

        # Save best model
        save_models_nn([best_model_nn], 'models/best_models/nn/best_soil_tension_prediction_nn_model_')

        # Generate predictions with NN model
        Predictions = generate_predictions_nn(Tuned_best, Z_scaled, future_features.index[0], future_features.index[-1])

    # Cut passed time from predictions
    Predictions = Predictions.loc[pd.Timestamp((datetime.datetime.now()).replace(microsecond=0, second=0, minute=0)).tz_localize(Timezone):]    
    
    # Calculate when threshold will be meet
    Threshold_timestamp = calc_threshold(Predictions)

    # Add volumetric water content
    Predictions = add_volumetric_col_to_df(Predictions, "prediction_label")

    # Return last accumulated reading and threshold timestamp
    return Data['rolling_mean_grouped_soil'][-1], Threshold_timestamp, Predictions

if __name__ == '__main__':
    sys.exit(main())