# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:54:49 2023

@author: felix markwordt
"""

#TODO:general names in csv export

from datetime import timedelta
import json
import subprocess
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
from sklearn.impute import KNNImputer
import urllib
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

# local
import main


# URL of API to retrive devices
#ApiUrl = "/" # Production mode
#ApiUrl = "http://localhost:8080/" # Debug mode
ApiUrl = "http://192.168.189.2/" # Debug mode on local gw
Token = None

# Initialize an empty dictionary to store the current config
Current_config = {}

# Extracted variables from Current_config
DeviceAndSensorIdsMoisture = []
DeviceAndSensorIdsTemp = []

# Std sample rate set by arduino code of microcontroller
StdSamplingRate = 10

# Actual sampling rate for machine learning purpose
ActualSamplingRate = 60

# Rolling mean window
RollingMeanWindowData = 15
RollingMeanWindowGrouped = 5

# Sampling rate of training dataset
Sample_rate = 60

# Forecast horizon TODO: add config or adjust automa
Forcast_horizon = 5 #days

# Created features that are dropped later
To_be_dropped = ['Timestamp', 'minute', 'grouped_soil', 'grouped_soil_temp', 'gradient']

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
Predictions = pd.DataFrame


def read_config():
    global Current_config
    global DeviceAndSensorIdsMoisture
    global DeviceAndSensorIdsTemp

    # Specify the path to the JSON file you want to read
    json_file_path = 'config/current_config.json'

    # Read the JSON data from the file
    with open(json_file_path, 'r') as json_file:
        Current_config = json.load(json_file)

    DeviceAndSensorIdsMoisture = Current_config["DeviceAndSensorIdsMoisture"]
    DeviceAndSensorIdsTemp = Current_config["DeviceAndSensorIdsTemp"]

# not ready
def get_token():
    global Token
    # Generate token to fetch data from another gateway
    if ApiUrl.startswith('/'):
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
        

# Load from CSV file
def load_data(path):
    # creating a data frame
    data = pd.read_csv("binned_removed.csv",header=0)
    print(data.head())
    return data

# Load from wazigate API
def load_data_api(sensor_name, from_timestamp):#, token):
    if ApiUrl.startswith('/'):
        print('There is no token needed, fetch data from local gateway.')
    elif Token != None:
        print('There is no token needed, already present.')
    # Get token, important for non localhost devices
    else:
        get_token()


    # Create URL for API call
    api_url = ApiUrl + "devices/" + sensor_name.split('/')[0] + "/sensors/" + sensor_name.split('/')[1] + "/values" + "?from=" + from_timestamp
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

# Impute the fuck out of it 
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

# Get offset to UTC time
def get_timezone_offset(timezone_str):
    timezone = pytz.timezone(timezone_str)
    utc_offset = timezone.utcoffset(None).total_seconds() / 3600.0

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

    # need to add one day, overlapping have to be cut off later => useless because data is not available to fetch via api
    last_date = data.index[-1]
    last_day_plus_one = last_date + timedelta(days=0)
    last_day_plus_one_str = last_day_plus_one.strftime("%Y-%m-%d")

    url = (
        f'https://archive-api.open-meteo.com/v1/archive'
        f'?latitude={Current_config["Gps_info"]["lattitude"]}'
        f'&longitude={Current_config["Gps_info"]["longitude"]}'
        f'&start_date={data.index[0].strftime("%Y-%m-%d")}'
        f'&end_date={last_day_plus_one_str}'
        f'&hourly=temperature_2m,relativehumidity_2m,rain,cloudcover,shortwave_radiation,windspeed_10m,winddirection_10m,soil_temperature_7_to_28cm,soil_moisture_0_to_7cm,et0_fao_evapotranspiration'
        f'&timezone=UTC'
    )
    dct = subprocess.check_output(['curl', url]).decode()
    dct = json.loads(dct)

    # Also convert it to a pandas dataframe
    data_w = (pd.DataFrame([dct['hourly']['temperature_2m'], 
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
            .assign(Timestamp = lambda x : pd.to_datetime(x.Timestamp, format='%Y-%m-%dT%H:%M', utc=True))
            .set_index(['Timestamp'])
            .dropna())
    
    # convert cols to float64
    data_w = convert_cols(data_w)

    return data_w

# Get weather forecast from open-meteo
def get_weather_forecast_api(start_date, end_date):
    # Define the API URL for weather forecast
    url = (
        f'https://api.open-meteo.com/v1/forecast'
        f'?latitude={Current_config["Gps_info"]["lattitude"]}'
        f'&longitude={Current_config["Gps_info"]["longitude"]}'
        f'&hourly=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,et0_fao_evapotranspiration,wind_speed_10m,wind_direction_10m,soil_temperature_18cm,soil_moisture_3_to_9cm,shortwave_radiation'
        f'&start_date={start_date.strftime("%Y-%m-%d")}'
        f'&end_date={end_date.strftime("%Y-%m-%d")}'
        f'&timezone=UTC'
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
            .assign(Timestamp = lambda x : pd.to_datetime(x.Timestamp, format='%Y-%m-%dT%H:%M', utc=True))
            .set_index(['Timestamp'])
            .dropna())
    
    # convert cols to float64
    data_forecast = convert_cols(data_forecast)

    return data_forecast

# convert to float64
def convert_cols(data):
    obj_dtype = 'object'
    
    for col in data.columns:
        if data[col].dtype == obj_dtype:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        print("This is :",col, "it has the following dtype: ", data[col].dtype)

    return data

# Resample data
def resample(d):
    d_resample = d.resample(str(Sample_rate)+'T').mean()
    return d_resample

# Volumetric water content
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

def add_volumetric_col_to_df(df, col_name):
    # Iterate over the rows of the dataframe and calculate volumetric water content
    soil_water_retention_tupel_list = [(float(dct['Soil tension']), float(dct['VWC'])) for dct in Current_config['Soil_water_retention_curve']]
    # Sort the soil-water retention curve points by soil tension in ascending order => TODO: NOT EFFICIENT HERE, move out
    sorted_curve = sorted(soil_water_retention_tupel_list, key=lambda x: x[0])
    for index, row in df.iterrows():
        soil_tension = row[col_name]
        if not pd.isna(soil_tension):
            # Calculate volumetric water content
            volumetric_water_content = soil_tension_to_volumetric_water_content(soil_tension, sorted_curve)
            # Assign the calculated value to a new column in the dataframe
            df.at[index, col_name + '_vol'] = volumetric_water_content

    return df

# TODO: more sophisticated approach needed: needs to learn from former => introduce model
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

    # Calculate and add volumetric water content
    data = add_volumetric_col_to_df(data, 'rolling_mean_grouped_soil')

    # Check gaps => TODO: not every col should interpolated (month?), some data is lost here
    data = check_gaps(data) 

    # Add calculated pump state
    f = data.rolling_mean_grouped_soil
    data['gradient'] = np.gradient(f) #TODO: check gradient calc -> pump state seems to be wrong
    data['pump_state'] = int(0)
    data = add_pump_state(data)
    
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
    # Load data from local wazigate api -> each sensor individually
    data_moisture = []
    data_temp = []
    for moisture in DeviceAndSensorIdsMoisture:
        data_moisture.append(load_data_api(moisture, Current_config['Start_date']))
    for temp in DeviceAndSensorIdsTemp:
        data_temp.append(load_data_api(temp, Current_config['Start_date']))
    
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

    # Convert datatype of cols to float64 -> otherwise json parse will parse negative values as object
    #data = data.apply(pd.to_numeric, errors='coerce')
    #data = data.astype(float)
    data = convert_cols(data)

    # Rename index
    data.rename_axis('Timestamp', inplace=True)
        
    # Impute gaps in data TODO: have a look for evenly distributed timestamps
    data = fill_gaps(data)
    
    # cut timezones from time string to convert to datetime64 => TODO: somehow it does not work like in the notebook.
    #data.index = data.index.tz_convert(None)
    #data.index = data.index.floor('S')
    #data.index = data.index.strftime('%Y-%m-%dT%H:%M:%S')
    #data.index = pd.to_datetime(data.index, format='%Y-%m-%dT%H:%M:%S.%fZ').floor('S')
    #data.index = data.index.floor('S')
    
    #data.index = pd.to_datetime(data.index, format='%Y-%m-%dT%H:%M:%S.%fZ')
    #data.index = data.index.map(lambda x: x.replace(microsecond=0))
    
    #data.index = data.index.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    #data.index = data.index.str[:-8]

    #data = data.set_index('Time')
    #data.index = data.index.tz_convert('UTC')
    #data.index = pd.to_datetime(data.index, format='%Y-%m-%dT%H:%M:%S')

    # Convert the index to UTC TODO: get an idea of which timezone we use for model: UTC, local?
    #current_timezone = pytz.timezone(get_timezone(Current_config["Gps_info"]["lattitude"], Current_config["Gps_info"]["longitude"]))
    
    #data.index = data.index.tz_convert(get_timezone(Current_config["Gps_info"]["lattitude"], Current_config["Gps_info"]["longitude"]))
    data.index = pd.to_datetime(data.index, utc=True)
    #data.index = data.index.tz_convert('UTC').tz_localize(None)
    #data.index = pd.to_datetime(data.index)
    # Set the 'Time' column as the index
    #data.set_index('Time', inplace=True)

    print(data.index.dtype)
    
    # resample using timespan of long intervals => TODO: switch on
    # if ActualSamplingRate != StdSamplingRate:
    #     data_re = data.resample(str(ActualSamplingRate)+'T').mean()# median() ...was median
    #     data = data_re
    
    # create additional features
    data = create_features(data)

    # Drop the raw values -> better without raw values-> overfitting
    data.drop(columns = DeviceAndSensorIdsMoisture + DeviceAndSensorIdsTemp, errors='ignore', inplace=True)
    
    # Normalization
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
    
    # Run compare_models function TODO: configure setup accordingly
    best_re = re_exp.compare_models(
        n_select = 19, 
        fold = 10, 
        sort = 'R2',
        verbose = 1,
        exclude=['lar']
        #include=['xgboost', 'llar'] #debug
    )

    return re_exp, best_re

# Save the best models
def save_models(exp, best):
    # save pipeline
    model_names = []
    for i in range(len(best)):
        exp.save_model(best[i], 'production_model_' + str(i))
        model_names.append('production_model_' + str(i))
        
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
    mpe = np.mean(diff / values1.values) * 100

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
    
    print("The best model is:", best_model_for_df.__module__)
    print("Maximum R2 Value:", max_value)
    print('This is the rest of the metrics: mae', results_for_one_df[max_index]['results'][0],  'rmse', results_for_one_df[max_index]['results'][1], 'mpe', results_for_one_df[max_index]['results'][2])
    print("Index of Maximum R2 Value:", max_index,"\n")

    return best_model_for_df

# Create future value testset for prediction
def create_future_values(data, best):
    # Create ranges and dataframe with timestamps 
    start = data['Timestamp'].iloc[0]
    print("start: ", start)
    train_end = data['Timestamp'].iloc[-1] #+ timedelta(minutes=sample_rate)
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
    new_data['rolling_mean_grouped_soil_vol'] = new_data['Soil_moisture_0-7']
    new_data['rolling_mean_grouped_soil_temp'] = new_data['Soil_temperature_7-28']

    # also include pump_state
    new_data = new_data.assign(pump_state=0)

    return new_data

# eval model against formerly split testset
def evaluate_against_testset(test, exp, best):
    print("This is the evaluation against the split testset")
    ground_truth = test['rolling_mean_grouped_soil']
    test_features = test.drop(['rolling_mean_grouped_soil'], axis=1)
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
    data.reset_index(drop=False, inplace=True)
    data.rename(columns={'index': 'Timestamp'}, inplace=True)

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
    model = re_exp.create_model(Model_mapping[best_model.__class__.__name__])

    return model, re_exp

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

    # TODO: check if that is right
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
        
# Tune hyperparameters
def tune_models(exp, best):
    # tune hyperparameters of dt
    tuned_best_models = []
    for i in range(len(best)):
        print("This is for the",i,"model:",best[i])
        tuned_best_models.append(exp[i].tune_model(best[i]))
        
    return best

# Generate prediction with best_model and impute generated future_values
def generate_predictions(best, exp, features):
    return exp.predict_model(best, data=features)

# Data Getter
def get_Data():
    if Data.empty:
        return False
    else:
        return Data

# Predictions Getter
def get_predictions():
    if Predictions.empty:
        return False
    else:
        return Predictions

# Mighty main fuction ;)
def main() -> int:
    global Predictions
    global Data

    # Check version of pycaret, should be >= 3.0
    print("Check version of pycaret:", pycaret.__version__, "should be >= 3.0")

    # Read user set config and save to Current_config(global)
    read_config()

    # Generate token if data is not present on GW
    get_token()
    
    # Data preparation pipeline, calls other subfunction to perform the task
    Data = prepare_data()  
    train, test = split_by_ratio(Data, 20)  

    # Start pycaret pipeline: setup, train models, save the best ones to best-array 
    #exp, best = create_and_compare_model_ts(cut_sub_dfs)
    exp, best = create_and_compare_model_reg(train)
    
    # Save the best models for further evaluation
    model_names = save_models(exp, best)
    
    # Load model from disk  
    try:
        best
    except NameError:
        best = load_models(model_names)

    # Evaluate with testset
    best_eval, results = evaluate_against_testset(test, exp, best)

    # Train best model on whole dataset (without slip test)
    best_model, best_exp = train_best(best_eval, Data)
    
    # Create future value set to feed new data to model
    future_features = create_future_values(Data, best_model)

    # Compare dataframes cols to be sure that they match, otherwise drop
    future_features = compare_train_predictions_cols(train, future_features)
    
    # Tune hyperparameters, see notebook
    #tuned_best = tune_models(exp, best)

    # Ensemble, Stacking & ... not implemented yet, see notebook

    # Create predictions to forecast values
    Predictions = generate_predictions(best_model, best_exp, future_features)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit