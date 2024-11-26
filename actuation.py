import json
import os
import sys
from datetime import datetime, timedelta
import pytz

from dotenv import load_dotenv
import pandas as pd
import requests

import create_model


# Globals
# Timespan of hours 
TimeSpanOverThreshold = 12
OverThresholdAllowed = 1.2
Last_irrigation = ''

# Find global max and min => not used any more
def get_max_min(df, target_col='prediction_label'):
    # reset "new" index
    df = df.reset_index(inplace=False)
    
    # index
    global_min_index = df[target_col].idxmin()
    global_max_index = df[target_col].idxmax()
    # value
    global_min = df[target_col].min()
    global_max = df[target_col].max()
    
    print(global_min_index, "value:", global_min, global_max_index, "value:", global_max, "length:", global_max_index-global_min_index)
    print(global_min_index, "value:", df[target_col][global_min_index], global_max_index, "value:", df[target_col][global_max_index], "length:", global_max_index-global_min_index)

    return global_min_index, global_max_index, global_min, global_max


# Function to find next lower and higher value occurrence
def find_next_occurrences(df, column, threshold):
    timezone = create_model.Timezone

    # Start @current time
    # timezone = create_model.get_timezone(Current_config["Gps_info"]["lattitude"], Current_config["Gps_info"]["longitude"])
    idx = pd.Timestamp(datetime.now().replace(microsecond=0)).tz_localize(timezone) #TODO: timezone is missing here, replace with timezone, uncomment above

    # Filter the DataFrame to include only rows with indices greater than or equal to 'idx'
    filtered_df = df[df.index >= idx]
    filtered_df = df[df.index <= idx + timedelta(hours=TimeSpanOverThreshold)]

    # Further filter the DataFrame to include only rows where the specified column's value is less than the 'threshold'
    filtered_lower = filtered_df[filtered_df[column] < threshold]

    # Convert the filtered DataFrame's index to a list
    next_lower_idx = filtered_lower.index.tolist()
    
    # Take first occurance from list
    if next_lower_idx:
        next_lower_idx = next_lower_idx[0]
    else:
        next_lower_idx = None

    # Find the next occurrence of a value higher than the threshold after the next lower index
    if next_lower_idx is not None:
        # Filter the DataFrame to include only rows with indices greater than or equal to 'next_lower_idx'
        filtered_df_higher = df[df.index >= next_lower_idx]

        # Further filter the DataFrame to include only rows where the specified column's value is greater than the 'threshold'
        filtered_higher = filtered_df_higher[filtered_df_higher[column] > threshold]

        # Convert the filtered DataFrame's index to a list
        next_higher_idx = filtered_higher.index.tolist()

        # Take first occurance from list
        if next_higher_idx:
            next_higher_idx = next_higher_idx[0]
        else:
            next_higher_idx = None
    # Consequently if there is no occurance of lower, just take first one from inputdata
    else:
        next_higher_idx = df.index[0]

    return next_lower_idx, next_higher_idx 

# Function to read existing data from the JSON file
def read_data_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            return json.load(json_file)
    else:
        return {"irrigations": []}

# Function to save data to the JSON file
def save_data_to_file(filename, data):
    if not os.path.exists(filename):
        print(f"{filename} does not exist, creating a new one.")
        
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Function to add a new record
def add_record(data, timestamp, amount):
    record = {
        "timestamp": timestamp,
        "amount": amount
    }
    data["irrigations"].append(record)

    return data

# Function to round to the nearest 10 minutes
def round_to_nearest_10_minutes(dt):
    # Truncate milliseconds and seconds
    dt = dt.replace(microsecond=0, second=0)
    
    # Calculate the number of minutes to add to round to the nearest 10 minutes
    add_minutes = 10 - (dt.minute % 10) if dt.minute % 10 >= 5 else -(dt.minute % 10)
    dt += timedelta(minutes=add_minutes)
    
    return dt

# to json file -> not needed because can just ask api, more consistant state
def save_irrigation_time(amount):
    # Load from file
    filename = 'data/irrigations.json'
    data = read_data_from_file(filename)

    # obtain timezone
    timezone = pytz.timezone(create_model.Timezone)
    # add to current timestamp without converting it
    now = timezone.localize(datetime.now())

    # Round to the nearest 10 minutes
    rounded_tz = round_to_nearest_10_minutes(now)

    # Add new records
    data = add_record(data, str(rounded_tz), amount)

    # Save updated data back to the JSON file
    save_data_to_file(filename, data)

    print("Irrigation time has been saved to: ", filename)

    return 0

# Load from wazigate API
def irrigate_amount(amount):
    # Example API call: 
    # curl -X POST "http://192.168.189.2/devices/6645c4d468f31971148f2ab1/actuators/6673fcb568f31971148ff5f7/value"
    # -H "accept: */*" -H "Content-Type: application/json" -d "7.2"
    global ApiUrl
    global Timezone
    global Last_irrigation

    # Name of flow meter sensor to initiate irrigation
    flow_meter_name = create_model.DeviceAndSensorIdsFlow[0]

    # API URL
    load_dotenv()
    ApiUrl = create_model.os.getenv('API_URL')

    # Create URL for API call
    request_url = f"{ApiUrl}devices/{flow_meter_name.split('/')[0]}/actuators/{flow_meter_name.split('/')[1]}/value"
    
    # Define headers for the POST request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {create_model.Token}'
    }
    
    # Define the payload
    payload = amount

    try:
        # Send a POST request to the API
        response = requests.post(request_url, headers=headers, json=payload)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save times on when there was an irrigation TODO: wait for confirmation from microcontroller, irrigation could be skipped!!
            save_irrigation_time(amount)
            response_ok = True
        else:
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.text)
            response_ok = None
    except requests.exceptions.RequestException as e:
        # Handle request exceptions (e.g., connection errors)
        print("Request error:", e)
        response_ok = None  # TODO: introduce error handling
    
    return response_ok

# Mighty main function TODO: capsulate
def main_old(currentSoilTension, threshold_timestamp, predictions, irrigation_amount) -> int:
    global TimeSpanOverThreshold    
                                     
    # Get threshold from config                                      
    threshold = create_model.Current_config['Threshold']
    # set timestamp for debug reasons
    TimeSpanOverThreshold = create_model.Current_config['Look_ahead_time']
    future_time = datetime.now() + timedelta(hours=TimeSpanOverThreshold)
    threshold_timestamp = future_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    now = datetime.now().replace(microsecond=0)

        # "Weak" irrigation strategy
    # If threshold was met
    if currentSoilTension > threshold:
        print(f"Threshold: {threshold} was reached with a value of {currentSoilTension}.")

        # If current soil tension exceeds threshold by more than allowed margin, irrigate immediately
        if currentSoilTension > threshold * OverThresholdAllowed:
            print(f"Threshold: {threshold} was exceeded by 20%, irrigate immediately!")
            e = irrigate_amount(irrigation_amount)
            return e
        
        # Check predictions
        next_lower_idx, next_higher_idx = find_next_occurrences(predictions, 'prediction_label', threshold)

        # If no lower value is predicted within the forecast horizon, trigger irrigation
        if not next_lower_idx:
            print(f"No lower value predicted within {TimeSpanOverThreshold} hours, irrigate now!")
            e = irrigate_amount(irrigation_amount)
            return e
        
        # Otherwise, no immediate irrigation is needed
        else:
            print(f"Soil tension is high, but irrigation can wait. Since it is expected go below the threshold at: {next_lower_idx}")
            return 0

    # Threshold was not met, so do not irrigate
    else:
        print(f"Threshold: {threshold} is not reached, current soil tension is: {currentSoilTension}, do not irrigate.")
        return 0


# Mighty main function TODO: capsulate
def main(
    current_value, 
    threshold_timestamp, 
    predictions, 
    irrigation_amount, 
    strategy="tension"
) -> int:
    """
    Handles irrigation logic for soil tension and soil humidity based on the specified strategy.

    :param current_value: Current sensor value (e.g., soil tension or humidity).
    :param threshold_timestamp: Predicted threshold crossing timestamp.
    :param predictions: Predictions data (list or dataframe).
    :param irrigation_amount: Amount of irrigation to apply.
    :param strategy: Either "tension" or "humidity".
    :return: 1 if irrigation is triggered, otherwise 0.
    """
    global TimeSpanOverThreshold

    # Get configuration
    threshold = create_model.Current_config['Threshold']
    TimeSpanOverThreshold = create_model.Current_config['Look_ahead_time']

    # Define comparison logic based on strategy
    comparison_fn = (lambda value, threshold: value > threshold) if strategy == "tension" else (
        lambda value, threshold: value < threshold
    )

    # Define over-threshold logic
    over_threshold_fn = (
        lambda value, threshold: value > threshold * OverThresholdAllowed
        if strategy == "tension"
        else lambda value, threshold: value < threshold / OverThresholdAllowed
    )

    # "Weak" irrigation strategy
    if comparison_fn(current_value, threshold):
        print(f"Threshold: {threshold} was reached with a value of {current_value}.")

        # Immediate irrigation if over-threshold logic is satisfied
        if over_threshold_fn(current_value, threshold):
            print(f"Immediate irrigation triggered for strategy '{strategy}'!")
            return irrigate_amount(irrigation_amount)

        # Check predictions for next occurrence below/above threshold
        next_lower_idx, next_higher_idx = find_next_occurrences(predictions, 'prediction_label', threshold)

        # No recovery predicted within forecast horizon
        if (strategy == "tension" and not next_lower_idx) or (strategy == "humidity" and not next_higher_idx):
            print(f"No recovery predicted within {TimeSpanOverThreshold} hours, irrigate now!")
            return irrigate_amount(irrigation_amount)

        # Otherwise, delay irrigation
        else:
            target_time = next_lower_idx if strategy == "tension" else next_higher_idx
            print(f"Irrigation can wait. Recovery expected at: {target_time}")
            return 0

    # Threshold was not met, so do not irrigate
    else:
        print(f"Threshold: {threshold} is not reached, current value is: {current_value}. Do not irrigate.")
        return 0


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit