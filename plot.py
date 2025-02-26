# global:
from datetime import timedelta
import datetime
import json
import urllib
import pandas as pd
import requests

# local:
from utils import NetworkUtils, TimeUtils


class Plot:
  # Class init, called when created in UI
  def __init__(self, plot_number, configPath):
    # Fundamental
    self.plot_number = plot_number                # Int to enumerate plots
    self.configPath = configPath                  # Path to current_config.json

    # Variables that were global before, now plot-specific TODO: implement everywhere
    # Device
    self.device_and_sensor_ids_moisture = []      # Device address of humidity sensor
    self.device_and_sensor_ids_temp = []          # Device address of temperature sensor
    self.device_and_sensor_ids_flow = []          # Device address of flow meter
    self.gps_info = ""                            # Coordinates of sensors
    self.sensor_kind = "tension"                  # Type of humidity sensor
    self.sensor_unit = ""                         # Unit of humidity
    self.slope = 0                                # Slope to evaluate irrgation has taken place
    self.threshold = 0                            # Threshold to irrigate plants
    self.irrigation_amount = 0                    # Amount in liters to irrigate plants
    self.look_ahead_time = 0                      # Time to look ahead in forecast how long soil tension threshold can be exceeded in hours
    self.start_date = ""                          # Start date: use sensor and API data from this date
    self.period = 0                               # Time period to include into the model
    self.train_period_days = 1                    # Frequencies in days inbeween train cycles
    self.predict_period_hours = 6                 # Frequencies in hours inbeween predict cycles           
    self.soil_type = ""                           # Soil type for current field
    self.permanent_wilting_point = 40             # Soil is to dry, plant cannot access any water with its roots
    self.field_capacity_upper = 30                # Upper bound of soil is getting to dry
    self.field_capacity_lower = 10                # Lower bound of wet soil, no more retention, water seeps through soil
    self.saturation = 0                           # Soil is completly saturated with water
    self.soil_water_retention_curve = [           # Soil water retention curve init
        (0, 0.45),
        (5, 0.40),
        (10, 0.37),
        (20, 0.30),
        (50, 0.25),
        (100, 0.20),
        (200, 0.15),
        (500, 0.10),
        (1000, 0.05),
    ]

    # Initialize an empty dictionary to store the current config, obtained from "config.json"
    self.config = {} 

    # Threading
    self.training_thread = None
    self.prediction_thread = None
    self.training_finished = False
    self.currently_training = False
    self.currently_active = False

    # Data
    self.data = pd.DataFrame
    self.data_w = pd.DataFrame
    self.predictions = pd.DataFrame
    self.threshold_timestamp = ""
    self.use_pycaret = True
    self.tuned_best = None
    self.best_exp = None

    # Debug
    self.load_data_from_csv = False
    self.data_from_csv = "binned_removed_new_for_app_ww.csv"
    self.load_irrigations_from_file = False # Load former irrigations from file "data/irrigations.json" DEBUG
    self.irrigations_from_json = "data/irrigations.json"


  # Just print some class properies
  def printPlotNumber(self):
    print("Current object is plot number: " + str(self.plot_number),
          ", with the path: " + self.configPath)

  # Obtain current config from file
  def load_latest_data_api(self, sensor_name, type):  # , token)
    apiUrl = NetworkUtils.ApiUrl

    if apiUrl.startswith('http://wazigate/'):
        print('There is no token needed, fetch data from local gateway.')
    elif NetworkUtils.Token != None and NetworkUtils.Token != "":
        print('There is no token needed, already present.')
    # Get token, important for non localhost devices
    else:
        NetworkUtils.get_token()

    # Create URL for API call e.g.:curl -X GET "http://192.168.189.15/devices/669780aa68f319066a12444a/sensors/6697875968f319066a12444d/value" -H "accept: application/json"
    request_url = apiUrl + "devices/" + \
        sensor_name.split('/')[0] + "/" + type + "/" + \
                          sensor_name.split('/')[1] + "/value"
    # Parse the URL
    parsed_url = urllib.parse.urlsplit(request_url)

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
        'Authorization': f'Bearer {NetworkUtils.Token}',
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
            print("Response content:", response.text)
            response_ok = None
    except requests.exceptions.RequestException as e:
        # Handle request exceptions (e.g., connection errors)
        print("Request error:", e)
        return "Error in 'load_latest_data_api()'! ", e  # TODO: intruduce error handling!

    return response_ok

  # Load from CSV file -> obsolete
  def load_data(path):
    # creating a data frame
    data = pd.read_csv("binned_removed.csv", header=0)
    print(data.head())
    return data


  def read_config(self):
    # Specify the path to the JSON file you want to read
    json_file_path = self.configPath

    try:
        with open(self.data_from_csv, "r") as file:
            # Perform operations on the file
            data = pd.read_csv(file, header=0)
            self.device_and_sensor_ids_moisture = []
            self.device_and_sensor_ids_temp = []
            self.device_and_sensor_ids_flow = []

            # create array with sensors strings
            for col in data.columns:
                if col.startswith("tension"):
                    self.device_and_sensor_ids_moisture.append(col)
                elif col.startswith("soil_temp"):
                    self.device_and_sensor_ids_temp.append(col)
            self.load_data_from_csv = True
            # This is not implemented: self.device_and_sensor_ids_flow = config["DeviceAndSensorIdsFlow"]
    except FileNotFoundError:
        # Read the JSON data from the file
        with open(json_file_path, 'r') as json_file:
            config = json.load(json_file)

        self.device_and_sensor_ids_moisture = config["DeviceAndSensorIdsMoisture"]
        self.device_and_sensor_ids_temp = config["DeviceAndSensorIdsTemp"]
        if "DeviceAndSensorIdsFlow" in config:
             self.device_and_sensor_ids_flow = config["DeviceAndSensorIdsFlow"]
    except Exception as e:
        print("An error occurred: No devices are set in settings, there is also no local config file.", e)

    return config

  # Load from wazigate API
  def load_data_api(self, sensor_name, type, from_timestamp):  # , token)
    # Load config to obtain GPS coordinates
    self.config = self.read_config()

    # Obtain ApiUrl
    apiUrl = NetworkUtils.ApiUrl

    # Convert timestamp
    if not isinstance(from_timestamp, str):
        from_timestamp = from_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # Get timezone if no information avalable
    if TimeUtils.Timezone == '':
        TimeUtils.Timezone = TimeUtils.get_timezone(
            self.config["Gps_info"]["lattitude"], self.config["Gps_info"]["longitude"])

    # Correct timestamp for timezone => TODO: here is an ERROR, timezone var is not available in first start
    from_timestamp = (datetime.datetime.strptime(from_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ") -
                      timedelta(hours=TimeUtils.get_timezone_offset(TimeUtils.Timezone))).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    if apiUrl.startswith('http://wazigate/'):
        print('There is no token needed, fetch data from local gateway.')
    elif NetworkUtils.Token != None and NetworkUtils.Token != "":
        print('There is no token needed, already present.')
    # Get token, important for non localhost devices
    else:
        NetworkUtils.get_token()

    # Create URL for API call
    api_url = apiUrl + "devices/" + sensor_name.split('/')[0] + "/" + type + "/" + sensor_name.split('/')[
                                                      1] + "/values" + "?from=" + from_timestamp
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
        'Authorization': f'Bearer {NetworkUtils.Token}',
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
        return "", e  # TODO: intruduce error handling!

    return response_ok
  

  # Redundant set active state
  def setState(state):
    global Currently_active
    Currently_active = state

  # Redundant get active state
  def getState():
    return Currently_active
  
  # Threads

  # Set the current thread that runs prediction
  def setPredictionhread(thread):
    global prediction_thread
    prediction_thread.append(thread)

  # Set the current thread that runs training
  def setTrainingThread(thread):
    global training_thread
    training_thread.append(thread)

  # Set training thread
  def setTrainingThread(thread):
    global Training_thread
    Training_thread = thread

  # Get training thread
  def getTrainingThread(thread):
    return Training_thread
  
  # Set prediction thread
  def setPredictionThread(thread):
    global Prediction_thread
    Prediction_thread = thread

  # Get prediction thread
  def getPredictionThread(thread):
    return Prediction_thread
  
  # Data 
  
  # Data getter for display in UI
  def get_Data_for_display(self, to_be_dropped):
    if self.data.empty:
        return False
    else:
        return self.data.drop([item for item in to_be_dropped if item != "Timestamp"], axis=1) # This is needed to prevent the timestamp is being omitted

  # Predictions Getter
  def get_predictions(self):
      if self.predictions.empty:
          return False
      else:
          return self.predictions
      
  # threshold timestamp Getter
  def get_threshold_timestamp(self):
      if not self.threshold_timestamp:
          return False
      else:
          return self.threshold_timestamp