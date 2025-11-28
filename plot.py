# global:
from datetime import timedelta
import datetime
import json
import os
import re
import urllib
import pandas as pd
import requests

# local:
from utils import NetworkUtils, TimeUtils

# Class plot members represent individual plots in the application
class Plot:
    # Class init, called when created in UI
    def __init__(self, tab_number, configPath):
        # Fundamental
        # Int to enumerate plots/tabs
        self.tab_number = tab_number
        # Path to current_config.json
        self.configPath = configPath
        # Current unique number, always incremented
        self.id = int(re.search(r'(\d+)\.json$', self.configPath).group(1))
        # User given name is preset, but can be changed later
        self.user_given_name = "Plot " + str(self.id)

        # Variables that were global before, now plot-specific
        # Device
        self.device_and_sensor_ids_moisture = []            # Device address of humidity sensor
        self.device_and_sensor_ids_temp = []                # Device address of temperature sensor
        self.device_and_sensor_ids_flow = []                # Device address of flow meter
        self.device_and_sensor_ids_flow_confirmation = []   # Device address of flow meter confirmation sensor
        self.gps_info = ""                                  # Coordinates of sensors
        self.sensor_kind = "tension"                        # Type of humidity sensor
        self.sensor_unit = ""                               # Unit of humidity
        self.slope = 0                                      # Slope to evaluate irrgation has taken place
        self.threshold = 0                                  # Threshold to irrigate plants
        self.irrigation_amount = 0                          # Amount in liters to irrigate plants
        self.look_ahead_time = 0                            # Time to look ahead in forecast how long soil tension threshold can be exceeded in hours
        self.start_date = ""                                # Start date: use sensor and API data from this date
        self.period = 0                                     # Time period to include into the model
        self.train_period_days = 1                          # Frequencies in days inbeween train cycles
        self.predict_period_hours = 6                       # Frequencies in hours inbeween predict cycles
        self.soil_type = ""                                 # Soil type for current field                           
        self.permanent_wilting_point = 40                   # Soil is to dry, plant cannot access any water with its roots
        self.field_capacity_upper = 30                      # Upper bound of soil is getting to dry
        self.field_capacity_lower = 10                      # Lower bound of wet soil, no more retention, water seeps through soil
        self.saturation = 0                                 # Soil is completly saturated with water
        self.soil_water_retention_curve = [                 # Soil water retention curve init
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
        self.training_thread = None                         # Training thread of object
        self.prediction_thread = None                       # Prediction thread of object
        self.training_finished = False                      # Flag training process finished
        self.currently_training = False                     # Flag currently training this plot
        self.currently_active = False                       # Redundant as set in plot:manager

        # Data
        self.data = pd.DataFrame                            # Dataframe that holds data for training
        self.data_w = pd.DataFrame                          # Dataframe that stores weatherdata
        self.predictions = pd.DataFrame                     # Dataframe that holds latest predictions
        self.threshold_timestamp = ""                       # Threshold timestamp when soil will be to dry

        # Model
        self.best_model = None                              # Stores the currently best model
        self.best_exp = None                                # Stores the pycarets experiment object

        # Debug
        self.use_pycaret = True                             # Flag can be switched to decide on model usage
        self.load_data_from_csv = False                     # Flag can be switched to decide on data source
        self.ensemble = True                                # Flag to use ensemble/stacking model
        self.data_from_csv = "data/debug/binned_removed_new_for_app.csv"
        # Load former irrigations from file "data/irrigations.json" DEBUG
        self.load_irrigations_from_file = False
        self.irrigations_from_json = 'data/irrigations_plot_' + str(id)  + '.json'

        def __repr__(self):
            return (
                f"plotTabNumber(tab_number={self.tab_number}, "
                f"name='{self.user_given_name}', "
                f"configPath='{self.configPath}', "
                f"training_active={self.currently_training}, "
                f"prediction_active={self.prediction_thread is not None})"
            )

    # Just print some class properies

    def printPlotNumber(self):
        print("Current object is plot/tab number: " + str(self.tab_number),
              ", with the path: " + self.configPath + ", it has the internal id: ", self.id)
        

    # Load config from file TODO: move the rest
    def getConfigFromFile(self):
        # Get path
        currentConfigPath = self.configPath

        if os.path.exists(currentConfigPath):
            with open(currentConfigPath, 'r') as file:
                # Parse JSON from the file
                data = json.load(file)

            if not self.load_data_from_csv:
                # Get choosen sensors
                self.device_and_sensor_ids_moisture = data.get('DeviceAndSensorIdsMoisture', [])
                self.device_and_sensor_ids_temp = data.get('DeviceAndSensorIdsTemp', [])
                self.device_and_sensor_ids_flow = data.get('DeviceAndSensorIdsFlow', [])
                self.device_and_sensor_ids_flow_confirmation = data.get('DeviceAndSensorIdsFlowConfirmation', [])

            # Get data from forms
            self.user_given_name = data.get('Name', [])
            self.sensor_kind = data.get('Sensor_kind', [])
            self.gps_info = data.get('Gps_info', [])
            self.slope = float(data.get('Slope', []))
            self.threshold = float(data.get('Threshold', []))
            self.irrigation_amount = float(data.get('Irrigation_amount', []))
            self.look_ahead_time = float(data.get('Look_ahead_time', []))
            self.start_date = data.get('Start_date', [])
            self.period = int(data.get('Period', []))
            self.soil_type = data.get('Soil_type', [])
            self.permanent_wilting_point = float(data.get('PermanentWiltingPoint', []))
            self.field_capacity_upper = float(data.get('FieldCapacityUpper', []))
            self.field_capacity_lower = float(data.get('FieldCapacityLower', []))
            self.saturation = float(data.get('Saturation', []))

            # Get soil water retention curve -> currently not needed here
            self.soil_water_retention_curve = data.get('Soil_water_retention_curve', [])

            # Sensor kind
            if self.sensor_kind == "tension":
                self.sensor_unit = "Moisture in cbar (Soil Tension)"
            elif self.sensor_kind == "capacitive":
                self.sensor_unit = "Moisture in % (Volumetric Water Content)"
            else:
                self.sensor_unit = "Unit is unknown"

            return True
        else:
            return False
        

    #Get the device ID of the confirmation sensor (xlppChan == 5)
    def getConfirmationDeviceID(self, id):
        # API request to get sensor meta data of a device
        device_id = id[0].split('/')[0]
        url = f"{NetworkUtils.ApiUrl}devices/{device_id}/sensors" 
        headers = {
            'Authorization': f'Bearer {NetworkUtils.Token}'
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch sensors, status code: {response.status_code}")
                return ""

            data = response.json()

            # Find the sensor ID with xlppChan == 5
            sensor_id = next(
                (sensor["id"] for sensor in data
                if sensor.get("meta", {}).get("xlppChan") == 5),
                ""
            )
        except requests.exceptions.RequestException as e:
            print("Request error:", e)
            print(f"Determining the confirmation sensor of the actuator falied for plot {self.id}.")
            return ""
        
        return device_id + "/" + sensor_id

    # Obtain current sensor value from API
    def load_latest_data_api(self, sensor_name, type):  # , token)
        print("load_latest_data_api: will load data for plot: " +
              self.user_given_name + " For the sensor: " + sensor_name)
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

            # Handle token expiration (HTTP 401)
            if response.status_code == 401:
                print("Token expired, refreshing token...")
                NetworkUtils.get_token()  # Refresh token
                headers['Authorization'] = f'Bearer {NetworkUtils.Token}'
                response = requests.get(
                    request_url, headers=headers)  # Retry request

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
    
    # Load data from CSV file
    def load_latest_data_csv(self, sensor_name, type):
        print("load_latest_data_csv: will load data for plot: " +
              self.user_given_name + " For the sensor: " + sensor_name)

        # Load data from CSV file
        try:
            with open(self.data_from_csv, "r") as file:
                # Specify the column(s) you want to load
                data = pd.read_csv(file, header=0, usecols=[sensor_name])
            return data.iloc[-1, 0] # last element first col
        except FileNotFoundError:
            print("File not found:", self.data_from_csv)
            return None
        except Exception as e: 
            print("An error occurred, loading latest data from csv file:", e)
            return None

    # Load from CSV file -> obsolete
    def load_data(path):
        # creating a data frame
        data = pd.read_csv("binned_removed.csv", header=0)
        print(data.head())
        return data

    def read_config(self):
        # Specify the path to the JSON file you want to read
        json_file_path = self.configPath

        # Read the JSON data from the file
        with open(json_file_path, 'r') as json_file:
            config = json.load(json_file)

        # Check if the CSV file exists
        try:
            if self.load_data_from_csv is True:
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
                    # This is not implemented
                    elif col.startswith("flow"):
                        self.device_and_sensor_ids_flow.append(col)
            else:
                self.device_and_sensor_ids_moisture = config["DeviceAndSensorIdsMoisture"]
                self.device_and_sensor_ids_temp = config["DeviceAndSensorIdsTemp"]
                if "DeviceAndSensorIdsFlow" in config:
                    self.device_and_sensor_ids_flow = config["DeviceAndSensorIdsFlow"]
        # If the CSV file does not exist, use data from API
        except FileNotFoundError:
             print(f"Debug mode was set in .env, but no file with was found in {self.data_from_csv}. Received the following error: {e}")
        except Exception as e:
            print(
                "An error occurred in read config: No devices are set in settings, there is also no local config file.", e)

        return config

    # Load from wazigate API, !!!! TODO: investigate why HTTP server becomes unresponsive after running!!!!
    def load_data_api(self, sensor_name, type, from_timestamp):  # , token)
        print("load_data_api: will load data for plot: " + self.user_given_name
               + " For the sensor: " + sensor_name)

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

            # Handle token expiration (HTTP 401)
            if response.status_code == 401:
                print("Token expired, refreshing token...")
                NetworkUtils.get_token()  # Refresh token
                headers['Authorization'] = f'Bearer {NetworkUtils.Token}'
                response = requests.get(
                    encoded_url, headers=headers)  # Retry request

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
        
    def load_data_csv(self):
        print("load_data_csv: will load data from CSV for plot: " + self.user_given_name)

        # Load data from CSV file
        try:
            with open(self.data_from_csv, "r") as file:
                # Specify the column(s) you want to load
                data = pd.read_csv(file, header=0)#, usecols=[sensor_name])
            return data
        except FileNotFoundError:
            print("File not found:", self.data_from_csv)
            return None
        except Exception as e: 
            print("An error occurred, loading data from csv file:", e)
            return None


    # Redundant set active state

    def setState(self, state):
        self.currently_active = state

    # Redundant get active state
    def getState(self):
        return self.currently_active

    # Threads

  # Set the training thread
    def setTrainingThread(self, thread):
        self.training_thread = thread

    # Get the training thread
    def getTrainingThread(self):
        return self.training_thread

    # Set the prediction thread
    def setPredictionThread(self, thread):
        self.prediction_thread = thread

    # Get the prediction thread
    def getPredictionThread(self):
        return self.prediction_thread
    
    # Check if a training thread is already running
    def isTrainingRunning(self):
        return self.training_thread is not None and self.training_thread.is_alive()
    
    # surveillance, check threads are running
    def check_threads(self):
        print("Checking threads of plot: " + self.user_given_name)
        if not self.training_thread or not self.training_thread.is_alive():
            print("Training thread not alive, restarting...")
            self.training_thread.start(self)

        if not self.prediction_thread or not self.prediction_thread.is_alive():
            print("Prediction thread not alive, restarting...")
            self.prediction_thread.start(self)

    # Data

    # Data getter for display in UI

    def get_Data_for_display(self, to_be_dropped):
        if self.data.empty:
            return False
        else:
            # This is needed to prevent the timestamp is being omitted
            return self.data.drop([item for item in to_be_dropped if item != "Timestamp"], axis=1)

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
