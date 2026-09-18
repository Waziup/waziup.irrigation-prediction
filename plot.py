# global:
from datetime import timedelta, datetime
import json
import os
import re
import urllib
import pandas as pd
import requests

# local:
from utils import NetworkUtils, TimeUtils
from sensor_roles import update_soil_sensor_groups
from state_store import configured_database_path, get_app_state_store

# Class plot members represent individual plots in the application


class Plot:
    # Class init, called when created in UI
    def __init__(self, tab_number, configPath, stable_id=None, farm_id=None):
        # Fundamental
        # Int to enumerate plots/tabs
        self.tab_number = tab_number
        # Path to current_config.json
        self.configPath = configPath
        # Current unique number, always incremented
        self.id = int(re.search(r'(\d+)\.json$', self.configPath).group(1))
        self.stable_id = stable_id or f"plot-legacy-{self.id}"
        # User given name is preset, but can be changed later
        self.user_given_name = "Plot " + str(self.id)
        # Links this compatibility Plot object to its authoritative YAML record.
        self.farm_id = farm_id
        # Farm-level environmental context. Weather uses this location for all
        # plots in the farm; EO continues to use this plot's own gps_info.
        self.farm_gps_info = {}
        self.farm_name = ""
        self.farm_timezone = "UTC"
        self.area_unit = "m2"
        # Installation/settings values win over optional manual YAML values.
        self.configuration_source = "legacy"
        self.timezone = "UTC"

        # Variables that were global before, now plot-specific
        # Device
        # Device address of humidity sensor
        self.device_and_sensor_ids_moisture = []
        # Device address of temperature sensor
        self.device_and_sensor_ids_temp = []
        # Device address of flow meter
        self.device_and_sensor_ids_flow = []
        # Sensor that reports delivered volume (normally xlpp channel 5).
        self.device_and_sensor_ids_flow_confirmation = []
        # Channel 5 historically reports the latest irrigation event. Some
        # installations instead expose a cumulative meter total.
        self.flow_confirmation_mode = "event"
        self.gps_info = ""                                  # Coordinates of sensors
        self.enable_experimental_ndre_kc = False
        self.sensor_kind = "tension"                        # Type of humidity sensor
        self.sensor_unit = ""                               # Unit of humidity
        # Slope to evaluate irrigation has taken place
        self.slope = 0
        self.threshold = 0                                  # Threshold to irrigate plants
        # Farmer-configured field baseline for static or dynamic triggering.
        self.threshold_static = 0
        # Static uses the baseline unchanged; dynamic adjusts it by crop stage.
        self.threshold_mode = "static"
        # Field-specific inputs for FAO-56 dynamic root-zone depletion.
        self.field_capacity_vwc = None
        self.wilting_point_vwc = None
        self.root_depth_m = None
        self.sensor_depth_m = None
        self.depletion_fraction = None
        self.stage_depletion_fractions = {}
        self.stage_thresholds_cbar = {}
        self.threshold_hysteresis_cbar = 0.0
        # Fractions used to convert forecast crop demand to irrigation demand.
        self.application_efficiency = 0.85
        self.effective_rainfall_fraction = 0.80
        self.irrigation_mode = ""
        # Time to look ahead in forecast how long soil tension threshold can be exceeded in hours
        self.look_ahead_time = 0
        self.watch_horizon_time = 72
        # Start date: use sensor and API data from this date
        self.start_date = ""
        # Time period to include into the model
        self.period = 0
        # Frequencies in days in between train cycles
        self.train_period_days = 1
        # Frequencies in hours in between predict cycles
        self.predict_period_hours = 3
        # Tension-model output cadence and forecast horizon. These defaults
        # match create_model.constants and may be overridden by plot config.
        self.forecast_interval_minutes = 60
        self.forecast_horizon_days = 5
        self.retrain_interval_days = 1
        self.irrigation_confirmation_seconds = 10800
        self.error_retry_seconds = 1800
        self.soil_type = ""                                 # Soil type for current field
        # Soil is too dry for plant water uptake.
        self.soil_calibration = {}
        self.permanent_wilting_point = 1500
        # Upper bound of soil is getting to dry
        self.field_capacity_upper = 10
        # Lower bound of wet soil, no more retention, water seeps through soil
        self.field_capacity_lower = 10
        # Soil is completely saturated with water
        self.saturation = 0
        self.soil_water_retention_curve = [                 # Soil water retention curve init
            # USDA-ARS ROSETTA class-average curve for sand. This is only a
            # texture estimate; a locally measured curve should replace it.
            (0, 0.375),
            (10, 0.072660),
            (33, 0.054478),
            (100, 0.053132),
            (300, 0.053012),
            (500, 0.053004),
            (1000, 0.053001),
            (1500, 0.053000),
        ]

        # Phenology configuration
        self.crop_type = ""
        self.planting_date = ""
        self.harvest_date = None
        self.initial_gdd = 0.0

        # Raw observations support fixed sensor freshness and plausibility
        # safeguards in sensor_quality.py.
        self.tension_sensor_observations = []

        # Initialize an empty dictionary to store the current config, obtained from "config.json"
        self.config = {}

        # Threading
        self.training_thread = None                         # Training thread of object
        # Prediction thread of object
        self.prediction_thread = None
        # Flag training process finished
        self.training_finished = False
        # Flag currently training this plot
        self.currently_training = False
        # Redundant as set in plot:manager
        self.currently_active = False

        # Data
        # Dataframe that holds data for training
        self.data = pd.DataFrame
        # Dataframe that stores weatherdata
        self.data_w = pd.DataFrame
        # Dataframe that holds latest predictions
        self.predictions = pd.DataFrame
        # Threshold timestamp when soil will be to dry
        self.threshold_timestamp = ""
        # Latest unified result plus its persistent-cache state.
        self.pipeline_result = None
        self.pipeline_cache_status = "empty"
        self.pipeline_cache_reason = None

        # Model
        # Stores the currently best model
        self.best_model = None
        # Stores the pycarets experiment object
        self.best_exp = None

        # Debug
        # Flag can be switched to decide on model usage
        self.use_pycaret = True
        # Flag can be switched to decide on data source
        self.load_data_from_csv = False
        # Flag to use ensemble/stacking model
        self.ensemble = True
        # Synthetic but weather-consistent debug dataset: tension simulated from real ERA5
        # weather at plot1's GPS (51.023591, 13.744087 / Dresden), Jun 2022 - Jun 2023, via a
        # leaky soil-water bucket with ~15 irrigation events. The old file
        # (binned_removed_new_for_app.csv) has 2 of 3 sensors failing (77%/28% zero-dropouts)
        # which corrupts the grouped target and makes it unlearnable.
        self.data_from_csv = "data/debug/synthetic_tension_dresden.csv"
        # Load former irrigations from file "data/irrigations.json" DEBUG
        self.load_irrigations_from_file = False
        self.irrigations_from_json = 'data/irrigations_plot_' + \
            str(self.id) + '.json'

    # Just print some class properties

    def printPlotNumber(self):
        print("Current object is plot/tab number: " + str(self.tab_number),
              ", with the path: " + self.configPath + ", it has the internal id: ", self.id)

    # Load config from file TODO: move the rest

    def getConfigFromFile(self):
        # Get path
        currentConfigPath = self.configPath
        data = None
        if configured_database_path() is not None:
            store = get_app_state_store()
            scope = f"plot_config:{self.stable_id}"
            data = store.load_plot_config(self.stable_id)
            if (data is None and not store.legacy_import_complete(scope)
                    and os.path.exists(currentConfigPath)):
                with open(currentConfigPath, 'r') as file:
                    data = json.load(file)
                store.save_plot_config(self.stable_id, data)
            if not store.legacy_import_complete(scope):
                store.mark_legacy_import_complete(scope)
        elif os.path.exists(currentConfigPath):
            with open(currentConfigPath, 'r') as file:
                # Parse JSON from the file
                data = json.load(file)
        if data is not None:
            if self.load_data_from_csv:
                debug_csv = pd.read_csv(self.data_from_csv, header=0)
                self.device_and_sensor_ids_moisture = []
                self.device_and_sensor_ids_temp = []
                self.device_and_sensor_ids_flow = []
                self.device_and_sensor_ids_flow_confirmation = []
                for column in debug_csv.columns:
                    if column.startswith((
                            "tension", "vwc", "volumetric", "capacitive")):
                        self.device_and_sensor_ids_moisture.append(column)
                    elif column.startswith("soil_temp"):
                        self.device_and_sensor_ids_temp.append(column)
                    elif column.startswith("flow"):
                        self.device_and_sensor_ids_flow.append(column)
            else:
                # Get chosen sensors
                self.device_and_sensor_ids_moisture = data.get(
                    'DeviceAndSensorIdsMoisture', [])
                self.device_and_sensor_ids_temp = data.get(
                    'DeviceAndSensorIdsTemp', [])
                self.device_and_sensor_ids_flow = data.get(
                    'DeviceAndSensorIdsFlow', [])
                confirmation = data.get(
                    'DeviceAndSensorIdsFlowConfirmation', [])
                self.device_and_sensor_ids_flow_confirmation = (
                    confirmation if isinstance(confirmation, list) else [])

            # Get data from forms
            self.user_given_name = data.get('Name', [])
            self.owner = data.get('Owner', getattr(self, 'owner', ''))
            # Older JSON configurations may omit this during YAML migration.
            self.farm_id = data.get('Farm_id', self.farm_id)
            self.area_unit = data.get('Plot_area_unit', self.area_unit)
            self.configuration_source = data.get(
                'Configuration_source', self.configuration_source)
            self.timezone = data.get('Timezone', self.timezone) or "UTC"

            self.zone_name = data.get('Zone_name', self.user_given_name)
            self.sensor_kind = data.get('Sensor_kind', [])
            gps_info = data.get('Gps_info', {})
            if isinstance(gps_info, dict):
                lat = gps_info.get('latitude', gps_info.get('lattitude', 0))
                lon = gps_info.get('longitude', 0)
                self.gps_info = {
                    "latitude": lat,
                    "longitude": lon,
                    "lattitude": lat,
                }
            else:
                self.gps_info = gps_info
            self.enable_experimental_ndre_kc = bool(
                data.get('Enable_experimental_ndre_kc', False))
            self.slope = float(data.get('Slope', []))
            self.threshold = float(data.get('Threshold', []))
            self.threshold_static = float(data.get('Threshold', []))
            self.threshold_mode = str(data.get(
                'Threshold_mode',
                'dynamic' if data.get('Use_dynamic_threshold', False) else 'static',
            )).strip().lower()
            self.field_capacity_vwc = data.get('Field_capacity_vwc')
            self.wilting_point_vwc = data.get('Wilting_point_vwc')
            self.root_depth_m = data.get('Root_depth_m')
            self.sensor_depth_m = data.get('Sensor_depth_m')
            self.depletion_fraction = data.get('Depletion_fraction')
            self.stage_depletion_fractions = data.get(
                'Stage_depletion_fractions', {}) or {}
            self.stage_thresholds_cbar = data.get(
                'Stage_thresholds_cbar', {}) or {}
            self.threshold_hysteresis_cbar = float(data.get(
                'Threshold_hysteresis_cbar', 0.0) or 0.0)
            self.application_efficiency = float(
                data.get('Application_efficiency', self.application_efficiency))
            self.effective_rainfall_fraction = float(data.get(
                'Effective_rainfall_fraction', self.effective_rainfall_fraction))
            self.plot_area_m2 = float(data.get('Plot_area_m2', 0))
            self.irrigation_type = data.get(
                'Irrigation_type', 'unknown') or 'unknown'
            self.irrigation_mode = data.get(
                'Irrigation_mode', self.irrigation_mode) or ''
            self.flow_confirmation_mode = str(data.get(
                'Flow_confirmation_mode', self.flow_confirmation_mode)
            ).strip().lower()
            self.look_ahead_time = float(data.get('Look_ahead_time', []))
            # Retain JSON timing reads for backward compatibility; startup YAML
            # application overwrites them with authoritative farm values.
            self.predict_period_hours = float(
                data.get('Predict_period_hours', self.predict_period_hours))
            self.forecast_interval_minutes = int(data.get(
                'Forecast_interval_minutes', self.forecast_interval_minutes))
            self.forecast_horizon_days = float(data.get(
                'Forecast_horizon_days', self.forecast_horizon_days))
            self.retrain_interval_days = float(data.get(
                'Retrain_interval_days', self.retrain_interval_days))
            self.irrigation_confirmation_seconds = int(data.get(
                'Irrigation_confirmation_seconds',
                self.irrigation_confirmation_seconds))
            self.error_retry_seconds = int(data.get(
                'Error_retry_seconds', self.error_retry_seconds))
            self.start_date = data.get('Start_date', [])
            self.period = int(data.get('Period', []))
            self.soil_type = data.get('Soil_type', [])
            self.soil_texture_class = data.get('Soil_texture_class', None)
            self.soil_calibration = data.get('Soil_calibration', {}) or {}
            self.permanent_wilting_point = float(
                data.get('PermanentWiltingPoint', []))
            self.field_capacity_upper = float(
                data.get('FieldCapacityUpper', []))
            self.field_capacity_lower = float(
                data.get('FieldCapacityLower', []))
            self.saturation = float(data.get('Saturation', []))

            # Get soil water retention curve -> currently not needed here
            self.soil_water_retention_curve = data.get(
                'Soil_water_retention_curve', [])

            # Phenology configuration
            self.crop_type = data.get('Crop_type', '')
            self.planting_date = data.get('Planting_date', '')
            self.harvest_date = data.get('Harvest_date')
            try:
                self.initial_gdd = float(data.get('Initial_gdd', 0.0))
            except (TypeError, ValueError):
                self.initial_gdd = 0.0
            self.farm_data_bundle = data.get('Farm_data_bundle', None)

            # Sensor kind
            if self.sensor_kind in ("tension", "both"):
                self.sensor_unit = "Moisture in cbar (Soil Tension)"
            elif self.sensor_kind == "capacitive":
                self.sensor_unit = "Moisture in % (Volumetric Water Content)"
            else:
                self.sensor_unit = "Unit is unknown"

            return True
        else:
            return False

    def getConfirmationDeviceID(self, actuator_ids=None):
        """Return an explicit or xlpp-channel-5 delivery sensor reference."""
        configured = self.device_and_sensor_ids_flow_confirmation
        if isinstance(configured, list) and configured:
            return configured[0]
        actuator_ids = actuator_ids or self.device_and_sensor_ids_flow
        if not isinstance(actuator_ids, list) or not actuator_ids:
            return ""
        actuator_ref = str(actuator_ids[0])
        if "/" not in actuator_ref:
            return ""
        device_id = actuator_ref.split("/", 1)[0]
        url = f"{NetworkUtils.ApiUrl}devices/{device_id}/sensors"
        headers = {'Authorization': f'Bearer {NetworkUtils.Token}'}
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                return ""
            sensors = response.json()
            if isinstance(sensors, dict):
                sensors = sensors.get("sensors", [])
            for sensor in sensors if isinstance(sensors, list) else []:
                meta = sensor.get("meta", {}) if isinstance(sensor, dict) else {}
                channel = meta.get("xlppChan", meta.get("xlppchan"))
                try:
                    matches = int(channel) == 5
                except (TypeError, ValueError):
                    matches = False
                sensor_id = sensor.get("id") if isinstance(sensor, dict) else None
                if matches and sensor_id:
                    return f"{device_id}/{sensor_id}"
        except (requests.exceptions.RequestException, ValueError, TypeError):
            return ""
        return ""

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
            try:
                NetworkUtils.get_token()
            except requests.exceptions.RequestException:
                print("Gateway authentication request failed; sensor data unavailable.")
                return None

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
            response = requests.get(encoded_url, headers=headers, timeout=30)

            # Handle token expiration (HTTP 401)
            if response.status_code == 401:
                print("Token expired, refreshing token...")
                NetworkUtils.get_token()  # Refresh token
                headers['Authorization'] = f'Bearer {NetworkUtils.Token}'
                response = requests.get(
                    request_url, headers=headers, timeout=30)  # Retry request

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
            return None

        return response_ok

    # Load data from CSV file
    def load_latest_data_csv(self, sensor_name, type):
        print("load_latest_data_csv: will load data for plot: " +
              self.user_given_name + " For the sensor: " + sensor_name)

        # Load data from CSV file
        try:
            data = self.load_data_csv()
            return data[sensor_name].iloc[-1]
        except FileNotFoundError:
            print("File not found:", self.data_from_csv)
            return None
        except Exception as e:
            print("An error occurred, loading latest data from csv file:", e)
            return None

    def read_config(self):
        # Specify the path to the JSON file you want to read
        json_file_path = self.configPath

        store = get_app_state_store() if configured_database_path() is not None else None
        scope = f"plot_config:{self.stable_id}"
        config = store.load_plot_config(self.stable_id) if store is not None else None
        if config is None:
            if store is not None and store.legacy_import_complete(scope):
                raise FileNotFoundError(
                    f"No configuration exists for plot {self.stable_id}")
            with open(json_file_path, 'r') as json_file:
                config = json.load(json_file)
            if store is not None:
                store.save_plot_config(self.stable_id, config)
        if store is not None and not store.legacy_import_complete(scope):
            store.mark_legacy_import_complete(scope)

        # Check if the CSV file exists
        try:
            if self.load_data_from_csv is True:
                with open(self.data_from_csv, "r") as file:
                    # Perform operations on the file
                    data = pd.read_csv(file, header=0)

                self.device_and_sensor_ids_moisture = []
                self.device_and_sensor_ids_temp = []
                self.device_and_sensor_ids_flow = []
                self.device_and_sensor_ids_flow_confirmation = []

                # create array with sensors strings
                for col in data.columns:
                    if (
                        col.startswith("tension")
                        or col.startswith("vwc")
                        or col.startswith("volumetric")
                        or col.startswith("capacitive")
                    ):
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
                confirmation = config.get(
                    "DeviceAndSensorIdsFlowConfirmation", [])
                self.device_and_sensor_ids_flow_confirmation = (
                    confirmation if isinstance(confirmation, list) else [])
        # If the CSV file does not exist, use data from API
        except FileNotFoundError:
            print(
                f"Debug mode was set in .env, but no file with was found in {self.data_from_csv}. Received the following error: {e}")
        except Exception as e:
            print(
                "An error occurred in read config: No devices are set in settings, there is also no local config file.", e)

        update_soil_sensor_groups(self)
        return config

    # Load from wazigate API, !!!! TODO: investigate why HTTP server becomes unresponsive after running!!!!
    def load_data_api(self, sensor_name, type, from_timestamp):  # , token)
        print("load_data_api: will load data for plot: " + self.user_given_name
              + " For the sensor: " + sensor_name)

        # Obtain ApiUrl
        apiUrl = NetworkUtils.ApiUrl

        timezone_name = TimeUtils.for_plot(self)
        timestamp = pd.Timestamp(from_timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                timezone_name, ambiguous="raise", nonexistent="raise")
        timestamp = timestamp.tz_convert("UTC")
        from_timestamp = timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        if apiUrl.startswith('http://wazigate/'):
            print('There is no token needed, fetch data from local gateway.')
        elif NetworkUtils.Token != None and NetworkUtils.Token != "":
            print('There is no token needed, already present.')
        # Get token, important for non localhost devices
        else:
            try:
                NetworkUtils.get_token()
            except requests.exceptions.RequestException:
                print("Gateway authentication request failed; sensor data unavailable.")
                return None

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

        response_ok = None
        try:
            # Send a GET request to the API
            response = requests.get(encoded_url, headers=headers, timeout=30)

            # Handle token expiration (HTTP 401)
            if response.status_code == 401:
                print("Token expired, refreshing token...")
                NetworkUtils.get_token()  # Refresh token
                headers['Authorization'] = f'Bearer {NetworkUtils.Token}'
                response = requests.get(
                    encoded_url, headers=headers, timeout=30)  # Retry request

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # The response content contains the data from the API
                response_ok = response.json()
            else:
                print("Request failed with status code:", response.status_code)
        except requests.exceptions.RequestException as e:
            # Handle request exceptions (e.g., connection errors)
            print("Request error:", e)
            return None

        return response_ok

    def load_data_csv(self):
        print("load_data_csv: will load data from CSV for plot: " +
              self.user_given_name)

        # Load data from CSV file
        try:
            with open(self.data_from_csv, "r") as file:
                # Specify the column(s) you want to load
                data = pd.read_csv(file, header=0)  # , usecols=[sensor_name])
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
        # A Thread instance cannot be started twice. Restart prediction through
        # its module factory only after a successful training cycle.
        if self.training_finished and (
            not self.prediction_thread or not self.prediction_thread.is_alive()
        ):
            print("Prediction thread not alive, restarting...")
            import prediction_thread
            prediction_thread.start(self)

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
