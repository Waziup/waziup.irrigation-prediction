# TODO: delete ports from URLs


#!/usr/bin/python
import csv
from datetime import datetime, timedelta
from io import StringIO
import json
import pickle
import threading
import time
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import requests
import urllib
import usock
import os
import pathlib
import numpy as np

import create_model
import actuation



#---------------------#
# Path for proxy.sock, set by .env
#usock.sockAddr = ""

# Path to the root of the code
PATH = os.path.dirname(os.path.abspath(__file__))

# Path of config file
ConfigPath = 'config/current_config.json'

# global list of device and sensor ids
DeviceAndSensorIdsMoisture = []
DeviceAndSensorIdsTemp = []
DeviceAndSensorIdsFlow = []

# GPS
Gps_info = ""

# Soil moisture sensor kind
Sensor_kind = "tension"
Sensor_unit = ""

# Slope to evaluate irrgation has taken place
Slope = 0

# Threshold to irrigate plants
Threshold = 0

# Amount in liters to irrigate plants
Irrigation_amount = 0

# Time to look ahead in forecast how long soil tension threshold can be exceeded in hours
Look_ahead_time = 0

# Start date
Start_date = ""

# Time period to include into the model
Period = 0

# Frequencies train/predict
Train_period_days = 1
Predict_period_hours = 6 #6 DEBUG

# Soil_type
Soil_type = ""

# Retention curve => not needed any more
Soil_water_retention_curve = [
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

# Other soil related params
PermanentWiltingPoint = 40
FieldCapacityUpper = 30
FieldCapacityLower = 10
Saturation = 0


# Array of active threads TODO: if training started kill other threads.
Threads = []
ThreadId = 0
Training_thread = None
Prediction_thread = None
Restart_time = 1800 # DEBUG

TrainingFinished = False
CurrentlyTraining = False

# Load variables of training from file, to debug actuation part
Perform_training = True

# Set the threshold to cleanup models to 3 months (approximately 90 days)
THRESHOLD_DAYS_CLEANUP = 90

#---------------------#


def index(url, body=""):
    return 200, b"Salam Goloooo", []


usock.routerGET("/", index)

#------------------#


def ui(url, body=''):
    filename = urlparse(url).path.replace("/ui/", "")
    if (len(filename) == 0):
        filename = 'index.html'

    #---------------#

    ext = pathlib.Path(filename).suffix

    extMap = {
        '': 'application/octet-stream',
        '.manifest': 'text/cache-manifest',
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.svg':	'image/svg+xml',
        '.css':	'text/css',
        '.js': 'application-x/javascript',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
        '.xml': 'application/xml',
    }

    if ext not in extMap:
        ext = ""

    conType = extMap[ext]

    #---------------#

    try:
        with open(PATH + '/ui/' + filename, mode='rb') as file:
            return 200, file.read(), [conType]
    except Exception as e:
        print("Error: ", e)
        return 404, b"File not found", []


usock.routerGET("/ui/(.*)", ui)
usock.routerPOST("/ui/(.*)", ui)

#------------------#

# Cleans python and pycaret logs
class LogCleanerThread(threading.Thread):
    def __init__(self, file_path, age_limit_days=90, check_interval=86400):
        super().__init__()
        self.file_path = file_path
        self.age_limit_days = age_limit_days
        self.check_interval = check_interval
        self.stop_thread = threading.Event()

    def clean_log(self):
        """Clears log file if it is older than the age limit."""
        if os.path.exists(self.file_path):
            last_modified_time = datetime.fromtimestamp(os.path.getmtime(self.file_path))
            if datetime.now() - last_modified_time > timedelta(days=self.age_limit_days):
                open(self.file_path, 'w').close()  # Clear the file contents
                print(f"{self.file_path} has been cleaned.")
            else:
                print(f"{self.file_path} is not old enough to clean.")

    def run(self):
        while not self.stop_thread.is_set():
            self.clean_log()
            time.sleep(self.check_interval)  # Wait before the next check

    def stop(self):
        self.stop_thread.set()

# setup function for log cleaner
def schedule_log_cleanup():
    # Paths to the log files you want to monitor and clean
    logs_to_clean = [
        ("logs.log", 90),       # Python log file
        ("python_logs.log", 90) # Docker log file
    ]

    # Start a thread for each log file
    cleaner_threads = []
    for log_path, age_limit in logs_to_clean:
        cleaner = LogCleanerThread(file_path=log_path, age_limit_days=age_limit)
        cleaner.start()
        cleaner_threads.append(cleaner)

    try:
        # Keep the main thread alive while the cleaner threads run
        for cleaner in cleaner_threads:
            cleaner.join()
    except KeyboardInterrupt:
        print("Stopping log cleaners...")
        for cleaner in cleaner_threads:
            cleaner.stop()
        for cleaner in cleaner_threads:
            cleaner.join()

# Deletes files older than the threshold from the specified folder and its subfolders.
def delete_old_files(folder_path):
    current_time = time.time()
    threshold_time = current_time - THRESHOLD_DAYS_CLEANUP * 24 * 60 * 60

    # Traverse the directory, including subdirectories
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Check if the file is older than the threshold
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < threshold_time:
                try:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

# setup function for model cleaner
def schedule_model_cleanup(folder_path, interval_days=7):
    """
    Periodically runs the delete_old_files function every interval_hours.
    """
    from threading import Timer
    
    # Inner function to recursively schedule the cleanup
    def run_cleanup():
        delete_old_files(folder_path)
        Timer(interval_days * 24 * 3600, run_cleanup).start()
    
    # Start the first cleanup run
    run_cleanup()

# Get URL of API from .env file => TODO: better with try catch than locals, getenv can still stop backend
def getApiUrl(url, body):
    #load_dotenv()
    url = os.getenv("API_URL")
    if url not in (None, ''):
        data = url
        status_code = 200
    else:
        data=False,
        status_code = 400

    response = {
        "data": data,
        "status_code": status_code
    }

    return status_code, bytes(json.dumps(response), "utf8"), []

usock.routerGET("/api/getApiUrl", getApiUrl)


# Get historical sensor values from WaziGates API
def setConfig(url, body):
    global DeviceAndSensorIdsMoisture
    global DeviceAndSensorIdsTemp
    global DeviceAndSensorIdsFlow
    global Sensor_kind
    global Gps_info
    global Slope
    global Threshold
    global Irrigation_amount
    global Look_ahead_time
    global Start_date
    global Period
    global Soil_type
    global PermanentWiltingPoint
    global FieldCapacityUpper
    global FieldCapacityLower
    global Saturation
    global Soil_water_retention_curve

    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    # Get choosen sensors
    DeviceAndSensorIdsMoisture = parsed_data.get('selectedOptionsMoisture', [])
    DeviceAndSensorIdsTemp = parsed_data.get('selectedOptionsTemp', [])
    DeviceAndSensorIdsFlow = parsed_data.get('selectedOptionsFlow', [])

    # Get data from forms
    Sensor_kind = parsed_data.get('sensor_kind', [])[0]
    Gps_info = parsed_data.get('gps', [])[0]
    Slope = parsed_data.get('slope', [])[0]
    Threshold = float(parsed_data.get('thres', [])[0])
    Irrigation_amount = float(parsed_data.get('amount', [])[0])
    Look_ahead_time = float(parsed_data.get('lookahead', [])[0])
    Start_date = parsed_data.get('start', [])[0]
    Period = int(parsed_data.get('period', [])[0])
    Soil_type = parsed_data.get('soil', [])[0]
    PermanentWiltingPoint = int(parsed_data.get('pwp', [])[0])
    FieldCapacityUpper = int(parsed_data.get('fcu', [])[0])
    FieldCapacityLower = int(parsed_data.get('fcl', [])[0])
    Saturation = int(parsed_data.get('sat', [])[0]) 

    # Get soil water retention curve
    Soil_water_retention_curve = parsed_data.get('ret', [])[0]

    # Create a CSV file-like object from the CSV string
    csv_file = StringIO(Soil_water_retention_curve)

    # Parse the CSV data into a list of dictionaries
    csv_data = []
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        csv_data.append(row)

    # Organize the variables into a dictionary
    data = {
        "DeviceAndSensorIdsMoisture": DeviceAndSensorIdsMoisture,
        "DeviceAndSensorIdsTemp": DeviceAndSensorIdsTemp,
        "DeviceAndSensorIdsFlow": DeviceAndSensorIdsFlow,
        "Sensor_kind" : Sensor_kind,
        "Gps_info": {"lattitude": Gps_info.split(',')[0].lstrip(), "longitude": Gps_info.split(',')[1].lstrip()},
        "Slope": Slope,
        "Threshold": Threshold,
        "Irrigation_amount": Irrigation_amount,
        "Look_ahead_time": Look_ahead_time,
        "Start_date": Start_date,
        "Period": Period,
        "Soil_type": Soil_type,
        "Soil_water_retention_curve": csv_data,  # Use the parsed CSV data
        "PermanentWiltingPoint": PermanentWiltingPoint,
        "FieldCapacityUpper": FieldCapacityUpper,
        "FieldCapacityLower": FieldCapacityLower,
        "Saturation": Saturation
    }

    # Save the JSON data to the file
    with open(ConfigPath, 'w') as json_file:
        json.dump(data, json_file, indent=4)


    return 200, b"Configuration has been successfully saved!", []

usock.routerPOST("/api/setConfig", setConfig)

def getConfigFromFile():
    global DeviceAndSensorIdsMoisture
    global DeviceAndSensorIdsTemp
    global DeviceAndSensorIdsFlow
    global Sensor_kind
    global Gps_info
    global Slope
    global Irrigation_amount
    global Look_ahead_time
    global Threshold
    global Start_date
    global Period
    global Soil_type
    global Soil_water_retention_curve
    global PermanentWiltingPoint
    global FieldCapacityUpper
    global FieldCapacityLower
    global Saturation
    global Sensor_unit


    if os.path.exists(ConfigPath):
        with open(ConfigPath, 'r') as file:
            # Parse JSON from the file
            data = json.load(file)

        # Get choosen sensors
        DeviceAndSensorIdsMoisture = data.get('DeviceAndSensorIdsMoisture', [])
        DeviceAndSensorIdsTemp = data.get('DeviceAndSensorIdsTemp', [])
        DeviceAndSensorIdsFlow = data.get('DeviceAndSensorIdsFlow', [])

        # Get data from forms
        Sensor_kind = data.get('Sensor_kind', [])
        Gps_info = data.get('Gps_info', [])
        Slope = float(data.get('Slope', []))
        Threshold = float(data.get('Threshold', []))
        Irrigation_amount = float(data.get('Irrigation_amount', []))
        Look_ahead_time = float(data.get('Look_ahead_time', []))
        Start_date = data.get('Start_date', [])
        Period = int(data.get('Period', []))
        Soil_type = data.get('Soil_type', [])
        PermanentWiltingPoint = float(data.get('PermanentWiltingPoint', []))
        FieldCapacityUpper = float(data.get('FieldCapacityUpper', []))
        FieldCapacityLower = float(data.get('FieldCapacityLower', []))
        Saturation = float(data.get('Saturation', []))

        # Get soil water retention curve -> currently not needed here
        Soil_water_retention_curve = data.get('Soil_water_retention_curve', [])

        if Sensor_kind == "tension":
            Sensor_unit = "Moisture in cbar (Soil Tension)"
        elif Sensor_kind == "capacitive":
            Sensor_unit = "Moisture in % (Volumetric Water Content)"
        else :
            Sensor_unit = "Unit is unknown"

        return True
    else:
        return False

# Get the config from backend to disply it in frontend settings.html
def returnConfig(url, body):
    try:
        # Call the getConfigFromFile function to load variables
        if getConfigFromFile():

            # Check if all necessary global variables are properly defined
            if not all(isinstance(var, (int, float, str, list, dict)) for var in [
                DeviceAndSensorIdsMoisture, DeviceAndSensorIdsTemp, DeviceAndSensorIdsFlow, 
                Sensor_kind, Gps_info, Slope, Threshold, Irrigation_amount, Look_ahead_time, 
                Start_date, Period, PermanentWiltingPoint, FieldCapacityUpper, 
                FieldCapacityLower, Saturation]):
                raise ValueError("Variables are still missing or of incorrect type after loading from config.")
        
            # Construct the response data
            response_data = {
                "DeviceAndSensorIdsMoisture": DeviceAndSensorIdsMoisture,
                "DeviceAndSensorIdsTemp": DeviceAndSensorIdsTemp,
                "DeviceAndSensorIdsFlow": DeviceAndSensorIdsFlow,
                "Sensor_kind": Sensor_kind,
                "Gps_info": Gps_info,
                "Slope": Slope,
                "Threshold": Threshold,
                "Irrigation_amount": Irrigation_amount,
                "Look_ahead_time": Look_ahead_time,
                "Start_date": Start_date,
                "Period": Period,
                "Soil_type": Soil_type,
                "Soil_water_retention_curve": Soil_water_retention_curve,
                "PermanentWiltingPoint": PermanentWiltingPoint,
                "FieldCapacityUpper": FieldCapacityUpper,
                "FieldCapacityLower": FieldCapacityLower,
                "Saturation": Saturation
            }

            # If all is good, return a 200 status code and the data
            response = {
                "data": response_data,
                "status_code": 200
            }
            return 200, bytes(json.dumps(response), "utf8"), []
        else:
            error_response = {
                "error": "No config data present. Perform configuration.",
                "status_code": 400
            }
            return 400, bytes(json.dumps(error_response), "utf8"), []

    except ValueError as ve:
        # Return a 400 error for missing or invalid data
        error_response = {
            "error": str(ve),
            "status_code": 400
        }
        return 400, bytes(json.dumps(error_response), "utf8"), []

    except Exception as e:
        # Return a 500 error for any other internal server error
        error_response = {
            "error": "An unexpected error occurred: " + str(e),
            "status_code": 500
        }
        return 500, bytes(json.dumps(error_response), "utf8"), []
    
usock.routerGET("/api/returnConfig", returnConfig)

def checkConfigPresent(url, body):
    if os.path.exists(ConfigPath):
        response_data = {"config_present": True}
        status_code = 200
        getConfigFromFile()
    else:
        response_data = {"config_present": False}
        status_code = 404

    response = {
        "data": response_data,
        "status_code": status_code
    }

    return status_code, bytes(json.dumps(response), "utf8"), []

usock.routerGET("/api/checkConfigPresent", checkConfigPresent)

def checkActiveIrrigation(url, body):
    # Called on page load->important for checkActiveIrrigation
    if not getConfigFromFile():
        response_data = {"activeIrrigation": False}
        status_code = 404

    if len(DeviceAndSensorIdsFlow) != 0:
        response_data = {"activeIrrigation": True}
        status_code = 200
    else:
        response_data = {"activeIrrigation": False}
        status_code = 404

    response = {
        "data": response_data,
        "status_code": status_code
    }

    return status_code, bytes(json.dumps(response), "utf8"), []

usock.routerGET("/api/checkActiveIrrigation", checkActiveIrrigation)

# From key-value to series
def extract_and_format(data, key, datatype):
    values = []
    for items in data:
        for item in items:
            if datatype == "str":
                values.append(str(item[key]))
            elif datatype == "float":
                values.append(float(item[key]))
    
    return values

def irrigateManually(url, body):
    # Parse the query parameters from the URL
    query_params = parse_qs(urlparse(url).query)

    # Extract the 'amount' parameter (assuming it's passed as a query parameter)
    amount = int(query_params.get('amount', [0])[0])

    # Call the actuation function with the extracted amount
    response = actuation.irrigate_amount(amount)

    return 200, bytes(json.dumps({"status": "success", "amount": amount, "response": response}), "utf8"), []
    
usock.routerGET("/api/irrigateManually", irrigateManually)

def getValuesForDashboard(url, body):
    data_moisture = []
    data_temp = []

    for temp in DeviceAndSensorIdsTemp:
        data_temp.append(create_model.load_latest_data_api(temp, "sensors"))
    for moisture in DeviceAndSensorIdsMoisture:
        data_moisture.append(create_model.load_latest_data_api(moisture, "sensors"))

    # Calculate the temp average
    temp_average = sum(data_temp) / len(data_temp)
    # Calculate the moisture average
    moisture_average = sum(data_moisture) / len(data_moisture)
    # Calculate the VVO average if tension sensor is used
    if Sensor_kind == "tension":
        vwc_average = round(create_model.calc_volumetric_water_content_single_value(moisture_average)*100,2) 

        dashboard_data = {
            "temp_average": temp_average,
            "moisture_average": moisture_average,
            "vwc_average": vwc_average
        }
    else:
        dashboard_data = {
            "temp_average": temp_average,
            "moisture_average": "-- (Sensor not present)",
            "vwc_average": moisture_average
        }
    
    return 200, bytes(json.dumps(dashboard_data), "utf8"), []

usock.routerGET("/api/getValuesForDashboard", getValuesForDashboard)


def getHistoricalChartData(url, body): 
    # Load data from local wazigate api -> each sensor individually
    data_moisture = []
    data_temp = []
    #data_flow = [] # TODO:later also show flow in vis

    for moisture in DeviceAndSensorIdsMoisture:
        data_moisture.append(create_model.load_data_api(moisture, "sensors", Start_date))
    for temp in DeviceAndSensorIdsTemp:
        data_temp.append(create_model.load_data_api(temp, "sensors", Start_date))
    # for flow in DeviceAndSensorIdsFlow:
    #     data_flow.append(create_model.load_data_api(flow, "actuators", Start_date))
    
    # extract series from key value pairs
    f_data_time = extract_and_format(data_moisture, "time", "str")
    f_data_moisture = extract_and_format(data_moisture, "value", "float")
    f_data_temp = extract_and_format(data_temp, "value", "float")

    # Create the chart_data dictionary
    chart_data = {
        "timestamps": f_data_time,
        "temperatureSeries": f_data_temp,
        "moistureSeries": f_data_moisture,
        "unit": Sensor_unit
    }

    return 200, bytes(json.dumps(chart_data), "utf8"), []

usock.routerGET("/api/getHistoricalChartData", getHistoricalChartData)

# get values train + testset and display all elements -> stupid
def getDatasetChartData(url, body):
    data_dataset = create_model.get_Data()

    if data_dataset is False:
        response_data = {"model": False}
        status_code = 404

        return status_code, bytes(json.dumps(response_data), "utf8"), []
    
    # Conversion of dataframe to series 
    f_data_time = []
    items_to_render = []
    #data_dataset.set_index('Timestamp', inplace=True)
    col_names = data_dataset.columns

    # index
    for item in data_dataset.index:
        f_data_time.append(item.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S')) #TODO:timezone is lost here!!!

    # other cols
    for col in col_names:
        if data_dataset[col].dtype == "float64" or data_dataset[col].dtype == "int64":
            items_to_render.append(data_dataset[col].tolist())
        else:
            print("Missed the col:", col)

    # Create the chart_data dictionary => has to be created in a loop
    chart_data = {
        "timestamps": f_data_time,
    }
    # Add other cols 
    for i in range(0, len(items_to_render)):
        chart_data[col_names[i]] = items_to_render[i]

    return 200, bytes(json.dumps(chart_data), "utf8"), []

usock.routerGET("/api/getDatasetChartData", getDatasetChartData)


# get values from create_model.py if models had been trained
def getPredictionChartData(url, body): 
    data_pred = create_model.get_predictions()

    if data_pred is False:
        response_data = {"model": False}
        status_code = 404

        return status_code, bytes(json.dumps(response_data), "utf8"), []

    # Extract specific columns into lists TODO: timezone lost here!!!
    f_data_time = []
    #f_data_time = data_pred.index.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S%z').tolist()=>love python
    for item in data_pred.index:
        f_data_time.append(item.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S%z'))
    f_data_moisture = data_pred["smoothed_values"].tolist()

    adjustment = 1
    adjust_threshold = lambda Threshold, adjustment: Threshold - adjustment if Sensor_kind == "tension" else Threshold + adjustment

    # Add a horizontal line at Threshold # TODO: add direction of gradient
    annotations = {
        'yaxis': [
            {
                'y': Threshold,
                'y2': adjust_threshold(Threshold, adjustment),
                'borderColor': '#FF4560',
                'fillColor': '#FF4560',
                'opacity': 0.25,
                'fillPattern': {
                    'style': 'slantedLines',
                    'width': 4,
                    'height': 4,
                    'strokeWidth': 1
                },
                'label': {
                    'borderColor': '#FF4560',
                    'style': {
                        'color': '#fff',
                        'background': '#FF4560'
                    },
                }
            },
            {
                # Line annotation at Threshold
                'y': Threshold,
                'borderColor': '#FF4560',
                'strokeDashArray': 0,
                'borderWidth': 2,
                'label': {
                    'borderColor': '#FF4560',
                    'style': {
                        'color': '#fff',
                        'background': '#FF4560'
                    },
                    'text': 'Threshold for irrigation'
                }
            }
        ]
    }

    # Create the chart_data dictionary
    chart_data = {
        "timestamps": f_data_time,
        "moistureSeries": f_data_moisture,
        "annotations": annotations, # Could be also just the value instead of annotations object
        "permanentWiltingPoint": PermanentWiltingPoint,
        "fieldCapacityUpper": FieldCapacityUpper,
        "fieldCapacityLower": FieldCapacityLower,
        "saturation": Saturation,
        "kind": Sensor_kind,
        "unit": Sensor_unit
    }

    # Conditionally add 'moistureSeriesVol' if available
    if Sensor_kind == 'tension':# and 'f_data_moisture_vol' in locals() and f_data_moisture_vol is not None:
        f_data_moisture_vol = data_pred["smoothed_values_vol"].tolist()
        chart_data["moistureSeriesVol"] = f_data_moisture_vol

    return 200, bytes(json.dumps(chart_data), "utf8"), []

usock.routerGET("/api/getPredictionChartData", getPredictionChartData)

# get values from create_model.py if models had been trained
def getThreshold(url, body): 
    threshold_timestamp = create_model.get_threshold_timestamp()

    if threshold_timestamp is False:
        response_data = {"threshold": False}
        status_code = 404

        return status_code, bytes(json.dumps(response_data), "utf8"), []
    
    else:
        timestamp_data = {
            "timestamp": str(threshold_timestamp)
        }

        return 200, bytes(json.dumps(timestamp_data), "utf-8"), []
    
usock.routerGET("/api/getThreshold", getThreshold)

# Returns the senors kind: e.g. capacitive or tension
def getSensorKind(url, body):
    response_data = {"SensorKind": Sensor_kind}
    
    return 200, bytes(json.dumps(response_data), "utf8"), []

usock.routerGET("/api/getSensorKind", getSensorKind)

# Frontend polls this to reload page when training is ready => only active for first round of training
def isTrainingReady(url, body):
    response_data = {"isTrainingFinished": TrainingFinished}

    return 200, bytes(json.dumps(response_data), "utf8"), []

usock.routerGET("/api/isTrainingReady", isTrainingReady)

# surveillance, check threads are running
def check_threads():
    if not Training_thread or not Training_thread.is_alive():
        print("Training thread not alive, restarting...")
        startTraining(url=None, body=None)

    if not Prediction_thread or not Prediction_thread.is_alive():
        print("Prediction thread not alive, restarting...")
        startPrediction()

# Thread that runs prediction
def workerToPredict():
    def time_until_n_hours(hours):
        """Calculate the time difference from now until the next noon."""
        now = datetime.now()
        predict_time = now + timedelta(hours=hours, minutes=0, seconds=0, microseconds=0) #TODO: change to hours DEBUG

        return (predict_time - now).total_seconds()
    
    # Initial waiting, after model was trained, prediction was conducted and actuation was triggered 
    time_to_sleep = time_until_n_hours(Predict_period_hours)
    print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
    time.sleep(time_to_sleep)  # Sleep until threshold
    
    while True:
        try:
            start_time = datetime.now().replace(microsecond=0)
            print("Prediction started at:", start_time)

            file_path = pathlib.Path('saved_variables.pkl')

            if Perform_training: #same var is used here to preserve functionality
                # Call predict_with_updated_data function
                currentSoilTension, threshold_timestamp, predictions = create_model.predict_with_updated_data()

                # Create object to save
                variables_to_save = {
                    'currentSoilTension': currentSoilTension,
                    'threshold_timestamp': threshold_timestamp,
                    'predictions': predictions
                }
                # Save the variables to a file
                with open(file_path, 'wb') as f:
                    pickle.dump(variables_to_save, f)
            else:
                # Load the saved variables from the file
                with open(file_path, 'rb') as f:
                    loaded_variables = pickle.load(f)
                currentSoilTension = loaded_variables['currentSoilTension']
                threshold_timestamp = loaded_variables['threshold_timestamp']
                predictions = loaded_variables['predictions']

            end_time = datetime.now().replace(microsecond=0)
            duration = end_time - start_time
            print("Prediction finished at: ", end_time, "The duration was: ", duration)

            # Call routine to irrigate
            if len(DeviceAndSensorIdsFlow) > 0: 
                actuation.main(currentSoilTension, threshold_timestamp, predictions, Irrigation_amount, Sensor_kind)

            # After initial training and prediction, start surveillance
            threading.Timer(3600, check_threads).start()  # Check every hour if threads are alive

            # Wait for Predict_period_hours periodically for next cycle
            time_to_sleep = time_until_n_hours(Predict_period_hours)
            print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
            time.sleep(time_to_sleep)  # Sleep until threshold
        except Exception as e:
            print(f"Prediction thread error: {e}. Retrying after 30 minute.")
            time.sleep(Restart_time)  # Retry after 30 minute if there is an error

# Starts a thread that runs prediction
def startPrediction():
    global ThreadId
    global Prediction_thread

    if not CurrentlyTraining:
        # Create a new thread for training
        Prediction_thread = threading.Thread(target=workerToPredict)
        ThreadId += 1

        # Append thread to list
        Threads.append(Prediction_thread)

        # Start the thread
        Prediction_thread.start()

    
# Thread that runs training
def workerToTrain(thread_id, url, startTrainingNow):
    global TrainingFinished
    global CurrentlyTraining

    def time_until_noon(train_period_days):
        """Calculate the time difference from now until the next noon."""
        now = datetime.now()
        noon_today = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now >= noon_today:
            # If it's already past noon, calculate for the next day
            noon_today += timedelta(days=train_period_days)
        return (noon_today - now).total_seconds()

    while True:
        try:
            if not startTrainingNow:
                # Wait until the next noon
                time_to_sleep = time_until_noon(Train_period_days)
                print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until next training...")
                time.sleep(time_to_sleep)  # Sleep until noon

            start_time = datetime.now().replace(microsecond=0)
            print("Training started at:", start_time)

            file_path = pathlib.Path('saved_variables.pkl')

            if Perform_training:
                # Call create model function
                currentSoilTension, threshold_timestamp, predictions = create_model.main()

                # Create object to save
                variables_to_save = {
                    'currentSoilTension': currentSoilTension,
                    'threshold_timestamp': threshold_timestamp,
                    'predictions': predictions
                }
                # Save the variables to a file
                with open(file_path, 'wb') as f:
                    pickle.dump(variables_to_save, f)
            else:
                # Load the saved variables from the file
                with open(file_path, 'rb') as f:
                    loaded_variables = pickle.load(f)
                currentSoilTension = loaded_variables['currentSoilTension']
                threshold_timestamp = loaded_variables['threshold_timestamp']
                predictions = loaded_variables['predictions']

            TrainingFinished = True
            CurrentlyTraining = False
            startTrainingNow = False

            end_time = datetime.now().replace(microsecond=0)
            duration = end_time - start_time
            print("Training finished at: ", end_time, "The duration was: ", duration)

            # Call routine to irrigate
            if len(DeviceAndSensorIdsFlow) > 0: 
                actuation.main(currentSoilTension, threshold_timestamp, predictions, Irrigation_amount, Sensor_kind)

            # Start thread that creates predictions periodically
            if Prediction_thread is None:
                startPrediction()
            # if not Prediction_thread.is_alive():
            #     startPrediction()
        except Exception as e:
            print(f"Training error: {e}. Retrying after 30 minute.")
            time.sleep(Restart_time)  # Retry after 30 minute if there is an error

# Starts a thread that runs training
def startTraining(url, body):
    global ThreadId
    global TrainingFinished
    global CurrentlyTraining
    global Training_thread

    if not CurrentlyTraining:
        # Stop/kill all other threads for training a model
        if Training_thread is not None:
            Training_thread.terminate()

        # Reset flags for a new round
        TrainingFinished = False
        CurrentlyTraining = True

        # Create a new thread for training
        Training_thread = threading.Thread(target=workerToTrain, args=(ThreadId, url, True))
        ThreadId += 1

        # Append thread to list
        Threads.append(Training_thread)

        # Start the thread
        Training_thread.start()

    return 200, b"", []

usock.routerGET("/api/startTraining", startTraining)

#------------------#


if __name__ == "__main__":
    # Load variables
    load_dotenv()

    # Start serving
    usock.sockAddr = os.getenv("Proxy_URL")
    usock.start()

    # Start thread that deletes old models
    folder_to_check = "models"
    schedule_model_cleanup(folder_to_check, interval_days=7)  # Check every week

    # Clean logs
    schedule_log_cleanup()