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

# Slope to evaluate irrgation has taken place
Slope = 0

# Threshold to irrigate plants
Threshold = 0

# Start date
Start_date = ""

# Time period to include into the model
Period = 0

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

# Array of active threads
Threads = []
ThreadId = 0

TrainingFinished = False
CurrentlyTraining = False

# Load variables of training from file, to debugactuation part
Perform_training = False

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
    global Gps_info
    global Slope
    global Threshold
    global Start_date
    global Period

    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    # Get choosen sensors
    DeviceAndSensorIdsMoisture = parsed_data.get('selectedOptionsMoisture', [])
    DeviceAndSensorIdsTemp = parsed_data.get('selectedOptionsTemp', [])
    DeviceAndSensorIdsFlow = parsed_data.get('selectedOptionsFlow', [])

    # Get data from forms
    Gps_info = parsed_data.get('gps', [])[0]
    Slope = parsed_data.get('slope', [])[0]
    Threshold = float(parsed_data.get('thres', [])[0])
    Start_date = parsed_data.get('start', [])[0]
    Period = int(parsed_data.get('period', [])[0])

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
        "Gps_info": {"lattitude": Gps_info.split(',')[0].lstrip(), "longitude": Gps_info.split(',')[1].lstrip()},
        "Slope": Slope,
        "Threshold": Threshold,
        "Start_date": Start_date,
        "Period": Period,
        "Soil_water_retention_curve": csv_data  # Use the parsed CSV data
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
    global Gps_info
    global Slope
    global Threshold
    global Start_date
    global Period

    with open(ConfigPath, 'r') as file:
        # Parse JSON from the file
        data = json.load(file)

    # Get choosen sensors
    DeviceAndSensorIdsMoisture = data.get('DeviceAndSensorIdsMoisture', [])
    DeviceAndSensorIdsTemp = data.get('DeviceAndSensorIdsTemp', [])
    DeviceAndSensorIdsFlow = data.get('DeviceAndSensorIdsFlow', [])

    # Get data from forms
    Gps_info = data.get('Gps_info', [])
    Slope = float(data.get('Slope', []))
    Threshold = float(data.get('Threshold', []))
    Start_date = data.get('Start_date', [])
    Period = int(data.get('Period', []))

    # Get soil water retention curve -> currently not needed here
    # Soil_water_retention_curve = data.get('Soil_water_retention_curve', [])


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


def getHistoricalChartData(url, body): 
    # Load data from local wazigate api -> each sensor individually
    data_moisture = []
    data_temp = []
    #data_flow = [] # later also show flow in vis

    for moisture in DeviceAndSensorIdsMoisture:
        data_moisture.append(create_model.load_data_api(moisture, Start_date))
    for temp in DeviceAndSensorIdsTemp:
        data_temp.append(create_model.load_data_api(temp, Start_date))
    # for flow in DeviceAndSensorIdsFlow:
    #     data_flow.append(create_model.load_data_api(flow, Start_date))
    
    # extract series from key value pairs
    f_data_time = extract_and_format(data_moisture, "time", "str")
    f_data_moisture = extract_and_format(data_moisture, "value", "float")
    f_data_temp = extract_and_format(data_temp, "value", "float")

    # Create the chart_data dictionary
    chart_data = {
        "timestamps": f_data_time,
        "temperatureSeries": f_data_temp,
        "moistureSeries": f_data_moisture
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
    f_data_moisture = data_pred["prediction_label"].tolist()
    f_data_moisture_vol = data_pred["prediction_label_vol"].tolist()

    # Add a horizontal line at Threshold
    annotations = {
        'yaxis': [{
            'y': Threshold,
            'borderColor': '#FF4560',
            'label': {
                'borderColor': '#FF4560',
                'style': {
                    'color': '#fff',
                    'background': '#FF4560'
                },
                'text': 'Threshold'
            }
        }]
    }

    # Create the chart_data dictionary
    chart_data = {
        "timestamps": f_data_time,
        "moistureSeries": f_data_moisture,
        "moistureSeriesVol": f_data_moisture_vol,
        "annotations": annotations
    }

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

# Frontend polls this to reload page when training is ready => only active for first round of training
def isTrainingReady(url, body):
    response_data = {"isTrainingFinished": TrainingFinished}

    return 200, bytes(json.dumps(response_data), "utf8"), []

usock.routerGET("/api/isTrainingReady", isTrainingReady)
    

def workerToTrain(thread_id, url): # TODO: do we really need threading here?
    global TrainingFinished
    global CurrentlyTraining

    # Set the time interval in seconds (e.g., 60 seconds for 1 minute)
    time_interval = 43200 #12h

    while True:
        start_time = datetime.now().replace(microsecond=0)
        print("Training started at:", start_time)

        file_path = pathlib.Path('saved_variables.pkl')

        if (Perform_training):
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


        # TODO: reload page
        TrainingFinished = True
        CurrentlyTraining = False

        end_time = datetime.now().replace(microsecond=0)
        duration = end_time - start_time
        print("Traning finished at: ", end_time, "The duration was: ", duration)

        # Call routine to irrgate
        actuation.main(currentSoilTension, threshold_timestamp, predictions)

        # Send thread to sleep
        time.sleep(time_interval)  # Wait for the specified time interval

def startTraining(url, body):
    global ThreadId
    global TrainingFinished
    global CurrentlyTraining

    if not CurrentlyTraining:
        # Switch off for 2nd, ... round
        TrainingFinished = False
        CurrentlyTraining = True

        # Create a new thread
        thread = threading.Thread(target=workerToTrain, args=(ThreadId, url))    
        ThreadId += 1

        # Append thread to list
        Threads.append(thread)

        # Start the thread
        thread.start()

    return 200, b"", []

usock.routerGET("/api/startTraining", startTraining)

#------------------#


if __name__ == "__main__":
    load_dotenv()
    usock.sockAddr = os.getenv("Proxy_URL")
    usock.start()
