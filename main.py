# TODO: delete ports from URLs


#!/usr/bin/python
import csv
from datetime import datetime, timedelta
from io import StringIO
import json
import pickle
import re
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
from plot import Plot 
import plot_manager
from utils import NetworkUtils



#---------------------#
# Path for proxy.sock, set by .env
#usock.sockAddr = ""

# Path to the root of the code
PATH = os.path.dirname(os.path.abspath(__file__))

# Array of active threads TODO: if training started kill other threads.
Threads = []
ThreadId = 0
Restart_time = 1800 # DEBUG 1800 ~ 30 min in s

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

# setup function for log cleaner => TODO: changed function WITHOUT TESTING IT!!!!!!!!!!!!!!!!!
def schedule_log_cleanup():
    logs_to_clean = [
        ("logs.log", 90),       # Python log file
        ("python_logs.log", 90) # Docker log file
    ]

    # Start a thread for each log file
    for log_path, age_limit in logs_to_clean:
        cleaner = LogCleanerThread(file_path=log_path, age_limit_days=age_limit)
        cleaner.daemon = True  # Run thread in the background
        cleaner.start()


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
    url = NetworkUtils.ApiUrl

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

# Currently choosen plot in UI
def setPlot(url, body):
    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    # Get currentPlot
    currentTab = int(parsed_data.get('currentPlot', [])[0])

    if(plot_manager.setPlot(currentTab)):
            return 200, b"Plot has been set.", []
    else:
        return 200, b"Has been set but has no config yet.", []

usock.routerPOST("/api/setPlot", setPlot)

# When Page is reloaded it will return formerly loaded plots
def getPlots(url, body):
    # Call function in plot manager
    plots = plot_manager.getPlots()

    # Create array with names of tabs to return to frontend
    tab_name_array = []
    #for plot in plots:
    for i in range(1, len(plots)+1, 1):
        tab_name_array.append(plots[i].user_given_name)

    response = {
        "tabnames": tab_name_array,
        "status_code": 200
    }

    return response["status_code"], bytes(json.dumps(response), "utf8"), []

usock.routerGET("/api/getPlots", getPlots)

# Add a plot during runtine TODO: finish
def addPlot(url, body):
    # Call function in plot manager
    next_number, newfilename = plot_manager.addPlot()

    response = {
        "plot_number": next_number,
        "filename": newfilename,
        "status_code": 200
    }

    return response["status_code"], bytes(json.dumps(response), "utf8"), []

usock.routerGET("/api/addPlot", addPlot)

# Delete a plot during runtine TODO: ids adjust on remove, API call
def removePlot(url, body):
    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))
    plot_to_be_removed = int(parsed_data.get('currentPlot', [])[0])

    # Call function in plot manager
    removed_plot_id, oldfilename = plot_manager.removePlot(plot_to_be_removed)

    response = {
        "plot_number": removed_plot_id,
        "filename": oldfilename,
        "status_code": 200
    }

    return response["status_code"], bytes(json.dumps(response), "utf8"), []

usock.routerGET("/api/addPlot", addPlot)

# Get historical sensor values from WaziGates API
def setConfig(url, body):
    # Get current plot
    currentPlot = plot_manager.getCurrentPlot()

    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    # Get choosen sensors
    currentPlot.device_and_sensor_ids_moisture = parsed_data.get('selectedOptionsMoisture', [])
    currentPlot.device_and_sensor_ids_temp = parsed_data.get('selectedOptionsTemp', [])
    currentPlot.device_and_sensor_ids_flow = parsed_data.get('selectedOptionsFlow', [])

    # Get data from forms
    name_list = parsed_data.get('name', [])
    currentPlot.user_given_name = name_list[0] if name_list else ""
    currentPlot.sensor_kind = parsed_data.get('sensor_kind', [])[0]
    currentPlot.gps_info = parsed_data.get('gps', [])[0]
    currentPlot.slope = parsed_data.get('slope', [])[0]
    currentPlot.threshold = float(parsed_data.get('thres', [])[0])
    currentPlot.irrigation_amount = float(parsed_data.get('amount', [])[0])
    currentPlot.look_ahead_time = float(parsed_data.get('lookahead', [])[0])
    currentPlot.start_date = parsed_data.get('start', [])[0]
    currentPlot.period = int(parsed_data.get('period', [])[0])
    currentPlot.soil_type = parsed_data.get('soil', [])[0]
    currentPlot.permanent_wilting_point = int(parsed_data.get('pwp', [])[0])
    currentPlot.field_capacity_upper = int(parsed_data.get('fcu', [])[0])
    currentPlot.field_capacity_lower = int(parsed_data.get('fcl', [])[0])
    currentPlot.saturation = int(parsed_data.get('sat', [])[0]) 

    # Get soil water retention curve
    currentPlot.soil_water_retention_curve = parsed_data.get('ret', [])[0]

    # Create a CSV file-like object from the CSV string
    csv_file = StringIO(currentPlot.soil_water_retention_curve)

    # Parse the CSV data into a list of dictionaries
    csv_data = []
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        csv_data.append(row)

    # Organize the variables into a dictionary
    data = {
        "DeviceAndSensorIdsMoisture": currentPlot.device_and_sensor_ids_moisture,
        "DeviceAndSensorIdsTemp": currentPlot.device_and_sensor_ids_temp,
        "DeviceAndSensorIdsFlow": currentPlot.device_and_sensor_ids_flow,
        "Sensor_kind" : currentPlot.sensor_kind,
        "Name": currentPlot.user_given_name,
        #"Gps_info": {"lattitude": currentPlot.gps_info['lattitude'], "longitude": currentPlot.gps_info['longitude']},
        #"Gps_info": currentPlot.gps_info,
        "Gps_info": {
            "lattitude": currentPlot.gps_info.split(", ")[0], 
            "longitude": currentPlot.gps_info.split(", ")[1]
        },
        "Slope": currentPlot.slope,
        "Threshold": currentPlot.threshold,
        "Irrigation_amount": currentPlot.irrigation_amount,
        "Look_ahead_time": currentPlot.look_ahead_time,
        "Start_date": currentPlot.start_date,
        "Period": currentPlot.period,
        "Soil_type": currentPlot.soil_type,
        "Soil_water_retention_curve": csv_data,  # Use the parsed CSV data
        "PermanentWiltingPoint": currentPlot.permanent_wilting_point,
        "FieldCapacityUpper": currentPlot.field_capacity_upper,
        "FieldCapacityLower": currentPlot.field_capacity_lower,
        "Saturation": currentPlot.saturation
    }

    # Save the JSON data to the file
    with open(plot_manager.getCurrentConfig(), 'w') as json_file:
        json.dump(data, json_file, indent=4)


    return 200, b"Configuration has been successfully saved!", []

usock.routerPOST("/api/setConfig", setConfig)

# Load config from file
def getConfigFromFile():
    # Get currentPlot and path
    currentPlot = plot_manager.getCurrentPlot()
    currentConfigPath = plot_manager.getCurrentConfig()

    if os.path.exists(currentConfigPath):
        with open(currentConfigPath, 'r') as file:
            # Parse JSON from the file
            data = json.load(file)

        # Get choosen sensors
        currentPlot.device_and_sensor_ids_moisture = data.get('DeviceAndSensorIdsMoisture', [])
        currentPlot.device_and_sensor_ids_temp = data.get('DeviceAndSensorIdsTemp', [])
        currentPlot.device_and_sensor_ids_flow = data.get('DeviceAndSensorIdsFlow', [])

        # Get data from forms
        currentPlot.user_given_name = data.get('Name', [])
        currentPlot.sensor_kind = data.get('Sensor_kind', [])
        currentPlot.gps_info = data.get('Gps_info', [])
        currentPlot.slope = float(data.get('Slope', []))
        currentPlot.threshold = float(data.get('Threshold', []))
        currentPlot.irrigation_amount = float(data.get('Irrigation_amount', []))
        currentPlot.look_ahead_time = float(data.get('Look_ahead_time', []))
        currentPlot.start_date = data.get('Start_date', [])
        currentPlot.period = int(data.get('Period', []))
        currentPlot.soil_type = data.get('Soil_type', [])
        currentPlot.permanent_wilting_point = float(data.get('PermanentWiltingPoint', []))
        currentPlot.field_capacity_upper = float(data.get('FieldCapacityUpper', []))
        currentPlot.field_capacity_lower = float(data.get('FieldCapacityLower', []))
        currentPlot.saturation = float(data.get('Saturation', []))

        # Get soil water retention curve -> currently not needed here
        currentPlot.soil_water_retention_curve = data.get('Soil_water_retention_curve', [])

        if currentPlot.sensor_kind == "tension":
            currentPlot.sensor_unit = "Moisture in cbar (Soil Tension)"
        elif currentPlot.sensor_kind == "capacitive":
            currentPlot.sensor_unit = "Moisture in % (Volumetric Water Content)"
        else :
            currentPlot.sensor_unit = "Unit is unknown"

        return True
    else:
        return False
    
# Load config from all file
def getConfigsFromAllFiles():
    # Get plots
    plots = plot_manager.getPlots()

    for i in range(1,len(plots)+1,1):
        config = plots[i].configPath
        if os.path.exists(config):
            with open(config, 'r') as file:
                # Parse JSON from the file
                data = json.load(file)

            # Get choosen sensors
            plots[i].device_and_sensor_ids_moisture = data.get('DeviceAndSensorIdsMoisture', [])
            plots[i].device_and_sensor_ids_temp = data.get('DeviceAndSensorIdsTemp', [])
            plots[i].device_and_sensor_ids_flow = data.get('DeviceAndSensorIdsFlow', [])

            # Get data from forms
            plots[i].user_given_name = data.get('Name', [])
            plots[i].sensor_kind = data.get('Sensor_kind', [])
            plots[i].gps_info = data.get('Gps_info', [])
            plots[i].slope = float(data.get('Slope', []))
            plots[i].threshold = float(data.get('Threshold', []))
            plots[i].irrigation_amount = float(data.get('Irrigation_amount', []))
            plots[i].look_ahead_time = float(data.get('Look_ahead_time', []))
            plots[i].start_date = data.get('Start_date', [])
            plots[i].period = int(data.get('Period', []))
            plots[i].soil_type = data.get('Soil_type', [])
            plots[i].permanent_wilting_point = float(data.get('PermanentWiltingPoint', []))
            plots[i].field_capacity_upper = float(data.get('FieldCapacityUpper', []))
            plots[i].field_capacity_lower = float(data.get('FieldCapacityLower', []))
            plots[i].saturation = float(data.get('Saturation', []))

            # Get soil water retention curve -> currently not needed here
            plots[i].soil_water_retention_curve = data.get('Soil_water_retention_curve', [])

            if plots[i].sensor_kind == "tension":
                plots[i].sensor_unit = "Moisture in cbar (Soil Tension)"
            elif plots[i].sensor_kind == "capacitive":
                plots[i].sensor_unit = "Moisture in % (Volumetric Water Content)"
            else :
                plots[i].sensor_unit = "Unit is unknown"

# Get the config from backend to disply it in frontend settings.html
def returnConfig(url, body):
    try:
        # Call the getConfigFromFile function to load variables
        if getConfigFromFile():

            # Get currentPlot
            currentPlot = plot_manager.getCurrentPlot()

            # Check if all necessary global variables are properly defined
            if not all(isinstance(var, (int, float, str, list, dict)) for var in [
                currentPlot.device_and_sensor_ids_moisture, 
                currentPlot.device_and_sensor_ids_temp, 
                currentPlot.device_and_sensor_ids_flow, 
                currentPlot.sensor_kind, 
                currentPlot.name,
                currentPlot.gps_info, 
                currentPlot.slope, 
                currentPlot.threshold, 
                currentPlot.irrigation_amount, 
                currentPlot.look_ahead_time, 
                currentPlot.start_date, 
                currentPlot.period, 
                currentPlot.permanent_wilting_point, 
                currentPlot.field_capacity_upper, 
                currentPlot.field_capacity_lower, 
                currentPlot.saturation]):

                raise ValueError("Variables are still missing or of incorrect type after loading from config.")
        
            # Construct the response data
            response_data = {
                "DeviceAndSensorIdsMoisture": currentPlot.device_and_sensor_ids_moisture,
                "DeviceAndSensorIdsTemp": currentPlot.device_and_sensor_ids_temp,
                "DeviceAndSensorIdsFlow": currentPlot.device_and_sensor_ids_flow,
                "Sensor_kind": currentPlot.sensor_kind,
                "Name": currentPlot.name,
                "Gps_info": currentPlot.gps_info,
                "Slope": currentPlot.slope,
                "Threshold": currentPlot.threshold,
                "Irrigation_amount": currentPlot.irrigation_amount,
                "Look_ahead_time": currentPlot.look_ahead_time,
                "Start_date": currentPlot.start_date,
                "Period": currentPlot.period,
                "Soil_type": currentPlot.soil_type,
                "Soil_water_retention_curve": currentPlot.soil_water_retention_curve,
                "PermanentWiltingPoint": currentPlot.permanent_wilting_point,
                "FieldCapacityUpper": currentPlot.field_capacity_upper,
                "FieldCapacityLower": currentPlot.field_capacity_lower,
                "Saturation": currentPlot.saturation
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
    if os.path.exists(plot_manager.getCurrentConfig()): # solve multiple calls with dirty bit
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

# Called on page load->important for checkActiveIrrigation
def checkActiveIrrigation(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    if not getConfigFromFile():
        response_data = {"activeIrrigation": False}
        status_code = 404

    if len(currentPlot.device_and_sensor_ids_flow) != 0:
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

    currentPlot = plot_manager.getCurrentPlot()

    for temp in currentPlot.device_and_sensor_ids_temp:
        data_temp.append(currentPlot.load_latest_data_api(temp, "sensors"))
    for moisture in currentPlot.device_and_sensor_ids_moisture:
        data_moisture.append(currentPlot.load_latest_data_api(moisture, "sensors"))

    # Calculate the temp average
    temp_average = sum(data_temp) / len(data_temp)
    # Calculate the moisture average
    moisture_average = sum(data_moisture) / len(data_moisture)
    # Calculate the VVO average if tension sensor is used
    if currentPlot.sensor_kind == "tension":
        vwc_average = round(create_model.calc_volumetric_water_content_single_value(moisture_average, currentPlot)*100,2) 

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

    currentPlot = plot_manager.getCurrentPlot()

    for moisture in currentPlot.device_and_sensor_ids_moisture:
        data_moisture.append(currentPlot.load_data_api(moisture, "sensors", currentPlot.start_date))
    for temp in currentPlot.device_and_sensor_ids_temp:
        data_temp.append(currentPlot.load_data_api(temp, "sensors", currentPlot.start_date))
    # for flow in currentPlot.device_and_sensor_ids_flow: # TODO: maybe display that also here (is displayed in datasets data)
    #     data_flow.append(currentPlot.load_data_api(flow, "actuators", currentPlot.start_date))
    
    # extract series from key value pairs
    f_data_time = extract_and_format(data_moisture, "time", "str")
    f_data_moisture = extract_and_format(data_moisture, "value", "float")
    f_data_temp = extract_and_format(data_temp, "value", "float")

    # Create the chart_data dictionary
    chart_data = {
        "timestamps": f_data_time,
        "temperatureSeries": f_data_temp,
        "moistureSeries": f_data_moisture,
        "unit": currentPlot.sensor_unit
    }

    return 200, bytes(json.dumps(chart_data), "utf8"), []

usock.routerGET("/api/getHistoricalChartData", getHistoricalChartData)

# get values train + testset and display all elements -> stupid
def getDatasetChartData(url, body):
    # Get current plot (selected in UI)
    currentPlot = plot_manager.getCurrentPlot()

    # Get dataset data for chart of current plot
    data_dataset = currentPlot.get_Data_for_display(create_model.To_be_dropped)

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
    # Get current plot (selected in UI)
    currentPlot = plot_manager.getCurrentPlot()

    # Get prediction data for chart of current plot
    data_pred = currentPlot.get_predictions()

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

    # Quick and dirty adjusting predictions to match sensor values TODO: ??? right approach ???
    adjustment = 1
    #adjust_threshold = lambda currentPlot.threshold, adjustment: currentPlot.threshold - adjustment if currentPlot.sensor_kind == "tension" else currentPlot.threshold + adjustment
    adjust_threshold = lambda adjustment: currentPlot.threshold - adjustment if currentPlot.sensor_kind == "tension" else currentPlot.threshold + adjustment

    #  Add a horizontal line at Threshold
    annotations = {
        'yaxis': [
            {
                'y': currentPlot.threshold,
                'y2': adjust_threshold(adjustment),
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
                'y': currentPlot.threshold,
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
        "permanentWiltingPoint": currentPlot.permanent_wilting_point,
        "fieldCapacityUpper": currentPlot.field_capacity_upper,
        "fieldCapacityLower": currentPlot.field_capacity_lower,
        "saturation": currentPlot.saturation,
        "kind": currentPlot.sensor_kind,
        "unit": currentPlot.sensor_unit
    }

    # Conditionally add 'moistureSeriesVol' if available
    if currentPlot.sensor_kind == 'tension':# and 'f_data_moisture_vol' in locals() and f_data_moisture_vol is not None:
        f_data_moisture_vol = data_pred["smoothed_values_vol"].tolist()
        chart_data["moistureSeriesVol"] = f_data_moisture_vol

    return 200, bytes(json.dumps(chart_data), "utf8"), []

usock.routerGET("/api/getPredictionChartData", getPredictionChartData)

# get values from create_model.py if models had been trained
def getThreshold(url, body):
    # Get current plot (selected in UI)
    currentPlot = plot_manager.getCurrentPlot()

    # Get prediction data for chart of current plot
    threshold_timestamp = currentPlot.get_threshold_timestamp()

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
    response_data = {"SensorKind": plot_manager.getCurrentPlot().sensor_kind}
    
    return 200, bytes(json.dumps(response_data), "utf8"), []

usock.routerGET("/api/getSensorKind", getSensorKind)

# Frontend polls this to reload page when training is ready => only active for first round of training TODO: different plots
def isTrainingReady(url, body):
    response_data = {"isTrainingFinished": plot_manager.getCurrentPlot().training_finished}

    return 200, bytes(json.dumps(response_data), "utf8"), []

usock.routerGET("/api/isTrainingReady", isTrainingReady)

# surveillance, check threads are running TODO: different plots
def check_threads(plot):
    if not plot.training_thread or not plot.training_thread.is_alive():
        print("Training thread not alive, restarting...")
        startTraining(url=None, body=None)

    if not plot.prediction_thread or not plot.prediction_thread.is_alive():
        print("Prediction thread not alive, restarting...")
        startPrediction()

# Thread that runs prediction
def workerToPredict(plot):
    def time_until_n_hours(hours):
        """Calculate the time difference from now until the next noon."""
        now = datetime.now()
        predict_time = now + timedelta(hours=hours, minutes=0, seconds=0, microseconds=0) #TODO: change to hours DEBUG

        return (predict_time - now).total_seconds()
    
    # Initial waiting, after model was trained, prediction was conducted and actuation was triggered 
    time_to_sleep = time_until_n_hours(plot.predict_period_hours)
    print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
    time.sleep(time_to_sleep)  # Sleep until threshold
    
    while True:
        try:
            start_time = datetime.now().replace(microsecond=0)
            print("Prediction started at:", start_time)

            file_path = pathlib.Path('saved_variables.pkl')

            if Perform_training: #same var is used here to preserve functionality
                # Call predict_with_updated_data function
                currentSoilTension, plot.threshold_timestamp, plot.predictions = create_model.predict_with_updated_data(plot)

                # Create object to save
                variables_to_save = {
                    'currentSoilTension': currentSoilTension,
                    'threshold_timestamp': plot.threshold_timestamp,
                    'predictions': plot.predictions
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
                plot.predictions = loaded_variables['predictions']

            end_time = datetime.now().replace(microsecond=0)
            duration = end_time - start_time
            print("Prediction finished at: ", end_time, "The duration was: ", duration)

            # Call routine to irrigate
            if len(plot.device_and_sensor_ids_flow) > 0: 
                actuation.main(currentSoilTension, threshold_timestamp, plot.predictions, plot)

            # After initial training and prediction, start surveillance
            threading.Timer(3600, check_threads(plot)).start()  # Check every hour if threads are alive

            # Wait for predict_period_hours periodically for next cycle
            time_to_sleep = time_until_n_hours(plot.predict_period_hours)
            print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until conducting next prediction...")
            time.sleep(time_to_sleep)  # Sleep until threshold
        except Exception as e:
            print(f"Prediction thread error: {e}. Retrying after 30 minute.")
            time.sleep(Restart_time)  # Retry after 30 minute if there is an error

# Starts a thread that runs prediction
def startPrediction(plot):
    global ThreadId

    if not plot.currently_training:
        # Create a new thread for training
        plot.prediction_thread = threading.Thread(target=workerToPredict, args=(plot,))
        ThreadId += 1

        # Append thread to list
        Threads.append(plot.prediction_thread)

        # Start the thread
        plot.prediction_thread.start()

    
# Thread that runs training
def workerToTrain(thread_id, currentPlot, url, startTrainingNow):
    def time_until_noon(train_period_days):
        """Calculate the time difference from now until the next noon."""
        now = datetime.now()
        noon_today = now.replace(hour=12, minute=0, second=0, microsecond=0) # TODO: adjust time so that 
        if now >= noon_today:
            # If it's already past noon, calculate for the next day
            noon_today += timedelta(days=train_period_days)
        return (noon_today - now).total_seconds()

    while True:
        try:
            if not startTrainingNow:
                # Wait until the next noon
                time_to_sleep = time_until_noon(currentPlot.train_period_days)
                print(f"Waiting {time_to_sleep // 3600:.0f} hours {time_to_sleep % 3600 // 60:.0f} minutes until next training...")
                time.sleep(time_to_sleep)  # Sleep until noon

            start_time = datetime.now().replace(microsecond=0)
            print("Training for Plot with the name ", str(currentPlot.user_given_name)," started at:", start_time)

            file_path = pathlib.Path('saved_variables.pkl')

            if Perform_training:
                # Call create model function
                currentSoilTension, threshold_timestamp, currentPlot.predictions = create_model.main(currentPlot)

                # Create object to save
                variables_to_save = {
                    'currentSoilTension': currentSoilTension,
                    'threshold_timestamp': threshold_timestamp,
                    'predictions': currentPlot.predictions
                }
                # Save the variables to a file
                with open(file_path, 'wb') as f:
                    pickle.dump(variables_to_save, f)
            else:
                # Load the saved variables from the file
                with open(file_path, 'rb') as f:
                    loaded_variables = pickle.load(f)
                currentSoilTension = loaded_variables['currentSoilTension']
                currentPlot.threshold_timestamp = loaded_variables['threshold_timestamp']
                currentPlot.predictions = loaded_variables['predictions']

            currentPlot.training_finished = True
            currentPlot.currently_training = False
            startTrainingNow = False

            end_time = datetime.now().replace(microsecond=0)
            duration = end_time - start_time
            print("Training finished at: ", end_time, "The duration was: ", duration)

            # Call routine to irrigate TODO:plots
            if len(currentPlot.device_and_sensor_ids_flow) > 0: 
                actuation.main(currentSoilTension, currentPlot.threshold_timestamp, currentPlot.predictions, currentPlot)

            # Start thread that creates predictions periodically
            if currentPlot.prediction_thread is None:
                startPrediction(currentPlot)

        except Exception as e:
            print(f"Training error: {e}. Retrying after 30 minute.")
            time.sleep(Restart_time)  # Retry after 30 minute if there is an error

# Starts a thread that runs training
def startTraining(url, body):
    global ThreadId

    currentPlot = plot_manager.getCurrentPlot()

    if not currentPlot.currently_training:
        # Stop/kill all other threads for training a model
        if currentPlot.training_thread is not None:
            currentPlot.training_thread.terminate()

        # Reset flags for a new round
        currentPlot.training_finished = False
        currentPlot.currently_training = True

        # Create a new thread for training
        currentPlot.training_thread = threading.Thread(target=workerToTrain, args=(ThreadId, currentPlot, url, True))
        ThreadId += 1

        # Append thread to list
        Threads.append(currentPlot.training_thread)

        # Start the thread
        currentPlot.training_thread.start()

    return 200, b"", []

usock.routerGET("/api/startTraining", startTraining)

#------------------#


if __name__ == "__main__":
    # Load environment variables
    NetworkUtils.get_env()

    # Load all plots once on startup
    plot_manager.loadPlots()

    # Get saved config from all plots and save it in objects
    getConfigsFromAllFiles()

    # Start thread that deletes old models
    folder_to_check = "models"
    schedule_model_cleanup(folder_to_check, interval_days=7)  # Check every week

    # Clean logs
    schedule_log_cleanup()

    # Start serving
    usock.sockAddr = NetworkUtils.Proxy
    usock.start() # will be "stuck" in here, code afterwards is not executed