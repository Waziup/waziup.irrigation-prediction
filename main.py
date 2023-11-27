# TODO: delete ports from URLs


#!/usr/bin/python
import csv
from datetime import datetime, timedelta
from io import StringIO
import json
from urllib.parse import urlparse, parse_qs
import requests
import urllib
import usock
import os
import pathlib

import create_model



#---------------------#

#usock.sockAddr = "/var/lib/waziapp/proxy.sock" # Production mode

usock.sockAddr = "proxy.sock" # Debug mode

# URL of API to retrive devices
#DeviceApiUrl = "http://wazigate/devices/" # Production mode
DeviceApiUrl = "http://localhost:8080/devices/" # Debug mode

# Path to the root of the code
PATH = os.path.dirname(os.path.abspath(__file__))

# Path of config file
ConfigPath = 'config/current_config.json'

# global list of device and sensor ids
DeviceAndSensorIdsMoisture = []
DeviceAndSensorIdsTemp = []

# GPS
Gps_info = ""

# Slope to evaluate irrgation has taken place
Slope = 0

# Threshold to irrigate plants
Threshold = 0

# Retention curve 
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

# Helper to search other sensor values and 
def getSensorAtTheSameTime(deviceAndSensorIds, dataOfFirstSensor):
    # TODO: USE THE DECODER NAMES -> TELL THEM, handle case if there are multiple sensor values for a timespan
    mapping = {
        "time": "timeStamp",
        "sensorId": "sensorId",
        "lon": "longitude",
        "lat": "latitude",
        "Air Temperature": "airTemp",
        "Air Humidity": "airHum",
        "Barometric Pressure": "pressure",
        "Wind Speed": "windSpeed",
        "Wind Direction Sensor": "windDirection",
        "Light Intensity": "lightIntensity",
        "UV Index": "uvIndex",
        "Rain Gauge": "rainfall"
    }

    # Dict to return in the end
    allSensorsDict = {
            "timeStamp": None,
            "sensorId": None,
            "longitude": None,
            "latitude": None,
            "airTemp": None,
            "airHum": None,
            "pressure": None,
            "windSpeed": None,
            "windDirection": None,
            "lightIntensity": None,
            "uvIndex": None,
            "rainfall": None
    }

    # Get time of first sensor in list
    time = dataOfFirstSensor['time']
    # Set time of the first selected sensor as time of the dict
    allSensorsDict["timeStamp"] = time
    # Set given sensor id to dict
    allSensorsDict["sensorId"] = Id
    # Set GPS coordinates
    coordinates = Gps_info.split(",")
    allSensorsDict["longitude"] = coordinates[1]
    allSensorsDict["latitude"] = coordinates[0]
    # Parse the ISO string into a datetime object
    dateObject = datetime.fromisoformat(time)
    # Subtract and add 5 seconds to get interval
    fromObject = dateObject - timedelta(seconds=int(Threshold))
    toObject = dateObject + timedelta(seconds=int(Threshold))

    # Search all other choosen sensors to see if there are occurances too
    for sensor in deviceAndSensorIds:
        # Create URL for API call
        api_url = DeviceApiUrl + sensor.split('/')[0] + "/sensors/" + sensor.split('/')[1] + "/values?from=" + fromObject.isoformat() + "&to=" + toObject.isoformat()
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

        try:
            # Send a GET request to the API
            response = requests.get(encoded_url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # The response content contains the data from the API
                response_ok = response.json()

                # Add values to the all_Sensors_dict
                if len(response_ok) != 0:
                    nameToAdd = mapping[sensor.split("/")[1]]
                    allSensorsDict[nameToAdd] = response_ok[0]["value"]
            else:
                print("Request failed with status code:", response.status_code)
        except requests.exceptions.RequestException as e:
            # Handle request exceptions (e.g., connection errors)
            print("Request error:", e)

    return allSensorsDict

# Get historical sensor values from WaziGates API
def setConfig(url, body):
    global DeviceAndSensorIdsMoisture
    global DeviceAndSensorIdsTemp
    global Gps_info
    global Slope
    global Threshold

    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    # Get choosen sensors
    DeviceAndSensorIdsMoisture = parsed_data.get('selectedOptionsMoisture', [])
    DeviceAndSensorIdsTemp = parsed_data.get('selectedOptionsTemp', [])

    # Get data from forms
    Gps_info = parsed_data.get('gps', [])[0]
    Slope = parsed_data.get('slope', [])[0]
    Threshold = float(parsed_data.get('thres', [])[0])

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
        "Gps_info": {"lattitude": Gps_info.split(',')[0].lstrip(), "longitude": Gps_info.split(',')[1].lstrip()},
        "Slope": Slope,
        "Threshold": Threshold,
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
    global Gps_info
    global Slope
    global Threshold

    with open(ConfigPath, 'r') as file:
        # Parse JSON from the file
        data = json.load(file)

    # Get choosen sensors
    DeviceAndSensorIdsMoisture = data.get('DeviceAndSensorIdsMoisture', [])
    DeviceAndSensorIdsTemp = data.get('DeviceAndSensorIdsTemp', [])

    # Get data from forms
    Gps_info = data.get('Gps_info', [])
    Slope = float(data.get('Slope', []))
    Threshold = float(data.get('Threshold', []))

    # Get soil water retention curve
    Soil_water_retention_curve = data.get('Soil_water_retention_curve', [])


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

def extract_and_format(data, key, datatype):
    values = []
    for items in data:
        for item in items:
            if datatype == "str":
                values.append(str(item[key]))
            else:
                values.append(float(item[key]))
    
    return values


def getHistoricalChartData(url, body): 
    # Load data from local wazigate api -> each sensor individually
    data_moisture = []
    data_temp = []
    for moisture in DeviceAndSensorIdsMoisture:
        data_moisture.append(create_model.load_data_api(moisture))
    for temp in DeviceAndSensorIdsTemp:
        data_temp.append(create_model.load_data_api(temp))
    
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


# get values from create_model.py if models had been trained
def getPredictionChartData(url, body): 
    data_moisture = create_model.get_predictions()

    if data_moisture is False:
        response_data = {"model": False}
        status_code = 404
        
        return status_code, bytes(json.dumps(response_data), "utf8"), []

    for moisture in DeviceAndSensorIdsMoisture:
        data_moisture.append(create_model.load_data_api(moisture))

    
    # extract series from key value pairs
    f_data_time = extract_and_format(data_moisture, "time", "str")
    f_data_moisture = extract_and_format(data_moisture, "value", "float")


    # Create the chart_data dictionary
    chart_data = {
        "timestamps": f_data_time,
        "moistureSeries": f_data_moisture
    }

    return 200, bytes(json.dumps(chart_data), "utf8"), []

usock.routerGET("/api/getPredictionChartData", getPredictionChartData)

def startTraining(url, body):
    create_model.main()

    return 200, b"", []

usock.routerGET("/api/startTraining", startTraining)

#------------------#


if __name__ == "__main__":
    usock.start()
