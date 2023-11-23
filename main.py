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
    global Gps_info
    global Threshold

    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    # Get choosen sensors
    DeviceAndSensorIdsMoisture = parsed_data.get('selectedOptionsMoisture', [])
    DeviceAndSensorIdsTemp = parsed_data.get('selectedOptionsTemp', [])

    # Get data from forms
    Gps_info = parsed_data.get('gps', [])[0]
    Slope = parsed_data.get('slope', [])[0]
    Threshold = int(parsed_data.get('thres', [])[0])

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


def checkConfigPresent(url, body):
    if os.path.exists(ConfigPath):
        response_data = {"config_present": True}
        status_code = 200
    else:
        response_data = {"config_present": False}
        status_code = 404

    response = {
        "data": response_data,
        "status_code": status_code
    }


    return status_code, bytes(json.dumps(response), "utf8"), []

usock.routerGET("/api/checkConfigPresent", checkConfigPresent)


def getChartData(url, body): 
    # Later it has to call a getter of the machine learning data
    chart_data = {
        "timestamps": [
            "2023-10-27T00:00:00.000Z",
            "2023-10-27T00:30:00.000Z",
            "2023-10-27T01:00:00.000Z",
            "2023-10-27T01:30:00.000Z",
            "2023-10-27T02:00:00.000Z",
            "2023-10-27T02:30:00.000Z",
            "2023-10-27T03:00:00.000Z",
            "2023-10-27T03:30:00.000Z",
            "2023-10-27T04:00:00.000Z",
            "2023-10-27T04:30:00.000Z",
            "2023-10-27T05:00:00.000Z",
            "2023-10-27T05:30:00.000Z",
            "2023-10-27T06:00:00.000Z",
            "2023-10-27T06:30:00.000Z",
            "2023-10-27T07:00:00.000Z",
            "2023-10-27T07:30:00.000Z",
            "2023-10-27T08:00:00.000Z",
            "2023-10-27T08:30:00.000Z",
            "2023-10-27T09:00:00.000Z",
            "2023-10-27T09:30:00.000Z",
            "2023-10-27T10:00:00.000Z",
            "2023-10-27T10:30:00.000Z",
            "2023-10-27T11:00:00.000Z",
            "2023-10-27T11:30:00.000Z",
            "2023-10-27T12:00:00.000Z",
            "2023-10-27T12:30:00.000Z",
            "2023-10-27T13:00:00.000Z",
            "2023-10-27T13:30:00.000Z",
            "2023-10-27T14:00:00.000Z",
            "2023-10-27T14:30:00.000Z",
            "2023-10-27T15:00:00.000Z",
            "2023-10-27T15:30:00.000Z",
            "2023-10-27T16:00:00.000Z",
            "2023-10-27T16:30:00.000Z",
            "2023-10-27T17:00:00.000Z",
            "2023-10-27T17:30:00.000Z",
            "2023-10-27T18:00:00.000Z",
            "2023-10-27T18:30:00.000Z",
            "2023-10-27T19:00:00.000Z",
            "2023-10-27T19:30:00.000Z",
            "2023-10-27T20:00:00.000Z",
            "2023-10-27T20:30:00.000Z",
            "2023-10-27T21:00:00.000Z",
            "2023-10-27T21:30:00.000Z",
            "2023-10-27T22:00:00.000Z",
            "2023-10-27T22:30:00.000Z",
            "2023-10-27T23:00:00.000Z",
            "2023-10-27T23:30:00.000Z"
        ],
        "temperatureSeries": [
            15.600074264548083,
            24.12311022074752,
            34.31782173126873,
            28.455899612440586,
            16.12128625242102,
            23.055810993645097,
            29.21163466829131,
            31.31963010928032,
            35.79820931897718,
            21.85963643704078,
            38.05857704255487,
            28.845221760761937,
            34.40664620713963,
            38.21868279023465,
            38.08516218141855,
            15.696997843665592,
            33.11393761915116,
            14.671548716062916,
            28.32757536227938,
            29.73616035135298,
            30.556430321073546,
            20.69021429672725,
            32.83523010915863,
            17.607988136959236,
            30.312475971110883,
            22.59268014990768,
            12.95740713257051,
            27.194886004678333,
            14.273240569785174,
            23.85288886643452,
            22.500754633198346,
            34.96481203215509,
            17.964894248113647,
            14.617909668774172,
            30.07231209714445,
            36.780072214086405,
            24.66452783813339,
            31.408879580566053,
            27.73056381512289,
            16.53859729126703,
            25.15145012443309,
            15.952146199243174,
            30.275672695479793,
            31.953425751136933,
            32.829274976078546,
            33.55825494742465,
            31.572225206748648,
            28.835270228832174,
            25.298747279279224
        ],
        "moistureSeries": [
            0.46040755653673824,
            0.15307631289089358,
            0.7344383899628705,
            0.4915939195966151,
            0.23659511015088312,
            0.09093519982375536,
            0.37664715459754445,
            0.4237122751500834,
            0.3691316521754932,
            0.3984403418537526,
            0.05004970485246545,
            0.792340374841263,
            0.0841243038000058,
            0.0365801462653678,
            0.2871615689658606,
            0.025398452814999967,
            0.9473469867073306,
            0.3732142644085125,
            0.7855359725754455,
            0.15921855181173213,
            0.9001694083044092,
            0.33678613233576684,
            0.31221459501438613,
            0.30949807715724725,
            0.19377612170966817,
            0.1400635354092256,
            0.05391876401440167,
            0.032767111983625075,
            0.5427687918905512,
            0.23482175242466362,
            0.5366222752260637,
            0.5026159364933511,
            0.5873404805482252,
            0.6651958292964214,
            0.2527526805055375,
            0.37820497429278005,
            0.8558364272798893,
            0.6360375048840632,
            0.25877568949818194,
            0.8524255121027485,
            0.35477189072721424,
            0.021328869491668688,
            0.02544467490308818,
            0.4468888280350119,
            0.3301872141132234,
            0.5702467657886484,
            0.21657381603095212
        ]
    }

    return 200, bytes(json.dumps(chart_data), "utf8"), []

usock.routerGET("/api/getChartData", getChartData)

def startTraining(url, body):
    create_model.main()

    return 200, b"", []

usock.routerGET("/api/startTraining", startTraining)

#------------------#


if __name__ == "__main__":
    usock.start()
