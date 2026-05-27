# TODO: delete ports from URLs


#!/usr/bin/python
from crops import get_crop_params
import training_thread
from utils import NetworkUtils, TimeUtils
import plot_manager
from plot import Plot
import actuation
import create_model
import csv
from datetime import datetime, timedelta
from io import StringIO
import json
import threading
from threading import Timer
import time
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import pandas as pd
import usock
import os
import glob
import shutil
import logging
from logging.handlers import RotatingFileHandler
import pathlib
import numpy as np
from collections import defaultdict
from dateutil import parser

# Quiet pycaret's very chatty INFO logging - must be set BEFORE create_model imports pycaret
os.environ.setdefault("PYCARET_CUSTOM_LOGGING_LEVEL", "CRITICAL")
os.environ.setdefault("PYCARET_NO_LOGGING", "1")


# ---------------------#
# Path to the root of the code
PATH = os.path.dirname(os.path.abspath(__file__))

# Set the threshold to cleanup models to 3 months (approximately 90 days)
THRESHOLD_DAYS_CLEANUP = 90
# Per-run scratch dirs in tmp/ are dead after the next run; set to 90 to match if preferred
TMP_DIR_CLEANUP_DAYS = 2
Auto_start_training = False  # TODO: set to false for production, then training is only started when user clicks on "start training" in UI, otherwise it is started directly when config is present, which can lead to long waiting times on page load if training is heavy
# ---------------------#


def index(url, body=""):
    return 200, b"Salam Goloooo", []


usock.routerGET("/", index)

# ------------------#


def ui(url, body=''):
    filename = urlparse(url).path.replace("/ui/", "")
    if (len(filename) == 0):
        filename = 'index.html'

    # ---------------#

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

    # ---------------#

    try:
        with open(PATH + '/ui/' + filename, mode='rb') as file:
            return 200, file.read(), [conType]
    except Exception as e:
        print("Error: ", e)
        return 404, b"File not found", []


usock.routerGET("/ui/(.*)", ui)
usock.routerPOST("/ui/(.*)", ui)

# ------------------#

# Cleans python and pycaret logs


class LogCleanerThread(threading.Thread):
    def __init__(self, file_path, age_limit_days=90, check_interval=86400, name=None):
        super().__init__(name=name)
        self.file_path = file_path
        self.age_limit_days = age_limit_days
        self.check_interval = check_interval
        self.stop_thread = threading.Event()

    def clean_log(self):
        """Clears log file if it is older than the age limit."""
        print(f"[{self.name}] Checking log file: {self.file_path}")
        if os.path.exists(self.file_path):
            last_modified_time = datetime.fromtimestamp(
                os.path.getmtime(self.file_path))
            if datetime.now() - last_modified_time > timedelta(days=self.age_limit_days):
                open(self.file_path, 'w').close()  # Clear the file contents
                print(f"[{self.name}] {self.file_path} has been cleaned.")
            else:
                print(f"[{self.name}] {self.file_path} is not old enough to clean.")
        else:
            print(f"[{self.name}] Log file does not exist: {self.file_path}")

    def run(self):
        while not self.stop_thread.is_set():
            self.clean_log()
            time.sleep(self.check_interval)  # Wait before the next check

    def stop(self):
        self.stop_thread.set()

# setup function for log cleaner => TODO: changed function WITHOUT TESTING IT!!!!!!!!!!!!!!!!!
# Bound logs.log by size via rotation. Age-based cleanup cannot bound an actively
# written log (its mtime stays recent), so rotation is the real cap here. Configured
# once on the root logger; pycaret's basicConfig call is then a no-op (root has a handler).


def setup_logging():
    handler = RotatingFileHandler(
        "logs.log", maxBytes=30 * 1024 * 1024, backupCount=1)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s:%(levelname)s:%(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    # runtime-effective even after pycaret is already imported
    logging.getLogger("pycaret").setLevel(logging.WARNING)


def schedule_log_cleanup():
    logs_to_clean = [
        ("logs.log", 90),       # Python log file
        ("python_logs.log", 90)  # Docker log file
    ]

    # Start a thread for each log file
    for log_path, age_limit in logs_to_clean:
        thread_name = f"LogCleaner-{os.path.basename(log_path)}"
        cleaner = LogCleanerThread(
            file_path=log_path, age_limit_days=age_limit, name=thread_name)
        cleaner.daemon = True  # Run thread in the background
        cleaner.start()


class ModelCleanerThread(threading.Thread):
    def __init__(self, file_paths, dir_globs=None, file_globs=None,
                 interval_days=7, name="ModelCleaner"):
        super().__init__(name=name)
        # file_paths: folders to walk, pruning old FILES but keeping the folder tree
        self.file_paths = file_paths if isinstance(
            file_paths, list) else [file_paths]
        # dir_globs: glob patterns whose WHOLE matching folders are deleted once old
        self.dir_globs = dir_globs or []
        # file_globs: glob patterns whose matching FILES are pruned (folder kept)
        self.file_globs = file_globs or []
        self.interval_days = interval_days
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            for folder in self.file_paths:
                # prune files, keep tree
                delete_old_files(folder)
            for pattern in self.dir_globs:
                # delete whole scratch dirs
                delete_old_dirs(pattern, TMP_DIR_CLEANUP_DAYS)
            for pattern in self.file_globs:
                # prune specific files
                delete_old_glob_files(pattern)
            time.sleep(self.interval_days * 24 * 3600)

    def stop(self):
        self.stop_event.set()

# Deletes files older than the threshold from the specified folder and its subfolders.


def delete_old_files(folder_path):
    current_time = time.time()
    threshold_time = current_time - THRESHOLD_DAYS_CLEANUP * 24 * 60 * 60

    # Traverse the directory, including subdirectories
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            # Skip .gitkeep, .gitignore and readme files
            if file_name in {".gitkeep", ".gitignore", "README.md"}:
                continue

            file_path = os.path.join(root, file_name)
            # Check if the file is older than the threshold
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < threshold_time:
                try:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

# Deletes WHOLE folders matching a glob once older than age_days (per-run scratch dirs).
# Uses glob.glob because os.walk does NOT expand '*' - passing "tmp/tuning_*" to os.walk
# silently matches nothing, which is why the old tmp cleanup never ran.


def delete_old_dirs(pattern, age_days=THRESHOLD_DAYS_CLEANUP):
    threshold_time = time.time() - age_days * 24 * 60 * 60
    for path in glob.glob(pattern):
        if os.path.isdir(path) and os.path.getmtime(path) < threshold_time:
            try:
                shutil.rmtree(path)
                print(f"Deleted old dir: {path}")
            except Exception as e:
                print(f"Error deleting dir {path}: {e}")

# Deletes only the files matching a glob (for folders shared with data we must keep,
# e.g. data/debug holds datasets alongside cache files - never rmtree that folder).


def delete_old_glob_files(pattern, age_days=THRESHOLD_DAYS_CLEANUP):
    threshold_time = time.time() - age_days * 24 * 60 * 60
    for file_path in glob.glob(pattern):
        if os.path.isfile(file_path) and os.path.getmtime(file_path) < threshold_time:
            try:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# setup function for model cleaner


def schedule_model_cleanup(file_paths, dir_globs=None, file_globs=None, interval_days=7):
    """
    Periodically prunes old files/folders every interval_days.
    """
    cleaner = ModelCleanerThread(
        file_paths, dir_globs, file_globs, interval_days)
    cleaner.start()

    return cleaner

# Get URL of API from .env file => TODO: better with try catch than locals, getenv can still stop backend


def getApiUrl(url, body):
    url = NetworkUtils.ApiUrl

    if url not in (None, ''):
        data = url
        status_code = 200
    else:
        data = False,
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

    # if(plot_manager.setPlot(currentTab)):
    #         return 200, b"Plot has been set.", []
    # else:
    #     return 200, b"Has been set but has no config yet.", []

    plot_manager.setPlot(currentTab)

    return 200, f"Plot has been set. PlotId = {currentTab}".encode(), []


usock.routerPOST("/api/setPlot", setPlot)

# When Page is reloaded it will return formerly loaded plots


def getPlots(url, body):
    # Call function in plot manager
    plots = plot_manager.getPlots()

    # Create array with names of tabs to return to frontend
    tab_name_array = []
    # for plot in plots:
    for i in range(1, len(plots)+1, 1):
        try:
            tab_name_array.append(plots[i].user_given_name)
        except KeyError:
            continue

    response = {
        "tabnames": tab_name_array,
        "status_code": 200
    }

    return response["status_code"], bytes(json.dumps(response), "utf8"), []


usock.routerGET("/api/getPlots", getPlots)

# Add a plot during runtine TODO: finish


def addPlot(url, body):
    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    # ID of tab in UI
    amount_tabs = int(parsed_data.get('tab_nr', [])[0])

    # Call function in plot manager
    next_number, newfilename = plot_manager.addPlot(amount_tabs)

    response = {
        "plot_number": next_number,
        "filename": newfilename,
        "status_code": 200
    }

    return response["status_code"], bytes(json.dumps(response), "utf8"), []


usock.routerPOST("/api/addPlot", addPlot)

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


usock.routerPOST("/api/removePlot", removePlot)


# Get historical sensor values from WaziGates API
def setConfig(url, body):
    # Get current plot
    currentPlot = plot_manager.getCurrentPlot()

    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    def _get_first(key, default=""):
        return parsed_data.get(key, [default])[0]

    def _parse_float(value, field, errors, min_value=None, max_value=None):
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors[field] = "Must be a number."
            return None
        if min_value is not None and num < min_value:
            errors[field] = f"Must be >= {min_value}."
        if max_value is not None and num > max_value:
            errors[field] = f"Must be <= {max_value}."
        return num

    def _parse_int(value, field, errors, min_value=None, max_value=None):
        try:
            num = int(float(value))
        except (TypeError, ValueError):
            errors[field] = "Must be an integer."
            return None
        if min_value is not None and num < min_value:
            errors[field] = f"Must be >= {min_value}."
        if max_value is not None and num > max_value:
            errors[field] = f"Must be <= {max_value}."
        return num

    # Get choosen sensors
    currentPlot.device_and_sensor_ids_moisture = parsed_data.get(
        'selectedOptionsMoisture', [])
    currentPlot.device_and_sensor_ids_temp = parsed_data.get(
        'selectedOptionsTemp', [])
    currentPlot.device_and_sensor_ids_flow = parsed_data.get(
        'selectedOptionsFlow', [])
    if len(currentPlot.device_and_sensor_ids_flow) != 0:
        currentPlot.device_and_sensor_ids_flow_confirmation = [currentPlot.getConfirmationDeviceID(
            # get confirmation sensors, part of the flow meter, always on xlpp channel 5
            currentPlot.device_and_sensor_ids_flow)]
    else:
        currentPlot.device_and_sensor_ids_flow_confirmation = []
    # Parse JSON

    # Get data from forms
    errors = {}
    name_list = parsed_data.get('name', [])
    currentPlot.user_given_name = name_list[0].strip() if name_list else ""
    if not currentPlot.user_given_name:
        errors["name"] = "Farm name is required."

    currentPlot.zone_name = _get_first(
        'zone_name', currentPlot.user_given_name)
    if not currentPlot.zone_name:
        errors["zone_name"] = "Zone name is required."

    currentPlot.sensor_kind = _get_first('sensor_kind')
    if not currentPlot.sensor_kind:
        errors["sensor_kind"] = "Sensor kind is required."

    gps_raw = _get_first('gps')
    gps_parts = [p.strip() for p in gps_raw.strip().split(",") if p.strip()]
    gps_lat = None
    gps_lon = None
    if len(gps_parts) >= 2:
        gps_lat = _parse_float(gps_parts[0], "gps_latitude", errors, -90, 90)
        gps_lon = _parse_float(
            gps_parts[1], "gps_longitude", errors, -180, 180)
    else:
        errors["gps"] = "GPS must be 'lat, lon'."

    currentPlot.slope = _parse_float(_get_first('slope'), "slope", errors)
    currentPlot.threshold = _parse_float(
        _get_first('thres'), "threshold", errors)
    currentPlot.irrigation_amount = _parse_float(
        _get_first('amount'), "irrigation_amount", errors, 0.0)
    plot_area_raw = parsed_data.get(
        'plot_area_m2', parsed_data.get('area', []))
    if plot_area_raw and str(plot_area_raw[0]).strip() != '':
        currentPlot.plot_area_m2 = _parse_float(
            plot_area_raw[0], "plot_area_m2", errors, 0.0)
    else:
        currentPlot.plot_area_m2 = float(
            getattr(currentPlot, 'plot_area_m2', 0.0))
    currentPlot.irrigation_type = parsed_data.get(
        'irrigation_type', [getattr(currentPlot, 'irrigation_type', 'unknown')]
    )[0] or 'unknown'
    currentPlot.look_ahead_time = _parse_float(
        _get_first('lookahead'), "look_ahead_time", errors, 0.0)

    currentPlot.start_date = _get_first('start')
    if not currentPlot.start_date:
        errors["start_date"] = "Start date is required."
    else:
        try:
            parser.parse(currentPlot.start_date)
        except Exception:
            errors["start_date"] = "Start date must be a valid ISO timestamp."

    currentPlot.period = _parse_int(_get_first('period'), "period", errors, 1)
    currentPlot.soil_type = _get_first('soil')
    if not currentPlot.soil_type:
        errors["soil_type"] = "Soil type is required."
    soil_texture_raw = parsed_data.get('soil_texture_class', [None])[0]
    try:
        currentPlot.soil_texture_class = int(soil_texture_raw)
    except (TypeError, ValueError):
        currentPlot.soil_texture_class = getattr(
            currentPlot, 'soil_texture_class', None)
    currentPlot.permanent_wilting_point = _parse_int(
        _get_first('pwp'), "permanent_wilting_point", errors, 0)
    currentPlot.field_capacity_upper = _parse_int(
        _get_first('fcu'), "field_capacity_upper", errors, 0)
    currentPlot.field_capacity_lower = _parse_int(
        _get_first('fcl'), "field_capacity_lower", errors, 0)
    currentPlot.saturation = _parse_int(
        _get_first('sat'), "saturation", errors, 0)
    currentPlot.crop_type = _get_first(
        'crop_type', getattr(currentPlot, 'crop_type', 'generic'))
    if not currentPlot.crop_type:
        errors["crop_type"] = "Crop type is required."
    currentPlot.planting_date = _get_first(
        'planting_date', getattr(currentPlot, 'planting_date', ''))
    if not currentPlot.planting_date:
        errors["planting_date"] = "Planting date is required."
    else:
        try:
            parser.parse(currentPlot.planting_date)
        except Exception:
            errors["planting_date"] = "Planting date must be YYYY-MM-DD."

    def _extract_coords(plot):
        gps = getattr(plot, 'gps_info', None)
        if isinstance(gps, dict):
            lat = gps.get('latitude', gps.get('lattitude'))
            lon = gps.get('longitude')
            try:
                return float(lat), float(lon)
            except (TypeError, ValueError):
                return None
        return None

    if currentPlot.user_given_name and gps_lat is not None and gps_lon is not None:
        for pid, other in plot_manager.getPlots().items():
            if other is currentPlot:
                continue
            other_name = getattr(other, 'user_given_name', '') or ''
            if other_name and other_name.strip().lower() == currentPlot.user_given_name.strip().lower():
                errors["duplicate"] = (
                    f"Plot name already exists (plot {pid})."
                )
                break
            other_coords = _extract_coords(other)
            other_crop = getattr(other, 'crop_type', None)
            if other_coords is not None and other_crop:
                lat2, lon2 = other_coords
                if (
                    abs(lat2 - float(gps_lat)) <= 1e-6
                    and abs(lon2 - float(gps_lon)) <= 1e-6
                    and other_crop == currentPlot.crop_type
                ):
                    errors["duplicate"] = (
                        f"Plot with same GPS and crop already exists (plot {pid})."
                    )
                    break
    initial_gdd_raw = _get_first(
        'initial_gdd', getattr(currentPlot, 'initial_gdd', 0.0))
    try:
        currentPlot.initial_gdd = float(initial_gdd_raw)
    except (TypeError, ValueError):
        currentPlot.initial_gdd = float(
            getattr(currentPlot, 'initial_gdd', 0.0))
    currentPlot.use_dynamic_threshold = _get_first(
        'use_dynamic_threshold',
        str(getattr(currentPlot, 'use_dynamic_threshold', False))
    ) in ('true', 'True', '1')

    # Get soil water retention curve
    currentPlot.soil_water_retention_curve = _get_first('ret')
    if not currentPlot.soil_water_retention_curve:
        errors["soil_water_retention_curve"] = (
            "Soil water retention curve is required."
        )

    # Create a CSV file-like object from the CSV string
    csv_file = StringIO(currentPlot.soil_water_retention_curve or "")

    # Parse the CSV data into a list of dictionaries
    csv_data = []
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        csv_data.append(row)
    if not csv_data:
        errors["soil_water_retention_curve"] = (
            "Soil water retention curve must include at least one row."
        )

    # Organize the variables into a dictionary
    gps_lat = gps_lat if gps_lat is not None else 0.0
    gps_lon = gps_lon if gps_lon is not None else 0.0
    currentPlot.gps_info = {
        "latitude": gps_lat,
        "longitude": gps_lon,
        "lattitude": gps_lat,
    }

    pending_sensors = (
        len(currentPlot.device_and_sensor_ids_moisture) == 0
        or len(currentPlot.device_and_sensor_ids_temp) == 0
    )

    if errors:
        response = {
            "status": "error",
            "message": "Validation failed",
            "errors": errors,
        }
        return 400, bytes(json.dumps(response), "utf8"), []

    data = {
        "DeviceAndSensorIdsMoisture": currentPlot.device_and_sensor_ids_moisture,
        "DeviceAndSensorIdsTemp": currentPlot.device_and_sensor_ids_temp,
        "DeviceAndSensorIdsFlow": currentPlot.device_and_sensor_ids_flow,
        "DeviceAndSensorIdsFlowConfirmation": currentPlot.device_and_sensor_ids_flow_confirmation,
        "Sensor_kind": currentPlot.sensor_kind,
        "Name": currentPlot.user_given_name,
        "Zone_name": getattr(currentPlot, 'zone_name', currentPlot.user_given_name),
        # "Gps_info": {"lattitude": currentPlot.gps_info['lattitude'], "longitude": currentPlot.gps_info['longitude']},
        # "Gps_info": currentPlot.gps_info,
        "Gps_info": {
            "latitude": gps_lat,
            "longitude": gps_lon,
            "lattitude": gps_lat,
        },
        "Slope": currentPlot.slope,
        "Threshold": currentPlot.threshold,
        "Irrigation_amount": currentPlot.irrigation_amount,
        "Plot_area_m2": currentPlot.plot_area_m2,
        "Irrigation_type": getattr(currentPlot, 'irrigation_type', 'unknown'),
        "Look_ahead_time": currentPlot.look_ahead_time,
        "Start_date": currentPlot.start_date,
        "Period": currentPlot.period,
        "Soil_type": currentPlot.soil_type,
        "Soil_water_retention_curve": csv_data,  # Use the parsed CSV data
        "PermanentWiltingPoint": currentPlot.permanent_wilting_point,
        "FieldCapacityUpper": currentPlot.field_capacity_upper,
        "FieldCapacityLower": currentPlot.field_capacity_lower,
        "Saturation": currentPlot.saturation,
        "Soil_texture_class": getattr(currentPlot, 'soil_texture_class', None),
        "Crop_type": getattr(currentPlot, 'crop_type', 'generic'),
        "Planting_date": getattr(currentPlot, 'planting_date', ''),
        "Initial_gdd": float(getattr(currentPlot, 'initial_gdd', 0.0)),
        "Use_dynamic_threshold": getattr(currentPlot, 'use_dynamic_threshold', False),
        "Pending_sensors": pending_sensors,
        "Pending_sensors_reason": (
            "Missing moisture or temperature sensors"
            if pending_sensors else ""
        )
    }

    # Save the JSON data to the file
    with open(plot_manager.getCurrentConfig(), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    if pending_sensors:
        response = {
            "status": "pending",
            "message": (
                "Configuration saved, but sensors are pending. "
                "Select moisture and temperature sensors to enable predictions."
            ),
        }
        return 202, bytes(json.dumps(response), "utf8"), []

    response = {
        "status": "ok",
        "message": "Configuration has been successfully saved!",
    }
    sync_sensor_registry_from_plots()
    return 200, bytes(json.dumps(response), "utf8"), []


usock.routerPOST("/api/setConfig", setConfig)

# Load config from all file


def getConfigsFromAllFiles():
    # Get plots
    plots = plot_manager.getPlots()

    for i in range(1, len(plots)+1, 1):
        config = plots[i].configPath
        if os.path.exists(config):
            with open(config, 'r') as file:
                # Parse JSON from the file
                data = json.load(file)
            if plots[i].load_data_from_csv:
                with open(plots[i].data_from_csv, "r") as file:
                    # Perform operations on the file
                    debug_csv = pd.read_csv(file, header=0)

                plots[i].device_and_sensor_ids_moisture = []
                plots[i].device_and_sensor_ids_temp = []
                plots[i].device_and_sensor_ids_flow = []

                # create array with sensors strings
                for col in debug_csv.columns:
                    if (
                        col.startswith("tension")
                        or col.startswith("vwc")
                        or col.startswith("volumetric")
                        or col.startswith("capacitive")
                    ):
                        plots[i].device_and_sensor_ids_moisture.append(col)
                    elif col.startswith("soil_temp"):
                        plots[i].device_and_sensor_ids_temp.append(col)
                    # This is not implemented
                    elif col.startswith("flow"):
                        plots[i].device_and_sensor_ids_flow.append(col)
            else:
                # Get choosen sensors
                # print("Before assignment:",  plot_manager.Plots[i].device_and_sensor_ids_moisture)
                plots[i].device_and_sensor_ids_moisture = data.get(
                    'DeviceAndSensorIdsMoisture', [])
                # print("After assignment:",  plot_manager.Plots[i].device_and_sensor_ids_moisture)
                plots[i].device_and_sensor_ids_temp = data.get(
                    'DeviceAndSensorIdsTemp', [])
                plots[i].device_and_sensor_ids_flow = data.get(
                    'DeviceAndSensorIdsFlow', [])
                plots[i].device_and_sensor_ids_flow_confirmation = data.get(
                    'DeviceAndSensorIdsFlowConfirmation', [])

            # Get data from forms
            plots[i].user_given_name = data.get('Name', [])
            plots[i].zone_name = data.get(
                'Zone_name', plots[i].user_given_name)
            plots[i].sensor_kind = data.get('Sensor_kind', [])
            gps_info = data.get('Gps_info', {})
            if isinstance(gps_info, dict):
                lat = gps_info.get('latitude', gps_info.get('lattitude', 0))
                lon = gps_info.get('longitude', 0)
                plots[i].gps_info = {
                    "latitude": lat,
                    "longitude": lon,
                    "lattitude": lat,
                }
            else:
                plots[i].gps_info = gps_info
            plots[i].slope = float(data.get('Slope', []))
            plots[i].threshold = float(data.get('Threshold', []))
            plots[i].irrigation_amount = float(
                data.get('Irrigation_amount', []))
            plots[i].plot_area_m2 = float(data.get('Plot_area_m2', 0))
            plots[i].irrigation_type = data.get(
                'Irrigation_type', 'unknown') or 'unknown'
            plots[i].look_ahead_time = float(data.get('Look_ahead_time', []))
            plots[i].start_date = data.get('Start_date', [])
            plots[i].period = int(data.get('Period', []))
            plots[i].soil_type = data.get('Soil_type', [])
            plots[i].soil_texture_class = data.get('Soil_texture_class', None)
            plots[i].permanent_wilting_point = float(
                data.get('PermanentWiltingPoint', []))
            plots[i].field_capacity_upper = float(
                data.get('FieldCapacityUpper', []))
            plots[i].field_capacity_lower = float(
                data.get('FieldCapacityLower', []))
            plots[i].saturation = float(data.get('Saturation', []))

            # Get soil water retention curve -> currently not needed here
            plots[i].soil_water_retention_curve = data.get(
                'Soil_water_retention_curve', [])

            # Phenology / dynamic threshold configuration
            plots[i].crop_type = data.get('Crop_type', 'generic')
            plots[i].planting_date = data.get('Planting_date', '')
            try:
                plots[i].initial_gdd = float(data.get('Initial_gdd', 0.0))
            except (TypeError, ValueError):
                plots[i].initial_gdd = 0.0
            plots[i].use_dynamic_threshold = data.get(
                'Use_dynamic_threshold', False)

            # Sensor kind
            if plots[i].sensor_kind in ("tension", "both"):
                plots[i].sensor_unit = "Moisture in cbar (Soil Tension)"
            elif plots[i].sensor_kind == "capacitive":
                plots[i].sensor_unit = "Moisture in % (Volumetric Water Content)"
            else:
                plots[i].sensor_unit = "Unit is unknown"

# Get the config from backend to disply it in frontend settings.html


def returnConfig(url, body):
    try:
        currentPlot = plot_manager.getCurrentPlot()
        # Call the getConfigFromFile function to load variables
        if currentPlot.getConfigFromFile():

            # Check if all necessary plot variables are properly defined
            if not all(isinstance(var, (int, float, str, list, dict)) for var in [
                    currentPlot.device_and_sensor_ids_moisture,
                    currentPlot.device_and_sensor_ids_temp,
                    currentPlot.device_and_sensor_ids_flow,
                    currentPlot.sensor_kind,
                    currentPlot.user_given_name,
                    currentPlot.gps_info,
                    currentPlot.slope,
                    currentPlot.threshold,
                    currentPlot.irrigation_amount,
                    currentPlot.plot_area_m2,
                    getattr(currentPlot, 'irrigation_type', 'unknown'),
                    currentPlot.look_ahead_time,
                    currentPlot.start_date,
                    currentPlot.period,
                    currentPlot.permanent_wilting_point,
                    currentPlot.field_capacity_upper,
                    currentPlot.field_capacity_lower,
                    currentPlot.saturation]):

                raise ValueError(
                    "Variables are still missing or of incorrect type after loading from config.")

            # Construct the response data
            response_data = {
                "DeviceAndSensorIdsMoisture": currentPlot.device_and_sensor_ids_moisture,
                "DeviceAndSensorIdsTemp": currentPlot.device_and_sensor_ids_temp,
                "DeviceAndSensorIdsFlow": currentPlot.device_and_sensor_ids_flow,
                "Sensor_kind": currentPlot.sensor_kind,
                "Name": currentPlot.user_given_name,
                "Zone_name": getattr(currentPlot, 'zone_name', currentPlot.user_given_name),
                "Gps_info": currentPlot.gps_info,
                "Slope": currentPlot.slope,
                "Threshold": currentPlot.threshold,
                "Irrigation_amount": currentPlot.irrigation_amount,
                "Plot_area_m2": currentPlot.plot_area_m2,
                "Irrigation_type": getattr(currentPlot, 'irrigation_type', 'unknown'),
                "Look_ahead_time": currentPlot.look_ahead_time,
                "Start_date": currentPlot.start_date,
                "Period": currentPlot.period,
                "Soil_type": currentPlot.soil_type,
                "Soil_water_retention_curve": currentPlot.soil_water_retention_curve,
                "PermanentWiltingPoint": currentPlot.permanent_wilting_point,
                "FieldCapacityUpper": currentPlot.field_capacity_upper,
                "FieldCapacityLower": currentPlot.field_capacity_lower,
                "Saturation": currentPlot.saturation,
                "Soil_texture_class": getattr(currentPlot, 'soil_texture_class', None),
                "Crop_type": getattr(currentPlot, 'crop_type', 'generic'),
                "Planting_date": getattr(currentPlot, 'planting_date', ''),
                "Initial_gdd": float(getattr(currentPlot, 'initial_gdd', 0.0)),
                "Use_dynamic_threshold": getattr(currentPlot, 'use_dynamic_threshold', False)
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
            "error": "An Value error occured: " + str(ve),
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
    if os.path.exists(plot_manager.ConfigPath):  # solve multiple calls with dirty bit
        currentPlot = plot_manager.getCurrentPlot()
        currentPlot.getConfigFromFile()
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

# Called on page load->important for checkActiveIrrigation


def checkActiveIrrigation(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    if not currentPlot.getConfigFromFile():
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


def group_sensor_data(sensor_lists, agg_func=lambda vals: sum(vals)/len(vals), resample_interval="30T"):
    """
    Given a list of sensor-lists (each a list of {'time':…, 'value':…}),
    return two lists:
      - sorted timestamps (strings)
      - aggregated values (floats) per timestamp
    agg_func receives the list of values for that timestamp.
    """
    bucket = defaultdict(list)
    for series in sensor_lists:
        for rec in series:
            # normalize time strings if you want
            t = rec['time']
            bucket[t].append(rec['value'])

    # sort timestamps chronologically
    timestamps = sorted(bucket.keys(), key=lambda t: parser.isoparse(t))
    values = [agg_func(bucket[t]) for t in timestamps]

    # Convert to timezone-aware datetime index in UTC
    df = pd.DataFrame({"value": values})
    df.index = pd.to_datetime(timestamps, utc=True)  # force UTC

    # Resampling
    df = df.resample(resample_interval).mean().dropna()

    # Display in local time
    df.index = df.index.tz_convert(TimeUtils.Timezone)

    # Convert back to lists
    resampled_timestamps = df.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    resampled_values = df["value"].tolist()

    return resampled_timestamps, resampled_values


# def interpolate_list(data_list):
#     data = np.array(data_list, dtype=float)

#     # Find indices where values are not NaN
#     not_nan = np.where(~np.isnan(data))[0]
#     nan = np.where(np.isnan(data))[0]

#     if len(nan) == 0 or len(not_nan) < 2:
#         return data.tolist()

#     # Interpolate
#     data[nan] = np.interp(nan, not_nan, data[not_nan])
#     return data.tolist()

def interpolate_list_with_limit(data_list, max_gap=10):
    data = np.array(data_list, dtype=float)
    n = len(data)

    isnan = np.isnan(data)
    result = data.copy()

    start = None

    for i in range(n):
        if isnan[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                gap_len = end - start
                if gap_len <= max_gap and start > 0:
                    # Interpolate between data[start - 1] and data[end]
                    x = [start - 1, end]
                    y = [data[start - 1], data[end]]
                    interp_vals = np.interp(range(start, end), x, y)
                    result[start:end] = interp_vals
                # else leave as NaN
                start = None

    return result.tolist()


def smooth_outliers(data_list, window=2, threshold=0.5):
    data = data_list.copy()

    for i in range(len(data)):
        # Define window bounds
        start = max(0, i - window)
        end = min(len(data), i + window + 1)

        # Get neighboring values excluding current
        neighbors = [data[j] for j in range(start, end) if j != i]

        if len(neighbors) < 2:
            continue  # Not enough context

        local_avg = sum(neighbors) / len(neighbors)

        # If current value deviates significantly, replace it
        if abs(data[i] - local_avg) > threshold:
            data[i] = local_avg

    return data

# From key-value to series for CSV


def extract_and_format_csv(data, key):
    values = []

    # Collect all columns that start with the given key
    for col in data:
        if col.startswith(key):
            values.append(data[col].tolist())

    amount_series = len(values)
    reduced_values = []

    # Iterate row-wise
    for i in range(len(values[0])):
        sum_vals = 0
        count = 0

        for j in range(amount_series):
            val = values[j][i]
            if not np.isnan(val):
                sum_vals += val
                count += 1

        # Append average if we have valid values, else np.nan
        reduced_values.append(sum_vals / count if count > 0 else np.nan)

    # Fill missing values first
    filled_series = interpolate_list_with_limit(reduced_values)

    # Then smooth suspicious values
    smoothed = smooth_outliers(filled_series)

    final = [smoothed[0]]  # initialize
    for i in range(1, len(smoothed)):
        final.append(smoothed[i] if not np.isnan(smoothed[i]) else final[-1])

    return final


def irrigateManually(url, body):
    # Parse the query parameters from the URL
    query_params = parse_qs(urlparse(url).query)

    # Extract the 'amount' parameter (assuming it's passed as a query parameter)
    amount = float(query_params.get('amount', [0])[0])

    currentPlot = plot_manager.getCurrentPlot()
    if not actuation._has_actuator_support(currentPlot):
        return 400, bytes(json.dumps({"status": "error", "message": "Manual irrigation is disabled for advisory-only irrigation modes (rainfed/furrow/gravity)."}), "utf8"), []

    # Call the actuation function with the extracted amount
    response = actuation.irrigate_amount(currentPlot, amount)

    if not response:
        return 400, bytes(json.dumps({"status": "error", "message": "Irrigation failed or no active irrigation system.", "response": response}), "utf8"), []

    return 200, bytes(json.dumps({"status": "success", "amount": amount, "response": response}), "utf8"), []


usock.routerGET("/api/irrigateManually", irrigateManually)

# Get latest values for dashboard


def getValuesForDashboard(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    # Load config, to get latest changes
    currentPlot.config = currentPlot.read_config()

    if not currentPlot.device_and_sensor_ids_temp or not currentPlot.device_and_sensor_ids_moisture:
        update_sensor_registry_status(currentPlot, online=False)
        response_data = {"available": False}
        status_code = 404

        return status_code, bytes(json.dumps(response_data), "utf8"), []

    load_latest = (
        currentPlot.load_latest_data_csv
        if currentPlot.load_data_from_csv
        else currentPlot.load_latest_data_api
    )

    temp_readings = {}
    moisture_readings = {}
    for temp in currentPlot.device_and_sensor_ids_temp:
        temp_readings[temp] = load_latest(temp, "sensors")
    for moisture in currentPlot.device_and_sensor_ids_moisture:
        moisture_readings[moisture] = load_latest(moisture, "sensors")

    temp_average = _numeric_average(temp_readings.values())
    sensor_kind = str(currentPlot.sensor_kind or "").lower()

    if sensor_kind == "both":
        tension_ids, vwc_ids, unknown_ids = _split_moisture_sensors(
            currentPlot.device_and_sensor_ids_moisture)
        tension_ids_for_calc = list(tension_ids)
        if not tension_ids_for_calc and not vwc_ids and unknown_ids:
            tension_ids_for_calc = list(unknown_ids)

        tension_values = [moisture_readings.get(sid)
                          for sid in tension_ids_for_calc]
        vwc_values = [moisture_readings.get(sid) for sid in vwc_ids]

        tension_average = _numeric_average(tension_values)
        vwc_direct = _numeric_average(vwc_values)
        vwc_derived = None
        if tension_average is not None:
            vwc_derived = round(
                create_model.calc_volumetric_water_content_single_value(
                    tension_average, currentPlot
                ) * 100,
                2,
            )
        vwc_average = vwc_direct if vwc_direct is not None else vwc_derived

        dashboard_data = {
            "temp_average": temp_average,
            "moisture_average": tension_average,
            "vwc_average": vwc_average,
            "sensor_kind": currentPlot.sensor_kind,
            "vwc_source": "sensor" if vwc_direct is not None else (
                "derived" if vwc_derived is not None else "none"
            ),
            "tension_source": "sensor" if tension_average is not None else "none",
            "moisture_sensor_counts": {
                "tension": len(tension_ids),
                "vwc": len(vwc_ids),
                "unknown": len(unknown_ids),
            },
        }
    else:
        moisture_average = _numeric_average(moisture_readings.values())
        if _is_tension_kind(currentPlot.sensor_kind):
            vwc_average = None
            if moisture_average is not None:
                vwc_average = round(
                    create_model.calc_volumetric_water_content_single_value(
                        moisture_average, currentPlot
                    ) * 100,
                    2,
                )
            dashboard_data = {
                "temp_average": temp_average,
                "moisture_average": moisture_average,
                "vwc_average": vwc_average,
                "sensor_kind": currentPlot.sensor_kind,
            }
        else:
            dashboard_data = {
                "temp_average": temp_average,
                "moisture_average": None,
                "vwc_average": moisture_average,
                "sensor_kind": currentPlot.sensor_kind,
            }

    reading_map = {"moisture": {}, "temperature": {}, "flow": {}}
    for sensor_id, value in moisture_readings.items():
        reading_map["moisture"][sensor_id] = value
    for sensor_id, value in temp_readings.items():
        reading_map["temperature"][sensor_id] = value
    update_sensor_registry_status(
        currentPlot, reading_map=reading_map, online=True)

    return 200, bytes(json.dumps(dashboard_data), "utf8"), []


usock.routerGET("/api/getValuesForDashboard", getValuesForDashboard)


def getHistoricalChartData(url, body):
    # Load data from local wazigate api -> each sensor individually, save as key-value pairs
    data_moisture = []
    data_temp = []
    # data_flow = [] # TODO:later also show flow in vis

    # Get current plot (selected in UI)
    currentPlot = plot_manager.getCurrentPlot()
    # Load config, to get latest changes
    currentPlot.config = currentPlot.read_config()

    if currentPlot.load_data_from_csv:
        data = currentPlot.load_data_csv()

        # extract series from key value pairs
        f_data_time = data["Time"].tolist()
        f_data_moisture = extract_and_format_csv(data, "tension")
        f_data_temp = extract_and_format_csv(data, "soil_temp")

        # unite similar series
        # f_data_moisture = [item for sublist in f_data_moisture for item in sublist]

        # Make sure that the data is not empty
        if not f_data_moisture or not f_data_temp:
            response_data = {"available": False}
            status_code = 404

            return status_code, bytes(json.dumps(response_data), "utf8"), []
    else:
        # There is an Error if sensor data is loaded from file
        moisture_ids = list(currentPlot.device_and_sensor_ids_moisture)
        if str(currentPlot.sensor_kind or "").lower() == "both":
            tension_ids, vwc_ids, unknown_ids = _split_moisture_sensors(
                moisture_ids)
            if tension_ids:
                moisture_ids = tension_ids
            elif unknown_ids:
                moisture_ids = unknown_ids
            else:
                moisture_ids = vwc_ids

        for moisture in moisture_ids:
            data_moisture.append(currentPlot.load_data_api(
                moisture, "sensors", currentPlot.start_date))
        for temp in currentPlot.device_and_sensor_ids_temp:
            data_temp.append(currentPlot.load_data_api(
                temp, "sensors", currentPlot.start_date))
        # for flow in currentPlot.device_and_sensor_ids_flow: # TODO: maybe display that also here (is displayed in datasets data)
        #     data_flow.append(currentPlot.load_data_api(flow, "actuators", currentPlot.start_date))

        f_data_time, f_data_moisture = group_sensor_data(data_moisture)
        f_data_time, f_data_temp = group_sensor_data(data_temp)

        # extract series from key value pairs
        # f_data_time = extract_and_format(data_moisture, "time", "str")
        # f_data_moisture = extract_and_format(data_moisture, "value", "float")
        # f_data_temp = extract_and_format(data_temp, "value", "float")

        if not data_moisture or not data_temp:
            response_data = {"available": False}
            status_code = 404

            return status_code, bytes(json.dumps(response_data), "utf8"), []

    # Create the chart_data dictionary
    chart_data = {
        "available": True,
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
    # data_dataset.set_index('Timestamp', inplace=True)
    col_names = data_dataset.columns

    # index
    for item in data_dataset.index:
        f_data_time.append(item.to_pydatetime().strftime(
            '%Y-%m-%dT%H:%M:%S'))  # TODO:timezone is lost here!!!

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
    # f_data_time = data_pred.index.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S%z').tolist()=>love python
    for item in data_pred.index:
        f_data_time.append(
            item.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S%z'))
    f_data_moisture = data_pred["smoothed_values"].tolist()

    # Quick and dirty adjusting predictions to match sensor values TODO: ??? right approach ??? -> NO, THE THRESHOLD WILL BE WRONG!!!! -> OMG
    adjustment = 1
    # adjust_threshold = lambda currentPlot.threshold, adjustment: currentPlot.threshold - adjustment if currentPlot.sensor_kind == "tension" else currentPlot.threshold + adjustment
    is_tension_kind = _is_tension_kind(currentPlot.sensor_kind)
    def adjust_threshold(adjustment): return currentPlot.threshold - \
        adjustment if is_tension_kind else currentPlot.threshold + adjustment

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
        # Could be also just the value instead of annotations object
        "annotations": annotations,
        "permanentWiltingPoint": currentPlot.permanent_wilting_point,
        "fieldCapacityUpper": currentPlot.field_capacity_upper,
        "fieldCapacityLower": currentPlot.field_capacity_lower,
        "saturation": currentPlot.saturation,
        "threshold_cbar": currentPlot.threshold,
        "kind": "tension" if is_tension_kind else currentPlot.sensor_kind,
        "unit": currentPlot.sensor_unit
    }

    dynamic_threshold = None
    try:
        recommendation = actuation.get_irrigation_recommendation(currentPlot)
        if recommendation.get("available") and recommendation.get("threshold_cbar") is not None:
            dynamic_threshold = recommendation.get("threshold_cbar")
    except Exception:
        dynamic_threshold = None

    chart_data["threshold_cbar_dynamic"] = dynamic_threshold

    horizon_targets = [24, 48, 120, 168]
    available_horizons = []
    max_hours = None
    if isinstance(data_pred.index, pd.DatetimeIndex) and len(data_pred.index) > 0:
        now = pd.Timestamp(datetime.now().replace(microsecond=0))
        if TimeUtils.Timezone:
            now = now.tz_localize(TimeUtils.Timezone)
        now_cmp = now
        idx = data_pred.index
        if idx.tz is None and now.tzinfo is not None:
            now_cmp = now.tz_localize(None)
        elif idx.tz is not None and now.tzinfo is None:
            now_cmp = now.tz_localize(idx.tz)
        elif idx.tz is not None and now.tzinfo is not None and idx.tz != now.tzinfo:
            now_cmp = now.tz_convert(idx.tz)
        max_hours = (idx.max() - now_cmp).total_seconds() / 3600.0
        if max_hours < 0:
            max_hours = 0.0
        available_horizons = [
            h for h in horizon_targets
            if max_hours is None or max_hours + 1e-6 >= h
        ]

    chart_data["forecast_horizons_target"] = horizon_targets
    chart_data["forecast_horizons_available"] = available_horizons
    chart_data["forecast_horizon_max_hours"] = max_hours

    # Conditionally add 'moistureSeriesVol' if available
    # and 'f_data_moisture_vol' in locals() and f_data_moisture_vol is not None:
    if is_tension_kind:
        f_data_moisture_vol = data_pred["smoothed_values_vol"].tolist()
        chart_data["moistureSeriesVol"] = f_data_moisture_vol

    return 200, bytes(json.dumps(chart_data), "utf8"), []


usock.routerGET("/api/getPredictionChartData", getPredictionChartData)


def getWeatherForecast(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    lat, lon = _gps_from_plot(currentPlot)
    if lat is None or lon is None:
        return 200, bytes(json.dumps({
            "available": False,
            "reason": "missing_gps",
            "days": [],
        }), "utf8"), []

    today = datetime.utcnow().date()
    horizon_days = 4
    start_date = today.strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=horizon_days)).strftime("%Y-%m-%d")

    try:
        frame = create_model.fetch_weather_frame(
            lat,
            lon,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        return 200, bytes(json.dumps({
            "available": False,
            "reason": "fetch_failed",
            "error": str(exc),
            "days": [],
        }), "utf8"), []

    if frame is None or frame.empty or "Rain" not in frame.columns:
        return 200, bytes(json.dumps({
            "available": False,
            "reason": "no_data",
            "days": [],
        }), "utf8"), []

    try:
        frame.index = pd.to_datetime(frame.index)
    except Exception:
        pass

    rain_series = pd.to_numeric(frame["Rain"], errors="coerce").fillna(0.0)
    daily = rain_series.resample("D").sum()
    days = []
    for idx, (ts, rain) in enumerate(daily.items()):
        if idx >= horizon_days:
            break
        label = "Today" if idx == 0 else f"+{idx} days"
        days.append({
            "date": ts.date().isoformat() if hasattr(ts, "date") else str(ts),
            "label": label,
            "rain_mm": round(float(rain), 2),
            "icon": _weather_icon_from_rain(float(rain)),
        })

    return 200, bytes(json.dumps({
        "available": True,
        "days": days,
        "unit": "mm",
    }), "utf8"), []


usock.routerGET("/api/getWeatherForecast", getWeatherForecast)


def getPhenologySummary(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    crop_type = getattr(currentPlot, "crop_type", None) or "generic"
    planting_date = getattr(currentPlot, "planting_date", None)
    initial_gdd = getattr(currentPlot, "initial_gdd", 0.0)

    try:
        params = get_crop_params(crop_type)
    except Exception as exc:
        return 200, bytes(json.dumps({
            "available": False,
            "error": str(exc),
        }), "utf8"), []

    stage_rows = [
        {
            "stage": "Pre-emergence",
            "gdd_start": 0,
            "gdd_end": params.gdd_emergence,
            "kc_start": params.kc_ini,
            "kc_end": params.kc_ini,
            "delta_cbar": params.delta_pre_emergence,
        },
        {
            "stage": "Development",
            "gdd_start": params.gdd_emergence,
            "gdd_end": params.gdd_dev_end,
            "kc_start": params.kc_ini,
            "kc_end": params.kc_mid,
            "delta_cbar": params.delta_development,
        },
        {
            "stage": "Mid-season",
            "gdd_start": params.gdd_dev_end,
            "gdd_end": params.gdd_mid_end,
            "kc_start": params.kc_mid,
            "kc_end": params.kc_mid,
            "delta_cbar": params.delta_mid_season,
        },
        {
            "stage": "Late-season",
            "gdd_start": params.gdd_mid_end,
            "gdd_end": params.gdd_maturity,
            "kc_start": params.kc_mid,
            "kc_end": params.kc_end,
            "delta_cbar": params.delta_late_season,
        },
        {
            "stage": "Post-maturity",
            "gdd_start": params.gdd_maturity,
            "gdd_end": None,
            "kc_start": params.kc_ini,
            "kc_end": params.kc_ini,
            "delta_cbar": None,
        },
    ]

    payload = {
        "available": True,
        "crop_type": crop_type,
        "crop_name": params.name,
        "planting_date": planting_date,
        "initial_gdd": float(initial_gdd or 0.0),
        "gdd_thresholds": {
            "emergence": params.gdd_emergence,
            "development_end": params.gdd_dev_end,
            "mid_season_end": params.gdd_mid_end,
            "maturity": params.gdd_maturity,
        },
        "kc": {
            "kc_ini": params.kc_ini,
            "kc_mid": params.kc_mid,
            "kc_end": params.kc_end,
        },
        "stage_rows": stage_rows,
    }

    return 200, bytes(json.dumps(payload), "utf8"), []


usock.routerGET("/api/getPhenologySummary", getPhenologySummary)

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


def getIrrigationRecommendation(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    recommendation = actuation.get_irrigation_recommendation(currentPlot)
    return 200, bytes(json.dumps(recommendation), "utf8"), []


usock.routerGET("/api/getIrrigationRecommendation",
                getIrrigationRecommendation)


def getAlertStatus(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    alert_path = os.path.join(
        "data",
        "alerts",
        f"plot_{currentPlot.id}.latest.json",
    )
    notify_path = os.path.join(
        "data",
        "alerts",
        f"plot_{currentPlot.id}.notify.json",
    )
    if not os.path.exists(alert_path):
        return 200, bytes(json.dumps({"available": False}), "utf8"), []

    def _parse_utc(ts_value):
        if not ts_value:
            return None
        if isinstance(ts_value, str) and ts_value.endswith("Z"):
            ts_value = ts_value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(ts_value)
        except (TypeError, ValueError):
            return None

    def _load_notify_state():
        if not os.path.exists(notify_path):
            return {}
        try:
            with open(notify_path, "r") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _save_notify_state(state):
        os.makedirs(os.path.dirname(notify_path), exist_ok=True)
        with open(notify_path, "w") as handle:
            json.dump(state, handle, indent=2)

    try:
        with open(alert_path, "r") as handle:
            payload = json.load(handle)
        payload["available"] = True

        urgency = str(payload.get("urgency", "")).lower()
        alert_timestamp = payload.get("timestamp_utc")
        now_utc = datetime.utcnow().replace(microsecond=0)
        notify = False
        notify_headline = None
        notify_message = None
        notify_next_allowed_utc = None

        if urgency in ALERT_NOTIFY_URGENCIES and alert_timestamp:
            notify_state = _load_notify_state()
            last_notified = _parse_utc(notify_state.get("last_notified_utc"))
            last_alert_ts = notify_state.get("last_alert_timestamp_utc")

            next_allowed = None
            if last_notified is not None:
                next_allowed = last_notified + timedelta(
                    seconds=ALERT_NOTIFY_THROTTLE_SECONDS)

            if last_alert_ts != alert_timestamp and (
                next_allowed is None or now_utc >= next_allowed
            ):
                notify = True
                notify_headline = f"Alert: {urgency.upper()}"
                breach = payload.get("first_breach_horizon", "none")
                tension = payload.get("current_tension")
                threshold = payload.get("stress_threshold")
                tension_txt = f"{tension:.1f} cbar" if isinstance(
                    tension, (int, float)) else "--"
                threshold_txt = f"{threshold:.1f} cbar" if isinstance(
                    threshold, (int, float)) else "--"
                notify_message = (
                    f"Breach {breach}. Current {tension_txt}, threshold {threshold_txt}."
                )

                notify_state.update({
                    "last_notified_utc": now_utc.isoformat() + "Z",
                    "last_alert_timestamp_utc": alert_timestamp,
                    "last_urgency": urgency,
                })
                _save_notify_state(notify_state)

            if next_allowed is not None:
                notify_next_allowed_utc = next_allowed.isoformat() + "Z"

        payload["notify"] = notify
        payload["notify_headline"] = notify_headline
        payload["notify_message"] = notify_message
        payload["notify_throttle_seconds"] = ALERT_NOTIFY_THROTTLE_SECONDS
        payload["notify_next_allowed_utc"] = notify_next_allowed_utc
        return 200, bytes(json.dumps(payload), "utf8"), []
    except Exception as exc:
        return 200, bytes(json.dumps({
            "available": False,
            "error": str(exc),
        }), "utf8"), []


usock.routerGET("/api/getAlertStatus", getAlertStatus)

# Returns the senors kind: e.g. capacitive or tension


def getSensorKind(url, body):
    response_data = {"SensorKind": plot_manager.getCurrentPlot().sensor_kind}

    return 200, bytes(json.dumps(response_data), "utf8"), []


usock.routerGET("/api/getSensorKind", getSensorKind)


def getFarmRegistry(url, body):
    plots = plot_manager.getPlots()
    registry = []
    for pid, plot in plots.items():
        gps = getattr(plot, 'gps_info', None)
        if isinstance(gps, dict):
            lat = gps.get('latitude', gps.get('lattitude'))
            lon = gps.get('longitude')
        else:
            lat = None
            lon = None
        registry.append({
            "plot_id": pid,
            "name": getattr(plot, 'user_given_name', ''),
            "crop_type": getattr(plot, 'crop_type', ''),
            "planting_date": getattr(plot, 'planting_date', ''),
            "sensor_kind": getattr(plot, 'sensor_kind', ''),
            "latitude": lat,
            "longitude": lon,
            "config_path": getattr(plot, 'configPath', ''),
            "pending_sensors": bool(getattr(plot, 'device_and_sensor_ids_moisture', []) == []
                                    or getattr(plot, 'device_and_sensor_ids_temp', []) == []),
        })

    return 200, bytes(json.dumps({"farms": registry}), "utf8"), []


usock.routerGET("/api/getFarmRegistry", getFarmRegistry)


def getSensorRegistry(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    registry = _load_sensor_registry()
    entries = [item for item in registry if item.get(
        "plot_id") == getattr(currentPlot, "id", None)]
    return 200, bytes(json.dumps({
        "plot_id": getattr(currentPlot, "id", None),
        "zone_name": getattr(currentPlot, "zone_name", getattr(currentPlot, 'user_given_name', '')),
        "sensors": entries,
    }), "utf8"), []


usock.routerGET("/api/getSensorRegistry", getSensorRegistry)

# Frontend polls this to reload page when training is ready => only active for first round of training TODO: different plots


def isTrainingReady(url, body):
    response_data = {
        "isTrainingFinished": plot_manager.getCurrentPlot().training_finished}

    return 200, bytes(json.dumps(response_data), "utf8"), []


usock.routerGET("/api/isTrainingReady", isTrainingReady)

# Starts a thread that runs training, this thread will also start prediction afterwards


def startTraining(url, body):
    training_thread.start(plot_manager.getCurrentPlot())

    return 200, b"", []


usock.routerGET("/api/startTraining", startTraining)

# Returns tabId of current plot


def getCurrentPlot(url, body):
    response_data = {"currentPlot": plot_manager.getCurrentPlot().tab_number}

    return 200, bytes(json.dumps(response_data), "utf8"), []


usock.routerGET("/api/getCurrentPlot", getCurrentPlot)

# ------------------#


if __name__ == "__main__":
    # Load environment variables
    NetworkUtils.get_env()

    # Load all plots once on startup
    plot_manager.loadPlots()

    # Detect debug configuration on start, adjust globals accordingly
    if os.getenv("LOAD_DATA_FROM_CSV") == "True":
        for plot in plot_manager.Plots.values():
            plot.load_data_from_csv = True
    if os.getenv("SKIP_DATA_PREPROCESSING") == "True":
        create_model.skip_data_preprocessing = True
    if os.getenv("SKIP_TRAINING") == "True":
        create_model.skip_training = True
    if os.getenv("PERFORM_TRAINING") == "False":
        create_model.perform_training = False

    # Bound logs.log by SIZE (an active log's mtime is always recent, so the age-based
    # cleaner below never fires on it) - set up before anything logs
    setup_logging()

    # Get saved config from all plots and save it in objects
    getConfigsFromAllFiles()
    sync_sensor_registry_from_plots()

    # Start thread that deletes old models and data regularly to save memory.
    # file paths: prune old FILES, keep the folder tree (models/<plot>/... stays intact)
    file_cleanup_paths = ["models", "hyperband_dir", "data/subprocess_temp",
                          "catboost_info", "test-reports"]
    # dir globs: delete the WHOLE matching folder once old (per-run scratch)
    dir_cleanup_globs = ["tmp/tuning_*", "tmp/nn_*"]
    # file globs: prune only these files (data/debug also holds datasets - never rmtree it)
    file_cleanup_globs = ["data/debug/saved_variables_plot_*.pkl"]
    schedule_model_cleanup(file_cleanup_paths, dir_cleanup_globs,
                           file_cleanup_globs, interval_days=7)  # Check every week

    # Clean logs
    schedule_log_cleanup()

    # Former Start serving -> obsolete, now start in thread with recovery mechanism
    # usock.sockAddr = NetworkUtils.Proxy
    # usock.start() # will be "stuck" in here, code afterwards is not executed

    # Start serving in a dedicated thread -> no blocking, always
    server_thread = threading.Thread(
        target=usock.start_with_recovery, name="HTTP_Server")
    server_thread.daemon = False  # Keep alive until shutdown
    server_thread.start()
    print("Server started and running in thread:", server_thread.name)

    # DEBUG: directly start training for testing purposes, if production check configuration is present
    # training_thread.start(plot_manager.getCurrentPlot())

    # Start training for all untrained plots that have a config, heavy, only PRODUCTION, DEBUG
    if Auto_start_training:
        for p in plot_manager.getPlots().values():
            if p.configPath and not p.training_finished:
                training_thread.start(p)

    # Keep main thread alive
    try:
        while True:
            time.sleep(3600)  # Check every hour
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # Add cleanup logic here if needed
