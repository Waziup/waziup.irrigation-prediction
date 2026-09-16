# TODO: delete ports from URLs


#!/usr/bin/python
import os
import copy

# Configure noisy ML libraries before importing create_model/pycaret.
os.environ.setdefault("PYCARET_CUSTOM_LOGGING_LEVEL", "CRITICAL")
os.environ.setdefault("PYCARET_NO_LOGGING", "1")

import requests
from crops import get_crop_params
from crop_model import _vwc_for_tension, get_stress_threshold
from utils import NetworkUtils, TimeUtils
import plot_manager
import actuation
import csv
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import threading
import time
from urllib.parse import unquote, urlparse, parse_qs
import pandas as pd
import usock
import glob
import shutil
import logging
from logging.handlers import RotatingFileHandler
import pathlib
import numpy as np
from collections import defaultdict
from dateutil import parser
from farm_config import apply_farm_configs_to_plots, load_all_farms
from spaceiotbox_client import fetch_weather_frame
from spaceiotbox_client import diagnose_agro_climate_land
from spaceiotbox_satellite import (
    diagnose_eo_catalog,
    fetch_satellite_history,
)
from eo_observation import analyse_vegetation_history
from api_contract import json_safe
from operations_store import get_operations_store
from dashboard_contract import build_farm_dashboard

# Heavy ML modules are loaded only after startup configuration passes. This
# keeps missing/invalid farm settings from being hidden by dependency errors.
training_thread = None
create_model = None

# ---------------------#
# Path to the root of the code
PATH = os.path.dirname(os.path.abspath(__file__))

# Set the threshold to cleanup models to 3 months (approximately 90 days)
THRESHOLD_DAYS_CLEANUP = 90
# Per-run scratch dirs in tmp/ are dead after the next run; set to 90 to match if preferred
TMP_DIR_CLEANUP_DAYS = 2
ALERT_NOTIFY_THROTTLE_SECONDS = 3600
ALERT_NOTIFY_URGENCIES = {"critical", "advise", "watch"}
SENSOR_REGISTRY_PATH = os.path.join(PATH, "config", "sensor_registry.json")

TENSION_SENSOR_KEYWORDS = ("tension", "tensiometer", "cbar", "cb", "kpa")
VWC_SENSOR_KEYWORDS = ("vwc", "volumetric", "capacitive", "cap")

Auto_start_training = False  # TODO: set to false for production, then training is only started when user clicks on "start training" in UI, otherwise it is started directly when config is present, which can lead to long waiting times on page load if training is heavy
# ---------------------#


def _trace_event(event, plot=None, **details):
    """Emit concise lifecycle evidence in explicitly traced deployments."""
    if os.getenv("IRRIGATION_TRACE_EVENTS", "").strip().lower() not in {
            "1", "true", "yes", "on"}:
        return
    payload = {
        "event": event,
        "plot_id": getattr(plot, "stable_id", None) if plot is not None else None,
        "plot_name": getattr(plot, "user_given_name", None) if plot is not None else None,
        **details,
    }
    serialized = json.dumps(payload, default=str, sort_keys=True)
    logging.getLogger("irrigation.trace").info(
        "[IRRIGATION_TRACE] %s", serialized)
    print(f"[IRRIGATION_TRACE] {serialized}", flush=True)


def _load_sensor_registry():
    if not os.path.exists(SENSOR_REGISTRY_PATH):
        return []
    try:
        with open(SENSOR_REGISTRY_PATH, "r") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
    except (OSError, ValueError, TypeError):
        logging.getLogger(__name__).exception(
            "Unable to load sensor registry: %s", SENSOR_REGISTRY_PATH)
    return []


def _save_sensor_registry(records):
    os.makedirs(os.path.dirname(SENSOR_REGISTRY_PATH), exist_ok=True)
    with open(SENSOR_REGISTRY_PATH, "w") as handle:
        json.dump(records, handle, indent=2)


def _sensor_registry_key(plot_id, role, sensor_id):
    return f"{plot_id}:{role}:{sensor_id}"


def _classify_moisture_sensor(sensor_id):
    name = str(sensor_id or "").lower()
    if any(keyword in name for keyword in TENSION_SENSOR_KEYWORDS):
        return "tension"
    if any(keyword in name for keyword in VWC_SENSOR_KEYWORDS):
        return "vwc"
    return "unknown"


def _split_moisture_sensors(sensor_ids):
    tension_ids = []
    vwc_ids = []
    unknown_ids = []
    for sensor_id in sensor_ids:
        kind = _classify_moisture_sensor(sensor_id)
        if kind == "tension":
            tension_ids.append(sensor_id)
        elif kind == "vwc":
            vwc_ids.append(sensor_id)
        else:
            unknown_ids.append(sensor_id)
    return tension_ids, vwc_ids, unknown_ids


def _is_tension_kind(sensor_kind):
    return str(sensor_kind or "").lower() in ("tension", "both")


def _numeric_average(values):
    numbers = [
        float(value) for value in values
        if isinstance(value, (int, float)) and np.isfinite(value)
    ]
    return sum(numbers) / len(numbers) if numbers else None


def _json_bytes(payload):
    """Encode strict JSON, replacing non-finite numeric values with null.

    Python's json module otherwise writes NaN/Infinity tokens. Those tokens are
    accepted by Python again but rejected by browsers, leaving fetch consumers
    stuck in their loading state.
    """
    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [normalize(item) for item in value]
        if isinstance(value, np.generic):
            return normalize(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if value is pd.NA:
            return None
        return value

    return json.dumps(
        normalize(payload), allow_nan=False, default=str
    ).encode("utf-8")


def _build_sensor_registry_entries(plot, include_readings=None, online=None):
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    zone_name = (
        getattr(plot, "zone_name", None)
        or getattr(plot, "user_given_name", "")
        or f"plot_{getattr(plot, 'id', 'unknown')}"
    )
    gps = getattr(plot, "gps_info", {})
    try:
        lat = float(gps.get("latitude", gps.get("lattitude")))
        lon = float(gps.get("longitude"))
    except (AttributeError, TypeError, ValueError):
        lat, lon = None, None
    sensor_groups = [
        ("moisture", getattr(plot, "device_and_sensor_ids_moisture", [])),
        ("temperature", getattr(plot, "device_and_sensor_ids_temp", [])),
        ("flow", getattr(plot, "device_and_sensor_ids_flow", [])),
    ]
    include_readings = include_readings or {}
    records = []
    for role, sensor_ids in sensor_groups:
        for sensor_id in sensor_ids:
            reading = include_readings.get(role, {}).get(sensor_id)
            sensor_kind = getattr(plot, "sensor_kind", "")
            if role == "moisture" and str(sensor_kind).lower() == "both":
                sensor_kind = _classify_moisture_sensor(sensor_id)
            status = {
                "last_seen": now_iso if online is not False else None,
                "online": bool(online) if online is not None else False,
                "battery": None,
                "readings": [],
            }
            if reading is not None:
                status["readings"].append({
                    "timestamp": now_iso,
                    "value": reading,
                    "role": role,
                })
                status["last_seen"] = now_iso
                status["online"] = True
            records.append({
                "sensor_key": _sensor_registry_key(
                    getattr(plot, "id", "unknown"), role, sensor_id),
                "plot_id": getattr(plot, "id", None),
                "plot_name": getattr(plot, "user_given_name", ""),
                "zone_name": zone_name,
                "latitude": lat,
                "longitude": lon,
                "device_sensor_id": sensor_id,
                "role": role,
                "sensor_kind": sensor_kind,
                "status": status,
            })
    return records


def sync_sensor_registry_from_plots():
    registry = _load_sensor_registry()
    registry_index = {
        item.get("sensor_key"): item
        for item in registry if item.get("sensor_key")
    }
    merged = []
    for plot in plot_manager.getPlots().values():
        for entry in _build_sensor_registry_entries(plot):
            existing = registry_index.get(entry["sensor_key"], {})
            existing_status = (
                existing.get("status", {}) if isinstance(existing, dict) else {}
            )
            merged_status = entry["status"]
            if isinstance(existing_status, dict):
                merged_status["battery"] = existing_status.get("battery")
                merged_status["readings"] = (
                    existing_status.get("readings", [])
                    or merged_status["readings"]
                )
                if existing_status.get("last_seen"):
                    merged_status["last_seen"] = existing_status["last_seen"]
                merged_status["online"] = existing_status.get(
                    "online", merged_status["online"])
            entry["status"] = merged_status
            merged.append(entry)
    _save_sensor_registry(merged)


def update_sensor_registry_status(plot, reading_map=None, online=None):
    registry = _load_sensor_registry()
    registry_index = {
        item.get("sensor_key"): item
        for item in registry if item.get("sensor_key")
    }
    updated_entries = _build_sensor_registry_entries(
        plot, include_readings=reading_map or {}, online=online)
    for entry in updated_entries:
        existing = registry_index.get(entry["sensor_key"])
        if isinstance(existing, dict):
            status = existing.get("status", {})
            status = status if isinstance(status, dict) else {}
            new_status = entry["status"]
            new_status["battery"] = status.get("battery")
            prior_readings = status.get("readings", [])
            prior_readings = prior_readings if isinstance(prior_readings, list) else []
            new_status["readings"] = prior_readings + new_status["readings"]
            if status.get("last_seen") and new_status.get("last_seen") is None:
                new_status["last_seen"] = status["last_seen"]
            if online is None and "online" in status:
                new_status["online"] = status["online"]
            entry["status"] = new_status
    plot_id = getattr(plot, "id", None)
    registry = [item for item in registry if item.get("plot_id") != plot_id]
    registry.extend(updated_entries)
    _save_sensor_registry(registry)


def index(url, body=""):
    return 200, b"Salam Goloooo", []


usock.routerGET("/", index)

# ------------------#


def ui(url, body=''):
    request_path = unquote(urlparse(url).path)
    if not request_path.startswith('/ui/'):
        return 404, b"File not found", ["text/plain"]
    filename = request_path[len('/ui/'):] or 'index.html'
    ui_root = (pathlib.Path(PATH) / 'ui').resolve()
    file_path = (ui_root / filename).resolve()
    try:
        file_path.relative_to(ui_root)
    except ValueError:
        return 404, b"File not found", ["text/plain"]

    # ---------------#

    ext = file_path.suffix

    extMap = {
        '': 'application/octet-stream',
        '.manifest': 'text/cache-manifest',
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.svg':	'image/svg+xml',
        '.css':	'text/css',
        '.js': 'text/javascript',
        '.wasm': 'application/wasm',
        '.json': 'application/json',
        '.xml': 'application/xml',
    }

    if ext not in extMap:
        ext = ""

    conType = extMap[ext]

    # ---------------#

    try:
        with file_path.open(mode='rb') as file:
            return 200, file.read(), [conType]
    except (OSError, ValueError):
        return 404, b"File not found", ["text/plain"]


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
            if self.stop_thread.wait(self.check_interval):
                break

    def stop(self):
        self.stop_thread.set()

# setup function for log cleaner => TODO: changed function WITHOUT TESTING IT!!!!!!!!!!!!!!!!!
# Bound logs.log by size via rotation. Age-based cleanup cannot bound an actively
# written log (its mtime stays recent), so rotation is the real cap here. Configured
# once on the root logger; pycaret's basicConfig call is then a no-op (root has a handler).


def setup_logging():
    """Install one bounded, key-value production log handler."""
    root = logging.getLogger()
    if any(getattr(handler, "_irrigation_handler", False) for handler in root.handlers):
        return
    try:
        handler = RotatingFileHandler(
            "logs.log", maxBytes=30 * 1024 * 1024, backupCount=1)
        file_error = None
    except OSError as exc:
        # Read-only or host-owned volumes must not hide startup diagnostics.
        handler = logging.StreamHandler()
        file_error = exc
    handler._irrigation_handler = True
    handler.setFormatter(logging.Formatter(
        "%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s"))
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if file_error is not None:
        root.warning("file_logging_unavailable error=%s", file_error)
    # runtime-effective even after pycaret is already imported
    logging.getLogger("pycaret").setLevel(logging.WARNING)


def schedule_log_cleanup():
    logs_to_clean = [
        ("logs.log", 90),       # Python log file
        ("python_logs.log", 90)  # Docker log file
    ]

    # Start a thread for each log file
    cleaners = []
    for log_path, age_limit in logs_to_clean:
        thread_name = f"LogCleaner-{os.path.basename(log_path)}"
        cleaner = LogCleanerThread(
            file_path=log_path, age_limit_days=age_limit, name=thread_name)
        cleaner.daemon = True  # Run thread in the background
        cleaner.start()
        cleaners.append(cleaner)
    return cleaners


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
            if self.stop_event.wait(self.interval_days * 24 * 3600):
                break

    def stop(self):
        self.stop_event.set()

# Deletes files older than the threshold from the specified folder and its subfolders.


def delete_old_files(folder_path):
    current_time = time.time()
    threshold_time = current_time - THRESHOLD_DAYS_CLEANUP * 24 * 60 * 60
    protected = set()
    if pathlib.Path(folder_path).name == "models":
        # Never prune active/rollback artifacts referenced by model manifests.
        for manifest_path in pathlib.Path(folder_path).glob("*/model_manifest.json"):
            protected.add(manifest_path.resolve())
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for generation in (manifest.get("active"), manifest.get("previous")):
                    for artifact in (generation or {}).get("artifacts", []):
                        protected.add(pathlib.Path(artifact).resolve())
            except (OSError, ValueError, TypeError):
                logging.getLogger(__name__).exception(
                    "Invalid model manifest during cleanup: %s", manifest_path)

    # Traverse the directory, including subdirectories
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            # Skip .gitkeep, .gitignore and readme files
            if file_name in {".gitkeep", ".gitignore", "README.md"}:
                continue

            file_path = os.path.join(root, file_name)
            if pathlib.Path(file_path).resolve() in protected:
                continue
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

# Deletes only files matching a glob in shared folders that must not be removed.


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


def stop_background_workers(plots, cleaners=(), join_timeout=30):
    """Signal plot and maintenance workers, then wait briefly for shutdown."""
    threads = list(cleaners)
    for plot in plots.values():
        for name in ("prediction_thread", "training_thread"):
            worker = getattr(plot, name, None)
            if worker is not None and worker.is_alive():
                worker.stop()
                threads.append(worker)
    for worker in threads:
        if worker.is_alive():
            worker.join(timeout=join_timeout)

# Get URL of API from .env file => TODO: better with try catch than locals, getenv can still stop backend


def getApiUrl(url, body):
    url = NetworkUtils.ApiUrl

    if url not in (None, ''):
        parsed = urlparse(url)
        # Never expose HTTP basic-auth/user-info credentials to browser code.
        # Preserve the endpoint itself for compatibility with this legacy API.
        safe_netloc = parsed.netloc.rsplit('@', 1)[-1]
        data = parsed._replace(netloc=safe_netloc).geturl()
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
    # Preserve blank values so clearing an editable setting is intentional.
    parsed_data = parse_qs(body.decode('utf-8'), keep_blank_values=True)

    identifier = parsed_data.get('plot_id', parsed_data.get('currentPlot', [None]))[0]
    try:
        plot_manager.setPlot(identifier)
        current = plot_manager.getCurrentPlot()
        payload = {"status": "ok", "plot_id": current.stable_id,
                   "currentPlot": current.tab_number}
        return 200, bytes(json.dumps(payload), "utf8"), []
    except (KeyError, TypeError, ValueError) as exc:
        return 404, bytes(json.dumps({"status": "error", "error": str(exc)}), "utf8"), []


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

    # Include the currently selected plot for the UI to avoid an extra request
    try:
        current_plot_number = getattr(
            plot_manager.getCurrentPlot(), 'tab_number', None)
    except Exception:
        current_plot_number = None

    registry = plot_manager.registrySnapshot()
    response = {
        "tabnames": tab_name_array,
        "currentPlot": current_plot_number,
        "current_plot_id": registry["current_plot_id"],
        "current_farm_id": registry["current_farm_id"],
        "farms": registry["farms"],
        "plots": registry["plots"],
        "status_code": 200
    }

    return response["status_code"], bytes(json.dumps(response), "utf8"), []


usock.routerGET("/api/getPlots", getPlots)

# Add a plot during runtine TODO: finish


def addPlot(url, body):
    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))

    amount_tabs = parsed_data.get('tab_nr', [None])[0]
    try:
        plot_id, newfilename = plot_manager.addPlot(
            amount_tabs, farm_id=parsed_data.get('farm_id', [None])[0],
            name=parsed_data.get('name', [''])[0],
            area=parsed_data.get('area', [0])[0],
            area_unit=parsed_data.get('area_unit', ['m2'])[0])
    except (KeyError, TypeError, ValueError) as exc:
        return 400, bytes(json.dumps({"status": "error", "error": str(exc)}), "utf8"), []

    response = {
        "plot_id": plot_id,
        "filename": newfilename,
        "status_code": 200
    }
    _trace_event("plot.created", plot_manager.getCurrentPlot(),
                 area=parsed_data.get('area', [0])[0],
                 area_unit=parsed_data.get('area_unit', ['m2'])[0])

    return response["status_code"], bytes(json.dumps(response), "utf8"), []


usock.routerPOST("/api/addPlot", addPlot)

# Delete a plot during runtine TODO: ids adjust on remove, API call


def removePlot(url, body):
    # Parse the query parameters from Body
    parsed_data = parse_qs(body.decode('utf-8'))
    plot_to_be_removed = parsed_data.get('plot_id', parsed_data.get('currentPlot', [None]))[0]

    # Call function in plot manager
    try:
        removed_plot_id, oldfilename = plot_manager.removePlot(plot_to_be_removed)
    except KeyError as exc:
        return 404, bytes(json.dumps({"status": "error", "error": str(exc)}), "utf8"), []
    except ValueError as exc:
        return 409, bytes(json.dumps({"status": "error", "error": str(exc)}), "utf8"), []

    response = {
        "plot_id": removed_plot_id,
        "filename": oldfilename,
        "status_code": 200
    }

    return response["status_code"], bytes(json.dumps(response), "utf8"), []


usock.routerPOST("/api/removePlot", removePlot)


# Get historical sensor values from WaziGates API
def _set_config_update(url, body):
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

    currentPlot.device_and_sensor_ids_moisture = parsed_data.get(
        'selectedOptionsMoisture', [])
    currentPlot.device_and_sensor_ids_temp = parsed_data.get(
        'selectedOptionsTemp', [])
    currentPlot.device_and_sensor_ids_flow = parsed_data.get(
        'selectedOptionsFlow', [])
    currentPlot.device_and_sensor_ids_flow_confirmation = parsed_data.get(
        'selectedOptionsFlowConfirmation', [])
    currentPlot.flow_confirmation_mode = str(_get_first(
        'flow_confirmation_mode',
        getattr(currentPlot, 'flow_confirmation_mode', 'event'))
    ).strip().lower()
    # Parse JSON

    # Get data from forms
    errors = {}
    if currentPlot.flow_confirmation_mode not in {'event', 'cumulative'}:
        errors['flow_confirmation_mode'] = (
            "Flow confirmation mode must be event or cumulative.")
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
    threshold_value = _parse_float(
        _get_first('thres'), "threshold", errors)
    currentPlot.threshold_static = threshold_value
    currentPlot.threshold = threshold_value
    threshold_mode_raw = _get_first('threshold_mode', '')
    if not threshold_mode_raw:
        threshold_mode_raw = (
            'dynamic' if _get_first('use_dynamic_threshold', 'false').lower()
            in {'true', '1'} else 'static'
        )
    currentPlot.threshold_mode = threshold_mode_raw.strip().lower()
    if currentPlot.threshold_mode not in {'static', 'dynamic'}:
        errors['threshold_mode'] = "Threshold mode must be static or dynamic."
    plot_area_raw = parsed_data.get(
        'plot_area_m2', parsed_data.get('area', []))
    area_unit = _get_first('area_unit', getattr(currentPlot, 'area_unit', 'm2'))
    area_factors = {'m2': 1.0, 'ha': 10000.0, 'acre': 4046.8564224}
    if area_unit not in area_factors:
        errors['area_unit'] = "Area unit must be m2, ha, or acre."
    if plot_area_raw and str(plot_area_raw[0]).strip() != '':
        entered_area = _parse_float(
            plot_area_raw[0], "plot_area_m2", errors, 0.0)
        currentPlot.plot_area_m2 = (
            entered_area * area_factors.get(area_unit, 1.0)
            if entered_area is not None else None)
    else:
        currentPlot.plot_area_m2 = float(
            getattr(currentPlot, 'plot_area_m2', 0.0))
    currentPlot.application_efficiency = _parse_float(
        _get_first('application_efficiency', getattr(
            currentPlot, 'application_efficiency', 0.85)),
        "application_efficiency", errors, 0.01, 1.0)
    currentPlot.effective_rainfall_fraction = _parse_float(
        _get_first('effective_rainfall_fraction', getattr(
            currentPlot, 'effective_rainfall_fraction', 0.80)),
        "effective_rainfall_fraction", errors, 0.01, 1.0)
    currentPlot.irrigation_type = parsed_data.get(
        'irrigation_type', [getattr(currentPlot, 'irrigation_type', 'unknown')]
    )[0] or 'unknown'
    currentPlot.irrigation_mode = parsed_data.get(
        'irrigation_mode', [actuation.resolve_irrigation_mode(currentPlot)]
    )[0]
    if currentPlot.irrigation_mode not in {
            'automatic', 'approval_required', 'manual', 'advisory_only'}:
        errors['irrigation_mode'] = "Unsupported irrigation mode."
    currentPlot.look_ahead_time = _parse_float(
        _get_first('lookahead'), "look_ahead_time", errors, 0.0)

    # ``start`` is a legacy hidden training-window field, not the farmer's
    # planting date.  A missing planting date must therefore not make a save
    # fail.  Use an existing value or today's UTC date for a new plot.
    currentPlot.start_date = _get_first(
        'start', getattr(currentPlot, 'start_date', ''))
    if not currentPlot.start_date:
        currentPlot.start_date = datetime.now(timezone.utc).date().isoformat()
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
    currentPlot.permanent_wilting_point = _parse_float(
        _get_first('pwp'), "permanent_wilting_point", errors, 0)
    currentPlot.field_capacity_upper = _parse_float(
        _get_first('fcu'), "field_capacity_upper", errors, 0)
    currentPlot.field_capacity_lower = _parse_float(
        _get_first('fcl'), "field_capacity_lower", errors, 0)
    currentPlot.saturation = _parse_float(
        _get_first('sat'), "saturation", errors, 0)
    for field in ('permanent_wilting_point', 'field_capacity_upper',
                  'field_capacity_lower', 'saturation'):
        value = getattr(currentPlot, field)
        if value is not None and not np.isfinite(value):
            errors[field] = 'Must be a finite number.'
    try:
        currentPlot.soil_calibration = json.loads(_get_first('soil_calibration',
            json.dumps(getattr(currentPlot, 'soil_calibration', {}))))
        if not isinstance(currentPlot.soil_calibration, dict):
            raise ValueError('Calibration must be an object')
    except (TypeError, ValueError) as exc:
        errors['soil_calibration'] = str(exc)
        currentPlot.soil_calibration = {}
    currentPlot.crop_type = _get_first(
        'crop_type', getattr(currentPlot, 'crop_type', ''))
    if not currentPlot.crop_type:
        errors["crop_type"] = "Crop type is required."
    else:
        from crops import CROP_PARAMS
        if currentPlot.crop_type not in CROP_PARAMS:
            errors["crop_type"] = (
            "Select a supported crop from the crop library.")
    currentPlot.planting_date = _get_first(
        'planting_date', getattr(currentPlot, 'planting_date', ''))
    if currentPlot.planting_date:
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
    if currentPlot.irrigation_mode in {'automatic', 'approval_required'}:
        control_label = (
            "Automatic mode" if currentPlot.irrigation_mode == 'automatic'
            else "Approval-required mode")
        if not currentPlot.device_and_sensor_ids_flow:
            errors['actuator'] = f"{control_label} requires a configured actuator."
        elif any(len(str(value).split('/')) != 2 for value in
                 currentPlot.device_and_sensor_ids_flow):
            errors['actuator'] = "Use device-id/actuator-id format."
        if (currentPlot.irrigation_mode == 'automatic'
                and not currentPlot.device_and_sensor_ids_flow_confirmation):
            discovered = currentPlot.getConfirmationDeviceID(
                currentPlot.device_and_sensor_ids_flow)
            if discovered:
                currentPlot.device_and_sensor_ids_flow_confirmation = [
                    discovered]
            else:
                errors['flow_confirmation'] = (
                    "Automatic mode requires a flow confirmation sensor "
                    "(WaziGate xlpp channel 5).")
        if not currentPlot.plot_area_m2 or currentPlot.plot_area_m2 <= 0:
            errors['plot_area_m2'] = (
                f"{control_label} requires plot area to calculate irrigation volume.")

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
    if csv_data:
        try:
            from crop_model import _retention_curve_points
            _retention_curve_points(csv_data)
        except ValueError as exc:
            errors['soil_water_retention_curve'] = str(exc)
    if currentPlot.threshold_mode == 'dynamic' and len(csv_data) < 3:
        errors['dynamic_threshold'] = (
            "Dynamic mode requires at least three points in the existing "
            "soil-water retention curve.")
    if csv_data:
        try:
            from soil_calibration import validate_soil_calibration
            validate_soil_calibration(csv_data, currentPlot.field_capacity_lower,
                currentPlot.permanent_wilting_point, currentPlot.saturation,
                currentPlot.soil_calibration)
        except (TypeError, ValueError) as exc:
            errors['soil_calibration'] = str(exc)

    # Organize the variables into a dictionary
    gps_lat = gps_lat if gps_lat is not None else 0.0
    gps_lon = gps_lon if gps_lon is not None else 0.0
    currentPlot.gps_info = {
        "latitude": gps_lat,
        "longitude": gps_lon,
        "lattitude": gps_lat,
    }
    try:
        currentPlot.timezone = TimeUtils.get_timezone(gps_lat, gps_lon) or "UTC"
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not determine timezone for plot %s; using UTC",
            getattr(currentPlot, "id", "?"),
        )
        currentPlot.timezone = "UTC"
    currentPlot.configuration_source = "ui"

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

    # Keep the live object on the same normalized representation that is
    # written to JSON. Leaving the raw CSV string in memory made retention-curve
    # readiness fail until the next process restart.
    currentPlot.soil_water_retention_curve = csv_data

    data = {

        "DeviceAndSensorIdsMoisture": currentPlot.device_and_sensor_ids_moisture,
        "DeviceAndSensorIdsTemp": currentPlot.device_and_sensor_ids_temp,
        "DeviceAndSensorIdsFlow": currentPlot.device_and_sensor_ids_flow,
        "DeviceAndSensorIdsFlowConfirmation": (
            currentPlot.device_and_sensor_ids_flow_confirmation),
        "Flow_confirmation_mode": currentPlot.flow_confirmation_mode,
        "Sensor_kind": currentPlot.sensor_kind,
        "Name": currentPlot.user_given_name,
        "Plot_id": currentPlot.stable_id,
        "Farm_id": currentPlot.farm_id,
        "Zone_name": getattr(currentPlot, 'zone_name', currentPlot.user_given_name),
        # "Gps_info": {"lattitude": currentPlot.gps_info['lattitude'], "longitude": currentPlot.gps_info['longitude']},
        # "Gps_info": currentPlot.gps_info,
        "Gps_info": {
            "latitude": gps_lat,
            "longitude": gps_lon,
            "lattitude": gps_lat,
        },
        "Enable_experimental_ndre_kc": bool(getattr(
            currentPlot, 'enable_experimental_ndre_kc', False)),
        "Timezone": currentPlot.timezone,
        "Configuration_source": "ui",
        "Slope": currentPlot.slope,
        "Threshold": getattr(currentPlot, 'threshold_static', currentPlot.threshold),
        "Threshold_mode": currentPlot.threshold_mode,
        "Application_efficiency": currentPlot.application_efficiency,
        "Effective_rainfall_fraction": currentPlot.effective_rainfall_fraction,
        "Plot_area_m2": currentPlot.plot_area_m2,
        "Plot_area_unit": area_unit,
        "Irrigation_type": getattr(currentPlot, 'irrigation_type', 'unknown'),
        "Irrigation_mode": actuation.resolve_irrigation_mode(currentPlot),
        "Look_ahead_time": currentPlot.look_ahead_time,
        "Start_date": currentPlot.start_date,
        "Period": currentPlot.period,
        "Soil_type": currentPlot.soil_type,
        "Soil_water_retention_curve": csv_data,  # Use the parsed CSV data
        "PermanentWiltingPoint": currentPlot.permanent_wilting_point,
        "Soil_calibration": getattr(currentPlot, "soil_calibration", {}),
        "FieldCapacityUpper": currentPlot.field_capacity_upper,
        "FieldCapacityLower": currentPlot.field_capacity_lower,
        "Saturation": currentPlot.saturation,
        "Soil_texture_class": getattr(currentPlot, 'soil_texture_class', None),
        "Crop_type": getattr(currentPlot, 'crop_type', ''),
        "Planting_date": getattr(currentPlot, 'planting_date', ''),
        "Harvest_date": getattr(currentPlot, 'harvest_date', None),
        "Initial_gdd": float(getattr(currentPlot, 'initial_gdd', 0.0)),
        "Farm_data_bundle": {

            "crop": {
                "type": getattr(currentPlot, 'crop_type', ''),
                "planting_date": getattr(currentPlot, 'planting_date', ''),
                "initial_gdd": float(getattr(currentPlot, 'initial_gdd', 0.0)),
            },
            "soil": {
                "type": currentPlot.soil_type,
                "texture_class": getattr(currentPlot, 'soil_texture_class', None),
                "threshold_mode": currentPlot.threshold_mode,
                "static_threshold": getattr(currentPlot, 'threshold_static', currentPlot.threshold),
            },
            "sensors": {
                "moisture": currentPlot.device_and_sensor_ids_moisture,
                "temperature": currentPlot.device_and_sensor_ids_temp,
                "flow": currentPlot.device_and_sensor_ids_flow,
                "flow_confirmation": (
                    currentPlot.device_and_sensor_ids_flow_confirmation),
                "flow_confirmation_mode": currentPlot.flow_confirmation_mode,
            },
            "water_demand": {
                "plot_area_m2": currentPlot.plot_area_m2,
                "application_efficiency": currentPlot.application_efficiency,
                "effective_rainfall_fraction": currentPlot.effective_rainfall_fraction,
                "demand_horizon_hours": currentPlot.look_ahead_time,
            },
            "weather_snapshot": getattr(currentPlot, 'weather_snapshot', None),
            "satellite_snapshot": getattr(currentPlot, 'satellite_snapshot', None),
            "model_snapshot": getattr(currentPlot, 'model_snapshot', None),
        },
        "Pending_sensors": pending_sensors,
        "Pending_sensors_reason": (
            "Missing moisture or temperature sensors"
            if pending_sensors else ""
        )
    }

    currentPlot.area_unit = data["Plot_area_unit"]
    plot_manager.updateCurrentPlotMetadata(
        currentPlot.user_given_name, currentPlot.plot_area_m2, currentPlot.area_unit)

    # Save the JSON data to the file
    config_path = plot_manager.getCurrentConfig()
    temporary_path = config_path + ".tmp"
    with open(temporary_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        json_file.flush()
        os.fsync(json_file.fileno())
    os.replace(temporary_path, config_path)

    _trace_event(
        "plot.configuration.saved", currentPlot,
        latitude=gps_lat, longitude=gps_lon,
        moisture_sensors=currentPlot.device_and_sensor_ids_moisture,
        temperature_sensors=currentPlot.device_and_sensor_ids_temp,
        crop=currentPlot.crop_type, planting_date=currentPlot.planting_date,
        threshold_mode=currentPlot.threshold_mode,
        pending_sensors=pending_sensors,
    )

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


_CONFIG_UPDATE_FIELDS = (
    "device_and_sensor_ids_moisture", "device_and_sensor_ids_temp",
    "device_and_sensor_ids_flow",
    "device_and_sensor_ids_flow_confirmation",
    "flow_confirmation_mode",
    "user_given_name", "zone_name", "sensor_kind", "gps_info",
    "slope", "threshold", "threshold_static", "threshold_mode",
    "application_efficiency", "effective_rainfall_fraction",
    "plot_area_m2", "area_unit", "irrigation_type",
    "irrigation_mode", "look_ahead_time", "start_date", "period",
    "soil_calibration", "soil_type", "soil_texture_class", "permanent_wilting_point",
    "field_capacity_upper", "field_capacity_lower", "saturation",
    "soil_water_retention_curve", "crop_type", "planting_date",
    "initial_gdd", "enable_experimental_ndre_kc",
    "timezone",
    "configuration_source", "farm_data_bundle",
)


def setConfig(url, body):
    """Apply a settings update transactionally to runtime and disk.

    The legacy handler necessarily assigns many fields while parsing the HTML
    form.  Snapshotting its exact mutable surface prevents a rejected or failed
    request from leaving worker threads on a configuration that was never
    committed.
    """
    plot = plot_manager.getCurrentPlot()
    snapshot = {
        name: copy.deepcopy(getattr(plot, name))
        for name in _CONFIG_UPDATE_FIELDS if hasattr(plot, name)
    }
    config_path = plot_manager.getCurrentConfig()
    try:
        with open(config_path, "rb") as handle:
            previous_config = handle.read()
    except FileNotFoundError:
        previous_config = None

    def restore():
        for name, value in snapshot.items():
            setattr(plot, name, value)
        # Restore registry metadata if the failure occurred after that durable
        # update but before the whole settings transaction completed.
        try:
            plot_manager.updateCurrentPlotMetadata(
                snapshot.get("user_given_name"),
                snapshot.get("plot_area_m2"), snapshot.get("area_unit"))
        except (OSError, KeyError, ValueError):
            logging.getLogger(__name__).exception(
                "Failed to restore plot registry after settings rollback")
        if previous_config is not None:
            rollback_path = config_path + ".rollback"
            with open(rollback_path, "wb") as handle:
                handle.write(previous_config)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(rollback_path, config_path)
        else:
            # A failed first-time save must not leave a configuration file that
            # the runtime state and registry have already rolled back.
            try:
                os.unlink(config_path)
            except FileNotFoundError:
                pass

    try:
        response = _set_config_update(url, body)
    except Exception:
        restore()
        raise
    if response[0] >= 400:
        restore()
    return response


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
                plots[i].device_and_sensor_ids_flow_confirmation = []

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
                confirmation = data.get(
                    'DeviceAndSensorIdsFlowConfirmation', [])
                plots[i].device_and_sensor_ids_flow_confirmation = (
                    confirmation if isinstance(confirmation, list) else [])

            # Get data from forms
            plots[i].user_given_name = data.get('Name', [])
            plots[i].owner = data.get('Owner', getattr(plots[i], 'owner', ''))
            plots[i].configuration_source = data.get(
                'Configuration_source', getattr(plots[i], 'configuration_source', 'legacy'))
            plots[i].timezone = data.get(
                'Timezone', getattr(plots[i], 'timezone', 'UTC')) or 'UTC'
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
            plots[i].enable_experimental_ndre_kc = bool(
                data.get('Enable_experimental_ndre_kc', False))
            plots[i].slope = float(data.get('Slope', []))
            plots[i].threshold = float(data.get('Threshold', []))
            plots[i].threshold_static = float(
                data.get('Threshold', plots[i].threshold))
            plots[i].threshold_mode = str(data.get(
                'Threshold_mode',
                'dynamic' if data.get('Use_dynamic_threshold', False) else 'static',
            )).strip().lower()
            plots[i].field_capacity_vwc = data.get('Field_capacity_vwc')
            plots[i].wilting_point_vwc = data.get('Wilting_point_vwc')
            plots[i].root_depth_m = data.get('Root_depth_m')
            plots[i].sensor_depth_m = data.get('Sensor_depth_m')
            plots[i].depletion_fraction = data.get('Depletion_fraction')
            plots[i].stage_depletion_fractions = data.get(
                'Stage_depletion_fractions', {}) or {}
            plots[i].stage_thresholds_cbar = data.get(
                'Stage_thresholds_cbar', {}) or {}
            plots[i].threshold_hysteresis_cbar = float(data.get(
                'Threshold_hysteresis_cbar', 0.0) or 0.0)
            plots[i].application_efficiency = float(
                data.get('Application_efficiency', 0.85))
            plots[i].effective_rainfall_fraction = float(
                data.get('Effective_rainfall_fraction', 0.80))
            plots[i].plot_area_m2 = float(data.get('Plot_area_m2', 0))
            plots[i].irrigation_type = data.get(
                'Irrigation_type', 'unknown') or 'unknown'
            plots[i].irrigation_mode = data.get(
                'Irrigation_mode', getattr(plots[i], 'irrigation_mode', '')) or ''
            plots[i].flow_confirmation_mode = str(data.get(
                'Flow_confirmation_mode',
                getattr(plots[i], 'flow_confirmation_mode', 'event'))
            ).strip().lower()
            plots[i].look_ahead_time = float(data.get('Look_ahead_time', []))
            plots[i].start_date = data.get('Start_date', [])
            plots[i].period = int(data.get('Period', []))
            plots[i].soil_type = data.get('Soil_type', [])
            plots[i].soil_texture_class = data.get('Soil_texture_class', None)
            plots[i].soil_calibration = data.get('Soil_calibration', {}) or {}
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

            # Phenology configuration
            plots[i].crop_type = data.get('Crop_type', '')
            plots[i].planting_date = data.get('Planting_date', '')
            plots[i].harvest_date = data.get('Harvest_date')
            try:
                plots[i].initial_gdd = float(data.get('Initial_gdd', 0.0))
            except (TypeError, ValueError):
                plots[i].initial_gdd = 0.0
            plots[i].farm_data_bundle = data.get('Farm_data_bundle', None)

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
                    currentPlot.application_efficiency,
                    currentPlot.effective_rainfall_fraction,
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

            training_missing = _training_prerequisites(currentPlot)

            # Construct the response data
            response_data = {

                "DeviceAndSensorIdsMoisture": currentPlot.device_and_sensor_ids_moisture,
                "DeviceAndSensorIdsTemp": currentPlot.device_and_sensor_ids_temp,
                "DeviceAndSensorIdsFlow": currentPlot.device_and_sensor_ids_flow,
                "DeviceAndSensorIdsFlowConfirmation": (
                    currentPlot.device_and_sensor_ids_flow_confirmation),
                "Flow_confirmation_mode": getattr(
                    currentPlot, 'flow_confirmation_mode', 'event'),
                "Sensor_kind": currentPlot.sensor_kind,
                "Name": currentPlot.user_given_name,
                "Plot_id": currentPlot.stable_id,
                "Farm_id": currentPlot.farm_id,
                "Zone_name": getattr(currentPlot, 'zone_name', currentPlot.user_given_name),
                "Gps_info": currentPlot.gps_info,
        "Enable_experimental_ndre_kc": bool(getattr(
            currentPlot, 'enable_experimental_ndre_kc', False)),
                "Slope": currentPlot.slope,
                "Threshold": getattr(currentPlot, 'threshold_static', currentPlot.threshold),
                "Threshold_mode": getattr(currentPlot, 'threshold_mode', 'static'),
                "Application_efficiency": currentPlot.application_efficiency,
                "Effective_rainfall_fraction": currentPlot.effective_rainfall_fraction,
                "Plot_area_m2": currentPlot.plot_area_m2,
                "Plot_area_unit": getattr(currentPlot, 'area_unit', 'm2'),
                "Irrigation_type": getattr(currentPlot, 'irrigation_type', 'unknown'),
                "Irrigation_mode": actuation.resolve_irrigation_mode(currentPlot),
                "Look_ahead_time": currentPlot.look_ahead_time,
                "Start_date": currentPlot.start_date,
                "Period": currentPlot.period,
                "Soil_type": currentPlot.soil_type,
                "Soil_water_retention_curve": currentPlot.soil_water_retention_curve,
                "PermanentWiltingPoint": currentPlot.permanent_wilting_point,
        "Soil_calibration": getattr(currentPlot, "soil_calibration", {}),
                "FieldCapacityUpper": currentPlot.field_capacity_upper,
                "FieldCapacityLower": currentPlot.field_capacity_lower,
                "Saturation": currentPlot.saturation,
                "Soil_texture_class": getattr(currentPlot, 'soil_texture_class', None),
                "Crop_type": getattr(currentPlot, 'crop_type', ''),
                "Planting_date": getattr(currentPlot, 'planting_date', ''),
                "Harvest_date": getattr(currentPlot, 'harvest_date', None),
                "Initial_gdd": float(getattr(currentPlot, 'initial_gdd', 0.0)),
                "Farm_data_bundle": getattr(currentPlot, 'farm_data_bundle', None),
                "Training_readiness": {
                    "ready": not training_missing,
                    "missing": training_missing,
                    "data_source": (
                        "csv" if currentPlot.load_data_from_csv else "gateway"
                    ),
                },
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
    config_present = bool(currentPlot.getConfigFromFile())
    response_data = {
        "activeIrrigation": bool(
            config_present and currentPlot.device_and_sensor_ids_flow),
    }
    status_code = 200

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


def group_sensor_data(
    sensor_lists,
    agg_func=lambda vals: sum(vals)/len(vals),
    resample_interval="30T",
    timezone_name="UTC",
):
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

    # Convert using the requesting plot's timezone; never shared mutable state.
    df.index = df.index.tz_convert(timezone_name)

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
    request_values = parse_qs(body.decode('utf-8'))
    currentPlot = plot_manager.getCurrentPlot()
    mode = actuation.resolve_irrigation_mode(currentPlot)
    if mode == "advisory_only" or not actuation._has_actuator_support(currentPlot):
        return 400, bytes(json.dumps({"status": "error", "message": "Manual irrigation is disabled for advisory-only plots or plots without an actuator."}), "utf8"), []
    recommendation = actuation.get_irrigation_recommendation(currentPlot)
    if not (recommendation.get("action") or {}).get("should_irrigate", False):
        return 409, bytes(json.dumps({
            "status": "not_required",
            "message": "The current crop-water recommendation does not require irrigation."
        }), "utf8"), []
    amount = (recommendation.get("water") or {}).get(
        "recommended_volume_m3")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = 0.0
    if not np.isfinite(amount) or amount <= 0:
        return 400, bytes(json.dumps({
            "status": "error",
            "message": (
                "Calculated irrigation volume is unavailable or zero. "
                "Check plot area, ET0, rainfall, and demand settings.")
        }), "utf8"), []

    request_key = request_values.get('idempotency_key', [None])[0]
    if not request_key:
        request_key = f"manual:{currentPlot.stable_id}:{amount}:{pd.Timestamp.now(tz='UTC').floor('min').isoformat()}"
    initial_status = "pending_approval" if mode == "approval_required" else "approved"
    operation, created = get_operations_store().create_operation(
        idempotency_key=request_key, plot_id=currentPlot.stable_id,
        farm_id=currentPlot.farm_id, plot_name=currentPlot.user_given_name,
        source="manual", mode=mode, status=initial_status, amount_m3=amount)
    if not created:
        return 200, bytes(json.dumps({"status": "duplicate", "operation": operation}), "utf8"), []
    if mode == "approval_required":
        return 202, bytes(json.dumps({"status": "pending_approval", "operation": operation}), "utf8"), []

    response = actuation.execute_operation_command(currentPlot, operation)
    updated = get_operations_store().get_operation(operation["operation_id"])
    if not response:
        return 400, bytes(json.dumps({"status": "error", "message": "Irrigation command failed.", "operation": updated}), "utf8"), []
    return 200, bytes(json.dumps({"status": "success", "amount": amount,
                                 "operation": updated}), "utf8"), []


usock.routerPOST("/api/irrigateManually", irrigateManually)


def _plot_by_stable_id(plot_id):
    for plot in plot_manager.getPlots().values():
        if getattr(plot, "stable_id", None) == plot_id:
            return plot
    return None


def listOperations(url, body):
    query = parse_qs(urlparse(url).query)
    first = lambda key, default=None: query.get(key, [default])[0]
    try:
        limit = int(first("limit", 100))
    except (TypeError, ValueError):
        return 400, bytes(json.dumps({"error": "limit must be an integer"}), "utf8"), []
    registry = plot_manager.registrySnapshot()
    operations = get_operations_store().list_operations(
        farm_id=first("farm_id", registry.get("current_farm_id")),
        plot_id=first("plot_id"), status=first("status"),
        search=first("search"), limit=limit)
    return 200, bytes(json.dumps({"operations": operations, "count": len(operations)}), "utf8"), []


usock.routerGET("/api/operations", listOperations)


def operationEvents(url, body):
    query = parse_qs(urlparse(url).query)
    operation_id = query.get("operation_id", [None])[0]
    if not operation_id:
        return 400, bytes(json.dumps({"error": "operation_id is required"}), "utf8"), []
    return 200, bytes(json.dumps({"events": get_operations_store().events(operation_id)}), "utf8"), []


usock.routerGET("/api/operationEvents", operationEvents)


def createIrrigationSchedule(url, body):
    values = parse_qs(body.decode("utf-8"))
    first = lambda key, default=None: values.get(key, [default])[0]
    plot_id = first("plot_id", getattr(plot_manager.getCurrentPlot(), "stable_id", None))
    plot = _plot_by_stable_id(plot_id)
    if plot is None:
        return 404, bytes(json.dumps({"error": "Unknown plot_id"}), "utf8"), []
    try:
        amount = float(first("amount_m3"))
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("Irrigation amount must be finite and greater than zero")
        planned_start = pd.Timestamp(first("planned_start"))
        if pd.isna(planned_start):
            raise ValueError("planned_start is required and must be a valid timestamp")
        planned_end_raw = first("planned_end")
        planned_end = pd.Timestamp(planned_end_raw) if planned_end_raw else None
        if planned_end is not None and pd.isna(planned_end):
            raise ValueError("planned_end must be a valid timestamp")
        if planned_end is not None and planned_end <= planned_start:
            raise ValueError("planned_end must be after planned_start")
        mode = actuation.resolve_irrigation_mode(plot)
        status = "pending_approval" if mode == "approval_required" else "planned"
        key = first("idempotency_key", f"schedule:{plot_id}:{planned_start.isoformat()}:{amount}")
        operation, created = get_operations_store().create_operation(
            idempotency_key=key, plot_id=plot_id, farm_id=plot.farm_id,
            plot_name=plot.user_given_name, source="schedule", mode=mode,
            status=status, amount_m3=amount, planned_start=planned_start.isoformat(),
            planned_end=planned_end.isoformat() if planned_end is not None else None)
    except (TypeError, ValueError) as exc:
        return 400, bytes(json.dumps({"error": str(exc)}), "utf8"), []
    return (201 if created else 200), bytes(json.dumps({"operation": operation,
                                                        "created": created}), "utf8"), []


usock.routerPOST("/api/irrigationSchedules", createIrrigationSchedule)


def approveIrrigation(url, body):
    values = parse_qs(body.decode("utf-8"))
    operation_id = values.get("operation_id", [None])[0]
    store = get_operations_store()
    try:
        operation, changed = store.transition(operation_id, "approved",
                                              {"actor": "farm_owner"})
    except KeyError as exc:
        return 404, bytes(json.dumps({"error": str(exc)}), "utf8"), []
    except ValueError as exc:
        return 409, bytes(json.dumps({"error": str(exc)}), "utf8"), []
    if not changed:
        return 200, bytes(json.dumps({"operation": operation, "duplicate": True}), "utf8"), []
    if actuation.schedule_window_error(operation) == 'schedule_not_due':
        return 200, bytes(json.dumps({"operation": operation,
                                     "message": "Approved and waiting for its scheduled start."}), "utf8"), []
    plot = _plot_by_stable_id(operation["plot_id"])
    if plot is None or not actuation._has_actuator_support(plot):
        operation = store.transition(operation_id, "failed",
                                     {"error": "plot_or_actuator_unavailable"})[0]
        return 409, bytes(json.dumps({"operation": operation}), "utf8"), []
    response = actuation.execute_operation_command(plot, operation)
    updated = store.get_operation(operation_id)
    return (200 if response else 409), bytes(json.dumps({"operation": updated}), "utf8"), []


usock.routerPOST("/api/approveIrrigation", approveIrrigation)


def declineIrrigation(url, body):
    values = parse_qs(body.decode("utf-8"))
    operation_id = values.get("operation_id", [None])[0]
    try:
        operation, changed = get_operations_store().transition(
            operation_id, "declined", {"actor": "farm_owner",
                                        "reason": values.get("reason", [""])[0]})
        return 200, bytes(json.dumps({"operation": operation,
                                     "duplicate": not changed}), "utf8"), []
    except KeyError as exc:
        return 404, bytes(json.dumps({"error": str(exc)}), "utf8"), []
    except ValueError as exc:
        return 409, bytes(json.dumps({"error": str(exc)}), "utf8"), []


usock.routerPOST("/api/declineIrrigation", declineIrrigation)


def _today_for_farm(registry, farm_id):
    farm = next((item for item in registry.get("farms", [])
                 if item.get("farm_id") == farm_id), {})
    timezone = farm.get("timezone") or "UTC"
    try:
        return pd.Timestamp.now(tz=timezone).date().isoformat()
    except (TypeError, ValueError):
        return pd.Timestamp.now(tz="UTC").date().isoformat()


def getTodaysPlan(url, body):
    query = parse_qs(urlparse(url).query)
    registry = plot_manager.registrySnapshot()
    farm_id = query.get("farm_id", [registry["current_farm_id"]])[0]
    today = query.get("date", [_today_for_farm(registry, farm_id)])[0]
    farm = next((item for item in registry["farms"] if item["farm_id"] == farm_id), None)
    if farm is None:
        return 404, bytes(json.dumps({"error": "Unknown farm_id"}), "utf8"), []
    try:
        operations = get_operations_store().list_operations(
            farm_id=farm_id, today=today, timezone_name=farm.get("timezone") or "UTC", limit=500)
    except ValueError as exc:
        return 400, bytes(json.dumps({"error": str(exc)}), "utf8"), []
    included = [item for item in operations if item["status"] not in {"declined", "failed"}]
    total = round(sum(float(item["amount_m3"] or 0) for item in included), 3)
    return 200, bytes(json.dumps({"date": today, "farm_id": farm_id,
                                 "planned_water_m3": total,
                                 "operations": operations}), "utf8"), []


usock.routerGET("/api/todaysIrrigationPlan", getTodaysPlan)


def getActiveAlerts(url, body):
    query = parse_qs(urlparse(url).query)
    farm_id = query.get("farm_id", [plot_manager.registrySnapshot()["current_farm_id"]])[0]
    alerts = get_operations_store().active_alerts(farm_id)
    return 200, bytes(json.dumps({"farm_id": farm_id, "alerts": alerts,
                                 "count": len(alerts)}), "utf8"), []


usock.routerGET("/api/activeAlerts", getActiveAlerts)


def getFarmDashboard(url, body):
    query = parse_qs(urlparse(url).query)
    registry = plot_manager.registrySnapshot()
    farm_id = query.get("farm_id", [registry["current_farm_id"]])[0]
    farm = next((item for item in registry["farms"] if item["farm_id"] == farm_id), None)
    if farm is None:
        return 404, bytes(json.dumps({"error": "Unknown farm_id"}), "utf8"), []
    plot_records = [item for item in registry["plots"] if item["farm_id"] == farm_id]
    runtime_plots = {getattr(plot, "stable_id", None): plot
                     for plot in plot_manager.getPlots().values()}
    recommendations = {}
    for record in plot_records:
        plot = runtime_plots.get(record["plot_id"])
        if plot is not None:
            try:
                recommendations[record["plot_id"]] = actuation.get_irrigation_recommendation(plot)
            except Exception as exc:
                recommendations[record["plot_id"]] = {
                    "available": False, "reason": str(exc)}
        else:
            recommendations[record["plot_id"]] = {
                "available": False, "reason": "No completed prediction cycle."}
    store = get_operations_store()
    alerts = store.active_alerts(farm_id)
    today = _today_for_farm(registry, farm_id)
    operations = store.list_operations(farm_id=farm_id, today=today,
        timezone_name=farm.get("timezone") or "UTC", limit=500)
    payload = build_farm_dashboard(
        farm=farm, plot_records=plot_records, runtime_plots=runtime_plots,
        recommendations=recommendations, alerts=alerts, operations=operations)
    return 200, _json_bytes(payload), []


usock.routerGET("/api/farmDashboard", getFarmDashboard)

# Get latest values for dashboard


def getValuesForDashboard(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    # Load config, to get latest changes
    currentPlot.config = currentPlot.read_config()

    if not currentPlot.device_and_sensor_ids_temp or not currentPlot.device_and_sensor_ids_moisture:
        _trace_event("sensors.readings.unavailable", currentPlot,
                     reason="sensor_selection_incomplete")
        update_sensor_registry_status(currentPlot, online=False)
        response_data = {"available": False}
        currentPlot.dashboard_snapshot = response_data
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
        # Cache dashboard and sensor snapshots on the Plot object so they are
        # available to `setConfig()` and other endpoints.
        currentPlot.dashboard_snapshot = dashboard_data
        currentPlot.sensor_snapshot = {
            "temperature_readings": temp_readings,
            "moisture_readings": moisture_readings,
            "dashboard": dashboard_data,
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

        currentPlot.dashboard_snapshot = dashboard_data
        currentPlot.sensor_snapshot = {
            "temperature_readings": temp_readings,
            "moisture_readings": moisture_readings,
            "dashboard": dashboard_data,
        }

    dashboard_data["data_source"] = (
        "CSV file" if currentPlot.load_data_from_csv else "WaziGate IoT API"
    )
    dashboard_data["recorded_or_live"] = (
        "recorded" if currentPlot.load_data_from_csv else "live"
    )

    reading_map = {"moisture": {}, "temperature": {}, "flow": {}}
    for sensor_id, value in moisture_readings.items():
        reading_map["moisture"][sensor_id] = value
    for sensor_id, value in temp_readings.items():
        reading_map["temperature"][sensor_id] = value
    update_sensor_registry_status(
        currentPlot, reading_map=reading_map, online=True)

    _trace_event(
        "sensors.readings.loaded", currentPlot,
        source=dashboard_data["data_source"],
        moisture_channels=list(moisture_readings),
        temperature_channels=list(temp_readings),
        moisture_average=dashboard_data.get("moisture_average"),
        temperature_average=dashboard_data.get("temp_average"),
    )

    return 200, bytes(json.dumps(dashboard_data), "utf8"), []


usock.routerGET("/api/getValuesForDashboard", getValuesForDashboard)


def _chart_vwc_series(plot, tension_values):
    """Return JSON-safe VWC fractions aligned with a tension chart series."""
    curve = getattr(plot, "soil_water_retention_curve", None)
    converted = []
    for value in tension_values:
        try:
            parsed = float(value)
            result = _vwc_for_tension(parsed, curve) if np.isfinite(parsed) else None
            converted.append(round(float(result), 4))
        except (TypeError, ValueError, KeyError, IndexError):
            converted.append(None)
    return converted


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
        if "Time" not in data.columns and "timestamp" in data.columns:
            data = data.rename(columns={"timestamp": "Time"})

        # extract series from key value pairs
        f_data_time = data["Time"].tolist()
        f_data_moisture = extract_and_format_csv(data, "tension")
        f_data_temp = extract_and_format_csv(data, "soil_temp")

        # unite similar series
        # f_data_moisture = [item for sublist in f_data_moisture for item in sublist]

        # Make sure that the data is not empty
        if not f_data_moisture or not f_data_temp:
            response_data = {"available": False}
            currentPlot.sensor_snapshot = response_data
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

        plot_timezone = TimeUtils.for_plot(currentPlot)
        f_data_time, f_data_moisture = group_sensor_data(
            data_moisture, timezone_name=plot_timezone)
        f_data_time, f_data_temp = group_sensor_data(
            data_temp, timezone_name=plot_timezone)

        # extract series from key value pairs
        # f_data_time = extract_and_format(data_moisture, "time", "str")
        # f_data_moisture = extract_and_format(data_moisture, "value", "float")
        # f_data_temp = extract_and_format(data_temp, "value", "float")

        if not data_moisture or not data_temp:
            response_data = {"available": False}
            currentPlot.sensor_snapshot = response_data
            status_code = 404

            return status_code, bytes(json.dumps(response_data), "utf8"), []

    # Create the chart_data dictionary
    static_threshold = getattr(
        currentPlot, "threshold_static", currentPlot.threshold)
    try:
        static_threshold = float(static_threshold)
    except (TypeError, ValueError):
        static_threshold = None
    if static_threshold is not None and not np.isfinite(static_threshold):
        static_threshold = None
    pipeline = getattr(currentPlot, "pipeline_result", None)
    crop_state = getattr(pipeline, "crop_state", None)
    active_threshold = (
        getattr(crop_state, "stress_threshold_cbar", None)
        if crop_state is not None else static_threshold
    )
    active_threshold_mode = (
        getattr(crop_state, "threshold_mode", None)
        if crop_state is not None else None
    ) or getattr(currentPlot, "threshold_mode", "static")

    chart_data = {
        "available": True,
        "timestamps": f_data_time,
        "temperatureSeries": f_data_temp,
        "moistureSeries": f_data_moisture,
        "moistureSeriesVol": (
            _chart_vwc_series(currentPlot, f_data_moisture)
            if _is_tension_kind(currentPlot.sensor_kind) else []
        ),
        "unit": currentPlot.sensor_unit,
        "kind": "tension" if _is_tension_kind(currentPlot.sensor_kind) else currentPlot.sensor_kind,
        "threshold_cbar": active_threshold,
        "threshold_cbar_static": static_threshold,
        "threshold_mode": active_threshold_mode,
        "threshold_reason": getattr(crop_state, "threshold_reason", ""),
        "saturation": currentPlot.saturation,
        "fieldCapacityLower": currentPlot.field_capacity_lower,
        "fieldCapacityUpper": currentPlot.field_capacity_upper,
        "permanentWiltingPoint": currentPlot.permanent_wilting_point,
        "data_source": (
            "CSV file"
            if currentPlot.load_data_from_csv else "WaziGate IoT API"
        ),
        "data_file": (
            getattr(currentPlot, "data_from_csv", None)
            if currentPlot.load_data_from_csv else None
        ),
    }

    currentPlot.sensor_snapshot = chart_data

    return 200, _json_bytes(chart_data), []


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
    pipeline = getattr(currentPlot, "pipeline_result", None)
    crop_state = getattr(pipeline, "crop_state", None)
    active_threshold_mode = getattr(
        crop_state, "threshold_mode",
        getattr(currentPlot, "threshold_mode", "static"))

    # Get prediction data for chart of current plot
    data_pred = currentPlot.get_predictions()

    if data_pred is False:
        _trace_event("prediction.unavailable", currentPlot,
                     reason="no_trained_model_forecast")
        response_data = {
            "available": False,
            "model": False,
            "reason": "No trained model forecast is available for this plot.",
        }
        return 200, bytes(json.dumps(response_data), "utf8"), []

    # Extract specific columns into lists TODO: timezone lost here!!!
    f_data_time = []
    # f_data_time = data_pred.index.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S%z').tolist()=>love python
    for item in data_pred.index:
        f_data_time.append(
            item.to_pydatetime().strftime('%Y-%m-%dT%H:%M:%S%z'))
    # Keep the chart contract JSON-safe if a model emits a non-finite point.
    moisture_values = pd.to_numeric(
        data_pred["smoothed_values"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
    if moisture_values.isna().any():
        moisture_values = moisture_values.interpolate(
            method="linear", limit_direction="both")
    f_data_moisture = [
        float(value) if pd.notna(value) and np.isfinite(float(value)) else None
        for value in moisture_values
    ]

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
        "annotations": (
            annotations if active_threshold_mode != "dynamic" else {"yaxis": []}),
        "permanentWiltingPoint": currentPlot.permanent_wilting_point,
        "fieldCapacityUpper": currentPlot.field_capacity_upper,
        "fieldCapacityLower": currentPlot.field_capacity_lower,
        "saturation": currentPlot.saturation,
        "threshold_cbar": (
            crop_state.stress_threshold_cbar
            if crop_state is not None else currentPlot.threshold),
        "kind": "tension" if is_tension_kind else currentPlot.sensor_kind,
        "unit": currentPlot.sensor_unit,
        "threshold_mode": active_threshold_mode,
        "threshold_reason": getattr(crop_state, "threshold_reason", ""),
    }

    static_threshold = getattr(
        currentPlot, "threshold_static", currentPlot.threshold)
    try:
        static_threshold_value = float(static_threshold)
    except (TypeError, ValueError):
        static_threshold_value = None
    if static_threshold_value is not None and np.isfinite(static_threshold_value):
        chart_data["threshold_cbar_static"] = round(static_threshold_value, 1)
    else:
        chart_data["threshold_cbar_static"] = None

    if chart_data["threshold_cbar_static"] is not None:
        chart_data["threshold_series_static"] = [
            chart_data["threshold_cbar_static"]
        ] * len(data_pred.index)
    else:
        chart_data["threshold_series_static"] = []

    active_threshold_value = chart_data["threshold_cbar"]
    threshold_points = getattr(
        pipeline, "stress_threshold_timestamps", {}) if pipeline is not None else {}
    dynamic_thresholds = {}
    for timestamp, value in (threshold_points or {}).items():
        try:
            key = pd.Timestamp(timestamp)
            idx = pd.DatetimeIndex(data_pred.index)
            if idx.tz is None and key.tzinfo is not None:
                key = key.tz_localize(None)
            elif idx.tz is not None and key.tzinfo is None:
                key = key.tz_localize(idx.tz)
            elif idx.tz is not None and key.tzinfo is not None:
                key = key.tz_convert(idx.tz)
            dynamic_thresholds[key] = float(value)
        except (TypeError, ValueError):
            continue
    if dynamic_thresholds:
        projected = pd.Series(dynamic_thresholds, dtype=float).sort_index()
        chart_data["threshold_series"] = projected.reindex(
            data_pred.index, method="ffill").fillna(
                active_threshold_value).round(3).tolist()
    elif active_threshold_value is not None:
        try:
            active_threshold_float = float(active_threshold_value)
        except (TypeError, ValueError):
            active_threshold_float = np.nan
        chart_data["threshold_series"] = (
            [active_threshold_float] * len(data_pred.index)
            if np.isfinite(active_threshold_float) else []
        )
    else:
        chart_data["threshold_series"] = []

    # Chart horizons mirror the actual model output instead of fixed 24/48/120h
    # labels that may not exist at the configured cadence.
    pipeline_result = getattr(currentPlot, "pipeline_result", None)
    horizon_targets = []
    if pipeline_result is not None:
        horizon_targets = [
            float(label[:-1])
            for label in getattr(pipeline_result, "tension_forecast", {})
            if str(label).endswith("h")
        ]
    available_horizons = []
    max_hours = None
    if isinstance(data_pred.index, pd.DatetimeIndex) and len(data_pred.index) > 0:
        now = pd.Timestamp(datetime.now().replace(microsecond=0))
        timezone_name = TimeUtils.for_plot(currentPlot)
        if timezone_name and now.tzinfo is None:
            now = now.tz_localize(timezone_name)
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
        available_horizons = horizon_targets

    chart_data["forecast_horizons_target"] = horizon_targets
    chart_data["forecast_horizons_available"] = available_horizons
    chart_data["forecast_horizon_max_hours"] = max_hours

    if is_tension_kind:
        if "smoothed_values_vol" in data_pred:
            raw_vwc = pd.to_numeric(
                data_pred["smoothed_values_vol"], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan)
            chart_data["moistureSeriesVol"] = [
                round(float(value), 4) if pd.notna(value) else None
                for value in raw_vwc
            ]
        else:
            chart_data["moistureSeriesVol"] = _chart_vwc_series(
                currentPlot, f_data_moisture)

    currentPlot.model_snapshot = chart_data

    _trace_event(
        "prediction.loaded", currentPlot,
        points=len(f_data_time), threshold_mode=active_threshold_mode,
        threshold_cbar=chart_data.get("threshold_cbar"),
        horizon_hours=max_hours,
    )

    return 200, _json_bytes(chart_data), []


usock.routerGET("/api/getPredictionChartData", getPredictionChartData)


def _gps_from_plot(plot):
    gps = getattr(plot, "gps_info", None)
    if not isinstance(gps, dict):
        return None, None
    try:
        latitude = float(gps.get("latitude", gps.get("lattitude")))
        longitude = float(gps.get("longitude"))
    except (TypeError, ValueError):
        return None, None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None, None
    return latitude, longitude


def _farm_weather_location(plot):
    """Resolve the one weather reference point shared by a farm's plots."""
    registry = plot_manager.registrySnapshot()
    farm_id = getattr(plot, "farm_id", None)
    farm = next((item for item in registry.get("farms", [])
                 if item.get("farm_id") == farm_id), None)
    if farm is None:
        return None, None, None
    try:
        latitude = float(farm.get("latitude"))
        longitude = float(farm.get("longitude"))
    except (TypeError, ValueError):
        return farm, None, None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return farm, None, None
    return farm, latitude, longitude


def _publish_farm_weather_snapshot(farm_id, payload):
    """Keep one farm forecast snapshot visible to all of its runtime plots."""
    for plot in plot_manager.getPlots().values():
        if getattr(plot, "farm_id", None) == farm_id:
            plot.weather_snapshot = payload


def _weather_icon_from_rain(rain_mm):
    if rain_mm >= 10:
        return "rainy"
    if rain_mm > 0:
        return "rainy_light"
    return "partly_cloudy_day"


def getWeatherForecast(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    farm, lat, lon = _farm_weather_location(currentPlot)
    if lat is None or lon is None:
        _trace_event("weather.forecast.skipped", currentPlot,
                     scope="farm", farm_id=getattr(currentPlot, "farm_id", None),
                     reason="missing_farm_gps")
        return 200, bytes(json.dumps({
            "available": False,
            "reason": "missing_farm_gps",
            "days": [],
        }), "utf8"), []

    today = pd.Timestamp.now(tz="UTC").date()
    # Display five forecast days by default; deployments can choose another
    # positive horizon without changing application code.
    try:
        horizon_days = max(1, int(os.getenv("WEATHER_FORECAST_DAYS", "5")))
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid WEATHER_FORECAST_DAYS; using 5")
        horizon_days = 5
    start_date = today.strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=horizon_days - 1)).strftime("%Y-%m-%d")
    _trace_event(
        "weather.forecast.started", currentPlot, scope="farm",
        farm_id=farm.get("farm_id"), farm_name=farm.get("name"),
        latitude=lat, longitude=lon, start=start_date, end=end_date)

    try:
        frame = fetch_weather_frame(
            lat,
            lon,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        _trace_event(
            "weather.forecast.failed", currentPlot, scope="farm",
            farm_id=farm.get("farm_id"), error=str(exc))
        return 200, bytes(json.dumps({
            "available": False,
            "reason": "fetch_failed",
            "error": str(exc),
            "days": [],
        }), "utf8"), []

    if frame is None or frame.empty:
        _trace_event("weather.forecast.unavailable", currentPlot,
                     reason="provider_returned_no_data")
        payload = {
            "available": False,
            "reason": "no_data",
            "days": [],
        }
        _publish_farm_weather_snapshot(farm.get("farm_id"), payload)
        return 200, bytes(json.dumps(payload), "utf8"), []

    try:
        frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(
            TimeUtils.for_plot(currentPlot))
    except Exception:
        pass

    def numeric(field):
        if field not in frame:
            return pd.Series(index=frame.index, dtype=float)
        return pd.to_numeric(frame[field], errors="coerce")

    rain_series = numeric("Rain").fillna(0.0).clip(lower=0.0)
    temperature = numeric("Temperature")
    humidity = numeric("Humidity")
    windspeed = numeric("Windspeed").clip(lower=0.0)
    winddirection = numeric("Winddirection")
    daily_index = pd.DatetimeIndex(frame.resample("D").size().index)
    days = []
    for idx, ts in enumerate(daily_index):
        if idx >= horizon_days:
            break
        end = ts + pd.Timedelta(days=1)
        rain = float(rain_series[(rain_series.index >= ts) & (rain_series.index < end)].sum())
        temp_day = temperature[(temperature.index >= ts) & (temperature.index < end)].dropna()
        humidity_day = humidity[(humidity.index >= ts) & (humidity.index < end)].dropna()
        wind_day = windspeed[(windspeed.index >= ts) & (windspeed.index < end)].dropna()
        direction_day = winddirection[(winddirection.index >= ts) & (winddirection.index < end)].dropna()
        direction = None
        if not direction_day.empty:
            radians = np.deg2rad(direction_day.to_numpy(dtype=float))
            direction = float((np.rad2deg(np.arctan2(
                np.sin(radians).mean(), np.cos(radians).mean())) + 360) % 360)
        label = "Today" if idx == 0 else f"+{idx} days"
        days.append({
            "date": ts.date().isoformat() if hasattr(ts, "date") else str(ts),
            "label": label,
            "rain_mm": round(float(rain), 2),
            "temperature_c": round(float(temp_day.mean()), 1) if not temp_day.empty else None,
            "temperature_min_c": round(float(temp_day.min()), 1) if not temp_day.empty else None,
            "temperature_max_c": round(float(temp_day.max()), 1) if not temp_day.empty else None,
            "humidity_percent": round(float(humidity_day.mean()), 1) if not humidity_day.empty else None,
            "wind_speed": round(float(wind_day.mean()), 1) if not wind_day.empty else None,
            "wind_gust": round(float(wind_day.max()), 1) if not wind_day.empty else None,
            "wind_direction_degrees": round(direction, 0) if direction is not None else None,
            "icon": _weather_icon_from_rain(float(rain)),
        })

    payload = {
        "available": True,
        "scope": "farm",
        "farm_id": farm.get("farm_id"),
        "farm_name": farm.get("name"),
        "coordinates": {"latitude": lat, "longitude": lon},
        "days": days,
        "today": days[0] if days else None,
        "units": {"rain": "mm", "temperature": "°C",
                  "humidity": "%", "wind_speed": "km/h"},
        "source": frame.attrs.get("provider", "unknown"),
        "source_detail": frame.attrs.get("fallback_reason"),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    _publish_farm_weather_snapshot(farm.get("farm_id"), payload)
    _trace_event(
        "weather.forecast.completed", currentPlot, scope="farm",
        farm_id=farm.get("farm_id"), farm_name=farm.get("name"),
        provider=payload["source"], days=len(days),
        fallback_reason=frame.attrs.get("fallback_reason"),
    )
    return 200, bytes(json.dumps(payload), "utf8"), []


usock.routerGET("/api/getWeatherForecast", getWeatherForecast)


def getEOObservation(url, body):
    """Return crop-agnostic EO history for the selected plot."""
    currentPlot = plot_manager.getCurrentPlot()
    lat, lon = _gps_from_plot(currentPlot)
    if lat is None or lon is None:
        _trace_event("eo.observation.skipped", currentPlot,
                     reason="missing_gps")
        return 200, _json_bytes({
            "available": False,
            "reason": "Plot GPS coordinates are unavailable.",
        }), []
    try:
        lookback_days = max(30, int(os.getenv("EO_OBSERVATION_LOOKBACK_DAYS", "365")))
    except ValueError:
        lookback_days = 365
    _trace_event("eo.observation.started", currentPlot, scope="plot",
                 latitude=lat, longitude=lon, lookback_days=lookback_days)
    try:
        history = fetch_satellite_history(
            lat,
            lon,
            lookback_days=lookback_days,
            limit=100,
            include_ndre=True,
        )
        payload = analyse_vegetation_history(history)
        source_values = []
        if isinstance(history, pd.DataFrame) and "source" in history:
            source_values = sorted({
                str(value) for value in history["source"].dropna().tolist()
            })
        provider = (
            " + ".join(
                "SpaceIoTBox agro-climate/land" if value == "agro_climate"
                else "SpaceIoTBox EO/STAC" if value == "eo_stac"
                else value
                for value in source_values
            )
            or "SpaceIoTBox agro-climate/land + EO/STAC"
        )
        greening = payload.get("greening") or {}
        payload.update({
            "provider": provider,
            "lookback_days": lookback_days,
            "crop_identity_confirmed": False,
            "phenology_inferred": bool(
                getattr(currentPlot, "crop_type", "")
                and greening.get("detected")
                and greening.get("confidence") in {"moderate", "high"}
            ),
        })
    except Exception as exc:
        payload = {
            "available": False,
            "provider": "SpaceIoTBox agro-climate/land + EO/STAC",
            "reason": f"EO observation retrieval failed: {exc}",
        }
    currentPlot.satellite_snapshot = payload
    _trace_event(
        "eo.observation.completed", currentPlot, scope="plot",
        available=payload.get("available", False),
        provider=payload.get("provider"),
        observations=payload.get("valid_observations", 0),
        reason=payload.get("reason"),
    )
    return 200, _json_bytes(json_safe(payload)), []


usock.routerGET("/api/getEOObservation", getEOObservation)


def getSpaceIoTBoxDiagnostics(url, body):
    """Return sanitized diagnostics suitable for a provider issue report."""
    currentPlot = plot_manager.getCurrentPlot()
    lat, lon = _gps_from_plot(currentPlot)
    if lat is None or lon is None:
        return 200, _json_bytes({
            "available": False,
            "reason": "Plot GPS coordinates are unavailable.",
        }), []
    payload = {
        "available": True,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "plot_id": getattr(currentPlot, "stable_id", currentPlot.id),
        "credentials_included": False,
        "agro_climate": diagnose_agro_climate_land(lat, lon),
        "eo_catalog": diagnose_eo_catalog(lat, lon),
        "weather_fallback_policy": "Open-Meteo fills missing weather coverage only.",
        "eo_fallback_policy": "No non-SpaceIoTBox EO fallback is enabled in this version.",
    }
    return 200, _json_bytes(json_safe(payload)), []


usock.routerGET("/api/getSpaceIoTBoxDiagnostics", getSpaceIoTBoxDiagnostics)


def getPhenologySummary(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    crop_type = getattr(currentPlot, "crop_type", None) or ""
    planting_date = getattr(currentPlot, "planting_date", None)
    initial_gdd = getattr(currentPlot, "initial_gdd", 0.0)

    inferred_crop_state = None
    phenology_reference_date = planting_date
    phenology_reference_source = (
        "farmer_reported_planting_date" if planting_date else None)
    phenology_reference_confidence = "reported" if planting_date else None
    phenology_reference_interval = None
    if not planting_date and crop_type:
        inferred_crop_state = actuation.get_runtime_crop_state(currentPlot)
        if inferred_crop_state is not None:
            phenology_reference_date = getattr(
                inferred_crop_state, "phenology_reference_date", None)
            phenology_reference_source = getattr(
                inferred_crop_state, "phenology_reference_source", None)
            phenology_reference_confidence = getattr(
                inferred_crop_state, "phenology_reference_confidence", None)
            phenology_reference_interval = getattr(
                inferred_crop_state, "phenology_reference_interval", None)

    if not phenology_reference_date:
        payload = {
            "available": False,
            "status": "not_planted_or_unknown",
            "crop_type": crop_type,
            "planting_date": None,
            "growth_stage": None,
            "reason": (
                "Planting date is not available and EO did not establish a "
                "reliable crop-emergence reference. Crop-stage, GDD, and "
                "dynamic-threshold calculations remain disabled."
            ),
        }
        currentPlot.phenology_snapshot = payload
        _trace_event("phenology.unavailable", currentPlot,
                     reason="missing_planting_date_and_no_reliable_eo_emergence")
        return 200, bytes(json.dumps(payload), "utf8"), []

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
        },
        {
            "stage": "Development",
            "gdd_start": params.gdd_emergence,
            "gdd_end": params.gdd_dev_end,
            "kc_start": params.kc_ini,
            "kc_end": params.kc_mid,
        },
        {
            "stage": "Mid-season",
            "gdd_start": params.gdd_dev_end,
            "gdd_end": params.gdd_mid_end,
            "kc_start": params.kc_mid,
            "kc_end": params.kc_mid,
        },
        {
            "stage": "Late-season",
            "gdd_start": params.gdd_mid_end,
            "gdd_end": params.gdd_maturity,
            "kc_start": params.kc_mid,
            "kc_end": params.kc_end,
        },
        {
            "stage": "Post-maturity",
            "gdd_start": params.gdd_maturity,
            "gdd_end": None,
            "kc_start": params.kc_ini,
            "kc_end": params.kc_ini,
        },
    ]

    threshold_mode = getattr(currentPlot, "threshold_mode", "static")
    if threshold_mode not in {"static", "dynamic"}:
        threshold_mode = "static"
    try:
        threshold_baseline = float(getattr(
            currentPlot, "threshold_static", currentPlot.threshold))
    except (TypeError, ValueError):
        threshold_baseline = None
    if threshold_baseline is not None and (
            not np.isfinite(threshold_baseline) or threshold_baseline <= 0):
        threshold_baseline = None
    representative_gdd = [
        0.0,
        params.gdd_emergence,
        params.gdd_dev_end,
        params.gdd_mid_end,
        params.gdd_maturity,
    ]
    pipeline = getattr(currentPlot, "pipeline_result", None)
    crop_state = getattr(pipeline, "crop_state", None) or inferred_crop_state
    from phenology_engine import compute_kc_gdd
    stage_farm = actuation._build_runtime_farm_config(currentPlot)
    stage_et0 = getattr(crop_state, "et0_today_mm", None)
    for row, stage_gdd in zip(stage_rows, representative_gdd):
        threshold = threshold_baseline
        threshold_source = "field_static"
        if threshold_mode == "dynamic":
            details = getattr(crop_state, "threshold_details", {}) or {}
            threshold_source = details.get("source", "unavailable")
            if row["stage"] == "Post-maturity":
                threshold = None
                threshold_source = "inactive_season"
            elif threshold_source != "unavailable":
                try:
                    stage_etc = (float(stage_et0) * compute_kc_gdd(stage_gdd, crop_type)
                                 if stage_et0 is not None else None)
                    threshold = get_stress_threshold(
                        stage_farm, stage_gdd, stage_etc)
                except (TypeError, ValueError):
                    threshold = None
            else:
                threshold = None
        row["threshold_cbar"] = threshold
        row["threshold_source"] = threshold_source
        row["threshold_basis"] = ("At current ET0; varies with weather"
                                  if threshold_mode == "dynamic" and threshold is not None
                                  else None)

    payload = {
        "available": True,
        "crop_type": crop_type,
        "crop_name": params.name,
        "planting_date": planting_date,
        "phenology_reference_date": phenology_reference_date,
        "phenology_reference_source": phenology_reference_source,
        "phenology_reference_confidence": phenology_reference_confidence,
        "phenology_reference_interval": phenology_reference_interval,
        "initial_gdd": float(initial_gdd or 0.0),
        "threshold_mode": threshold_mode,
        "threshold_baseline_cbar": threshold_baseline,
        "evaluated_at": getattr(pipeline, "calculated_at", None),
        "threshold_active_cbar": getattr(
            crop_state, "stress_threshold_cbar", threshold_baseline),
        "threshold_details": getattr(crop_state, "threshold_details", None),
        "current_stage": getattr(crop_state, "growth_stage_name", None),
        "current_gdd": getattr(crop_state, "gdd_cumulative", None),
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

    currentPlot.phenology_snapshot = payload

    _trace_event(
        "phenology.completed", currentPlot,
        crop=crop_type, stage=payload.get("current_stage"),
        reference_date=phenology_reference_date,
        reference_source=phenology_reference_source,
        gdd=payload.get("current_gdd"), threshold_mode=threshold_mode,
        threshold_cbar=payload.get("threshold_active_cbar"),
    )

    return 200, bytes(json.dumps(payload), "utf8"), []


usock.routerGET("/api/getPhenologySummary", getPhenologySummary)

# get values from create_model.py if models had been trained


def getThreshold(url, body):
    # Get current plot (selected in UI)
    currentPlot = plot_manager.getCurrentPlot()

    # Get prediction data for chart of current plot
    threshold_timestamp = currentPlot.get_threshold_timestamp()

    if threshold_timestamp is False:
        response_data = {
            "threshold": False,
            "available": bool(getattr(currentPlot, "training_finished", False)
                               or getattr(currentPlot, "pipeline_result", None)),
            "reason": "No completed forecast is available for this plot yet.",
        }
        return 200, bytes(json.dumps(response_data), "utf8"), []

    else:
        timestamp_data = {
            "timestamp": str(threshold_timestamp)
        }

        return 200, bytes(json.dumps(timestamp_data), "utf-8"), []


usock.routerGET("/api/getThreshold", getThreshold)


def getIrrigationRecommendation(url, body):
    currentPlot = plot_manager.getCurrentPlot()
    # Always retain the rich crop/water/freshness contract consumed by the UI.
    # When a pipeline cycle exists, merge its exact decision into that payload
    # so the UI and actuator use the same urgency and breach timing.
    recommendation = actuation.get_irrigation_recommendation(currentPlot)
    pipeline_result = getattr(currentPlot, "pipeline_result", None)
    if pipeline_result is not None and pipeline_result.recommendation is not None:
        decision = pipeline_result.to_dict().get("recommendation") or {}
        for key, value in decision.items():
            recommendation.setdefault(key, value)
        recommendation.setdefault("decision", decision)
    return 200, _json_bytes(recommendation), []


usock.routerGET("/api/getIrrigationRecommendation",
                getIrrigationRecommendation)


def getPipelineState(url, body):
    """Return the latest unified model, crop-state, and decision result."""
    currentPlot = plot_manager.getCurrentPlot()
    pipeline_result = getattr(currentPlot, "pipeline_result", None)
    if pipeline_result is None:
        # Return the concrete cache/lifecycle failure instead of hiding it.
        return 404, bytes(json.dumps({
            "available": False,
            "reason": getattr(
                currentPlot,
                "pipeline_cache_reason",
                None,
            ) or "No completed pipeline cycle for the selected plot.",
            "cache_status": getattr(currentPlot, "pipeline_cache_status", "empty"),
        }), "utf8"), []

    payload = pipeline_result.to_dict()
    payload["available"] = pipeline_result.error is None
    recommendation = payload.get("recommendation") or {}
    payload["recommendation_available"] = bool(recommendation) and recommendation.get("urgency") != "error"
    payload["recommendation_error"] = recommendation.get("error_message")
    return 200, _json_bytes(payload), []


usock.routerGET("/api/getPipelineState", getPipelineState)


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
            parsed = datetime.fromisoformat(ts_value)
            # Legacy notification records without an offset also represent UTC.
            return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None
                    else parsed.astimezone(timezone.utc))
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
        now_utc = datetime.now(timezone.utc).replace(microsecond=0)
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
                    "last_notified_utc": now_utc.isoformat().replace("+00:00", "Z"),
                    "last_alert_timestamp_utc": alert_timestamp,
                    "last_urgency": urgency,
                })
                _save_notify_state(notify_state)
                next_allowed = now_utc + timedelta(seconds=ALERT_NOTIFY_THROTTLE_SECONDS)

            if next_allowed is not None:
                notify_next_allowed_utc = next_allowed.isoformat().replace("+00:00", "Z")

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
    return 200, bytes(json.dumps(plot_manager.registrySnapshot()), "utf8"), []


usock.routerGET("/api/getFarmRegistry", getFarmRegistry)


def createFarm(url, body):
    values = parse_qs(body.decode('utf-8'))
    first = lambda key, default='': values.get(key, [default])[0]
    try:
        gateway_id = NetworkUtils.get_gateway_id()
    except (requests.RequestException, ValueError, RuntimeError, TypeError) as exc:
        logging.getLogger(__name__).warning(
            "Gateway identity unavailable while creating farm: %s", exc)
        return 503, _json_bytes({
            "status": "error",
            "error": "Gateway identity is unavailable. Try again when the gateway is connected.",
        }), []
    try:
        farm, plot = plot_manager.createFarm(
            first('name'), first('latitude', 0), first('longitude', 0),
            first('size', 0), first('area_unit', 'm2'), first('timezone', 'UTC'),
            first('owner'), first('plot_name'), gateway_id=gateway_id)
        _trace_event(
            "farm.created", plot_manager.getCurrentPlot(),
            farm_id=farm.get("farm_id"), farm_name=farm.get("name"),
            latitude=farm.get("latitude"), longitude=farm.get("longitude"),
            size=farm.get("size"), area_unit=farm.get("area_unit"),
        )
        return 201, bytes(json.dumps({"farm": farm, "plot": plot}), "utf8"), []
    except (KeyError, TypeError, ValueError) as exc:
        return 400, bytes(json.dumps({"status": "error", "error": str(exc)}), "utf8"), []


usock.routerPOST("/api/farms", createFarm)


def updateFarm(url, body):
    values = parse_qs(body.decode('utf-8'))
    farm_id = values.pop('farm_id', [None])[0]
    fields = {key: entries[0] for key, entries in values.items()}
    try:
        farm = plot_manager.updateFarm(farm_id, **fields)
        return 200, bytes(json.dumps({"farm": farm}), "utf8"), []
    except KeyError as exc:
        return 404, bytes(json.dumps({"status": "error", "error": str(exc)}), "utf8"), []
    except (TypeError, ValueError) as exc:
        return 400, bytes(json.dumps({"status": "error", "error": str(exc)}), "utf8"), []


usock.routerPOST("/api/updateFarm", updateFarm)


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


def getDevices(url, body):
    """Proxy endpoint to fetch `devices` from the configured WaziGate API.
    This lets the UI request `../api/devices` without needing direct access
    to the external gateway URL or handling tokens in the client.
    """
    api_base = NetworkUtils.ApiUrl or ""
    if not api_base:
        return 400, bytes(json.dumps({"error": "ApiUrl not configured"}), "utf8"), []

    devices_url = api_base + "devices"
    headers = {}
    if getattr(NetworkUtils, 'Token', None):
        headers['Authorization'] = f"Bearer {NetworkUtils.Token}"

    try:
        resp = requests.get(devices_url, headers=headers, timeout=10)
        return resp.status_code, bytes(resp.text, "utf8"), []
    except Exception as exc:
        logging.getLogger(__name__).exception("WaziGate device request failed")
        return 500, bytes(json.dumps({"error": "Gateway request failed"}), "utf8"), []


usock.routerGET("/api/devices", getDevices)
usock.routerGET("/api/getAvailableDevices", getDevices)

# Resolve training requests against an explicit stable plot ID when supplied.
# The fallback preserves compatibility with older clients, while the dashboard
# always sends the ID so another browser cannot change its training target.


def _training_request_plot(url):
    query = parse_qs(urlparse(url).query)
    requested_id = query.get("plot_id", [None])[0]
    if requested_id:
        return _plot_by_stable_id(requested_id), requested_id
    plot = plot_manager.getCurrentPlot()
    return plot, getattr(plot, "stable_id", None)


# Frontend polls this to reload the selected plot when training is ready.


def isTrainingReady(url, body):
    current_plot, requested_id = _training_request_plot(url)
    if current_plot is None:
        payload = {"status": "error", "error": "Unknown plot_id",
                   "plot_id": requested_id}
        return 404, _json_bytes(payload), []
    finished = bool(getattr(current_plot, "training_finished", False))
    running = bool(getattr(current_plot, "currently_training", False))
    cache_status = getattr(current_plot, "pipeline_cache_status", "empty")
    reason = getattr(current_plot, "pipeline_cache_reason", None)

    if finished:
        status = "finished"
    elif running:
        status = "running"
    elif cache_status == "error":
        status = "error"
    else:
        status = "idle"

    response_data = {
        "isTrainingFinished": finished,
        "currentlyTraining": running,
        "status": status,
        "error": reason if status == "error" else None,
        "plot_id": getattr(current_plot, "stable_id", requested_id),
    }
    pipeline = getattr(current_plot, "pipeline_result", None)
    recommendation = getattr(pipeline, "recommendation", None)
    response_data["recommendation_available"] = (
        recommendation is not None
        and getattr(recommendation, "urgency", "error") != "error"
    )
    response_data["recommendation_error"] = getattr(
        recommendation, "error_message", None)
    return 200, _json_bytes(response_data), []


usock.routerGET("/api/isTrainingReady", isTrainingReady)

# Starts a thread that runs training, this thread will also start prediction afterwards


def _training_prerequisites(plot):
    missing = []
    if not getattr(plot, 'device_and_sensor_ids_moisture', []):
        missing.append('moisture sensors')
    if not getattr(plot, 'device_and_sensor_ids_temp', []):
        missing.append('temperature sensors')
    if not getattr(plot, 'crop_type', ''):
        missing.append('crop')
    try:
        threshold = float(getattr(plot, 'threshold_static', 0))
    except (TypeError, ValueError):
        threshold = 0
    if threshold <= 0:
        missing.append('field-calibrated static threshold')
    return missing


def startTraining(url, body):
    plot, requested_id = _training_request_plot(url)
    if plot is None:
        payload = {"status": "error", "message": "Unknown plot_id",
                   "plot_id": requested_id}
        return 404, _json_bytes(payload), []
    missing = _training_prerequisites(plot)
    if missing:
        payload = {"status": "error", "message": "Training prerequisites missing",
                   "missing": missing}
        return 409, bytes(json.dumps(payload), "utf8"), []
    training_thread.start(plot)
    _trace_event(
        "training.started", plot,
        data_source="csv" if plot.load_data_from_csv else "gateway",
        moisture_sensors=plot.device_and_sensor_ids_moisture,
        temperature_sensors=plot.device_and_sensor_ids_temp,
        threshold_mode=getattr(plot, "threshold_mode", "static"),
    )
    return 200, _json_bytes({"status": "started",
                             "plot_id": getattr(plot, "stable_id", requested_id)}), []


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

    # Load all plots once on startup.
    plot_manager.loadPlots()

    # Activate optional CSV data loading before configuration discovery.
    if os.getenv("LOAD_DATA_FROM_CSV") == "True":
        for plot in plot_manager.Plots.values():
            plot.load_data_from_csv = True

    # Bound logs.log by SIZE (an active log's mtime is always recent, so the age-based
    # cleaner below never fires on it) - set up before anything logs
    setup_logging()

    # Get saved config from all plots and save it in objects
    getConfigsFromAllFiles()
    # Compatibility JSON loads installation/UI state first. Optional valid
    # YAML entries apply only to plots that are not explicitly UI-owned.
    try:
        # YAML entries are optional manual/model configurations. Installation
        # settings persisted by the UI are operational and must survive restart.
        farm_configs = load_all_farms(strict=True)
        apply_farm_configs_to_plots(
            plot_manager.getPlots(), farm_configs, preserve_ui_config=True)
    except (OSError, RuntimeError, ValueError) as exc:
        # Configuration errors are operator-facing and must stop before workers.
        raise SystemExit(f"Startup validation failed: {exc}") from None

    # Command acceptance and delivery confirmation are separate states. Any
    # accepted command awaiting its delayed meter check must survive restart.
    actuation.resume_pending_irrigation_verifications(
        plot_manager.getPlots())

    # Import the ML runtime only after fail-fast configuration checks succeed.
    import create_model as create_model_module
    import training_thread as training_thread_module
    create_model = create_model_module
    training_thread = training_thread_module

    # Apply optional development/testing configuration.
    if os.getenv("LOAD_DATA_FROM_CSV") == "True":
        for plot in plot_manager.Plots.values():
            plot.load_data_from_csv = True
    if os.getenv("SKIP_DATA_PREPROCESSING") == "True":
        # Workers read shared flags from create_model.state, not package aliases.
        create_model.state.SkipDataPreprocessing = True
    if os.getenv("SKIP_TRAINING") == "True":
        create_model.state.SkipTraining = True
    if os.getenv("PERFORM_TRAINING") == "False":
        create_model.state.Perform_training = False

    sync_sensor_registry_from_plots()

    # Start thread that deletes old models and data regularly to save memory.
    # file paths: prune old FILES, keep the folder tree (models/<plot>/... stays intact)
    file_cleanup_paths = ["models", "hyperband_dir", "data/subprocess_temp",
                          "catboost_info", "test-reports"]
    # dir globs: delete the WHOLE matching folder once old (per-run scratch)
    dir_cleanup_globs = ["tmp/tuning_*", "tmp/nn_*"]
    # file globs: prune only matching cache files.
    file_cleanup_globs = ["data/cache/saved_variables_plot_*.pkl"]
    model_cleaner = schedule_model_cleanup(
        file_cleanup_paths, dir_cleanup_globs,
        file_cleanup_globs, interval_days=7)  # Check every week

    # Clean logs
    log_cleaners = schedule_log_cleanup()

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
            missing = _training_prerequisites(p)
            if p.configPath and not p.training_finished and not missing:
                training_thread.start(p)
            elif missing:
                logging.getLogger(__name__).warning(
                    "Auto-training skipped for plot %s; prerequisites missing: %s",
                    getattr(p, "stable_id", getattr(p, "id", "?")), missing)

    # Keep main thread alive
    from schedule_dispatcher import ScheduleDispatcher
    schedule_worker = ScheduleDispatcher()
    schedule_worker.start()
    try:
        while True:
            time.sleep(3600)  # Check every hour
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        # Stop workers before the HTTP server so no background task continues
        # mutating plot state during container shutdown.
        schedule_worker.stop()
        schedule_worker.join()
        for cleaner in [model_cleaner, *log_cleaners]:
            cleaner.stop()
        stop_background_workers(
            plot_manager.getPlots(), [model_cleaner, *log_cleaners])
        usock.stop()
        server_thread.join(timeout=30)
