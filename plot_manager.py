"""Runtime plot objects backed by persistent, stable farm/plot identities."""

from plot_location import plot_reference_location
from pathlib import Path
from copy import deepcopy
import threading

from farm_registry import FarmRegistry
from plot import Plot
from removal_recovery import recover_removals
from state_store import RegistryConflictError

Plots = {}
CurrentPlotId = None
CurrentPlotTab = None
Config_folder_path = "config/"
ConfigPath = None
Threads = []
ThreadId = 0
_lock = threading.RLock()
_registry = FarmRegistry()


def _attach_farm_context(plot, farm):
    plot.farm_gps_info = {
        "latitude": farm.get("latitude"),
        "longitude": farm.get("longitude"),
    }
    plot.farm_name = str(farm.get("name") or "")
    plot.farm_timezone = str(farm.get("timezone") or "UTC")
    return plot


class SpatialSetupConflict(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _new_runtime_plot(tab, record, farm):
    """Construct before commit so constructor failures cannot leave saved ghosts."""
    path = str(Path(Config_folder_path) / record['config_file'])
    plot = Plot(tab, path, stable_id=record['plot_id'])
    plot.configuration_source = 'spatial_setup'
    plot.timezone = farm['timezone']
    _apply_spatial_record(plot, tab, record, farm)
    return plot


def _apply_spatial_record(plot, tab, record, farm):
    _attach_farm_context(plot, farm)
    plot.tab_number = tab
    plot.user_given_name = record['name']
    plot.plot_area_m2 = record['area_m2']
    plot.area_unit = record['area_unit']
    plot.crop_type = record['crop_type']
    plot.geometry = deepcopy(record['geometry'])
    plot.geometry_version = record['geometry_version']


def _prepare_runtime(data):
    existing = {plot.stable_id: plot for plot in Plots.values()}
    return {tab: existing.get(record['plot_id']) or _new_runtime_plot(tab, record, data['farm'])
            for tab, record in enumerate(data['plots'], start=1)}


def _publish_runtime(prepared, data):
    global Plots, CurrentPlotId, CurrentPlotTab, ConfigPath
    for tab, record in enumerate(data['plots'], start=1):
        _apply_spatial_record(prepared[tab], tab, record, data['farm'])
    Plots = prepared
    CurrentPlotTab = next(tab for tab, plot in prepared.items() if plot.stable_id == data['current_plot_id'])
    current = prepared[CurrentPlotTab]
    CurrentPlotId, ConfigPath = current.id, current.configPath


def _resolve_key(identifier):
    if isinstance(identifier, str):
        for key, plot in Plots.items():
            if plot.stable_id == identifier:
                return key
        try:
            identifier = int(identifier)
        except ValueError as exc:
            raise KeyError(f"Unknown plot: {identifier}") from exc
    identifier = int(identifier)
    if identifier in Plots:
        return identifier
    for key, plot in Plots.items():
        if plot.id == identifier:
            return key
    raise KeyError(f"Unknown plot: {identifier}")


def readFiles():
    return [record["config_file"] for record in _registry.snapshot()["plots"]]


def setPlot(identifier):
    global CurrentPlotId, CurrentPlotTab, ConfigPath
    with _lock:
        key = _resolve_key(identifier)
        plot = Plots[key]
        _registry.set_current_plot(plot.stable_id)
        # Publish the runtime target only after its persisted selection commits.
        CurrentPlotId, CurrentPlotTab, ConfigPath = plot.id, key, plot.configPath
        return ConfigPath


def loadPlots(registry_path=None, config_folder=None):
    """Load spatial records; a fresh installation has no scaffold plot."""
    global Plots, CurrentPlotId, CurrentPlotTab, ConfigPath
    global Config_folder_path, _registry
    with _lock:
        if config_folder is not None:
            Config_folder_path = str(config_folder).rstrip("/") + "/"
        registry_file = registry_path or Path(Config_folder_path) / "farm_registry.json"
        _registry = FarmRegistry(registry_file, Config_folder_path)
        data = _registry.load_or_migrate()
        recover_removals(Config_folder_path, data)
        Plots = {}
        records = sorted(data["plots"], key=lambda item: (item.get("position", 0), item["legacy_id"]))
        for tab, record in enumerate(records, start=1):
            Plots[tab] = _new_runtime_plot(tab, record, data['farm'])
        if not Plots:
            CurrentPlotId, CurrentPlotTab, ConfigPath = None, None, None
            return 0
        CurrentPlotTab = next(key for key, plot in Plots.items() if plot.stable_id == data["current_plot_id"])
        current = Plots[CurrentPlotTab]
        CurrentPlotId, ConfigPath = current.id, current.configPath
        return len(Plots)


def _create_spatial_record(*, registry_revision, farm_fields=None, plot_fields=None):
    """Prepare runtime objects inside the DB transaction, publish after commit."""
    with _lock:
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError('A nonnegative registry_revision is required')
        _registry.snapshot()  # initialize this installation's store
        store = _registry._state_store
        try:
            with store.transaction():
                current = _registry.reload()
                if registry_revision != current['registry_revision']:
                    raise RegistryConflictError('Farm layout changed; reload before saving')
                if farm_fields is not None:
                    if current['farm'] is not None:
                        raise SpatialSetupConflict('farm_exists', 'This installation already has a farm.')
                    _registry.create_farm(**farm_fields)
                else:
                    if current['farm'] is None:
                        raise SpatialSetupConflict('setup_required', 'Create the farm and first plot together before adding plots.')
                    _registry.add_plot(**plot_fields)
                saved = _registry.snapshot()
                prepared = _prepare_runtime(saved)
                # Retain the exact acknowledged resource/snapshot, not a later
                # selection or another process's subsequent registry write.
                plot = next(p for p in saved['plots'] if p['plot_id'] == saved['current_plot_id'])
                response = dict(status='created', farm=saved['farm'], plot=plot,
                    current_plot_id=saved['current_plot_id'], registry_revision=saved['registry_revision'])
        except BaseException:
            _registry.reload()
            raise
        _publish_runtime(prepared, saved)
        return response


def createFarm(*, name, geometry, first_plot, registry_revision, area_unit='m2',
               timezone_name='UTC', owner='', latitude=None, longitude=None):
    return _create_spatial_record(registry_revision=registry_revision,
        farm_fields=dict(name=name, geometry=geometry, first_plot=first_plot,
            area_unit=area_unit, timezone_name=timezone_name, owner=owner,
            latitude=latitude, longitude=longitude))


def addPlot(*, name, crop_type, geometry, registry_revision, area_unit='m2'):
    return _create_spatial_record(registry_revision=registry_revision,
        plot_fields=dict(name=name, crop_type=crop_type, geometry=geometry, area_unit=area_unit))


def updateFarm(**fields):
    with _lock:
        farm = _registry.update_farm(**fields)
        for plot in Plots.values():
            _attach_farm_context(plot, farm)
        return farm


def getPlotByStableId(plot_id):
    """Strict resource lookup: never interpret a stable ID as a tab/index."""
    with _lock:
        return next((plot for plot in Plots.values()
                     if plot.stable_id == plot_id), None)


def updatePlotMetadata(plot_id, name=None, area=None, area_unit=None):
    with _lock:
        plot = getPlotByStableId(plot_id)
        if plot is None:
            raise KeyError(f"Unknown plot_id: {plot_id}")
        fields = {key: value for key, value in {"name": name, "area": area,
                  "area_unit": area_unit}.items() if value is not None}
        record = _registry.update_plot(plot.stable_id, **fields)
        plot.user_given_name, plot.plot_area_m2 = record["name"], record["area"]
        plot.area_unit = record["area_unit"]
        return record


def updateCurrentPlotMetadata(name=None, area=None, area_unit=None):
    with _lock:
        return updatePlotMetadata(getCurrentPlot().stable_id, name, area, area_unit)


def removePlot(identifier):
    """Remove plot identity and settings together in SQLite."""
    global Plots, CurrentPlotId, CurrentPlotTab, ConfigPath
    with _lock:
        key = _resolve_key(identifier)
        plot = Plots[key]
        config_path = Path(plot.configPath)
        removed = _registry.remove_plot(plot.stable_id)
        del Plots[key]
        Plots = {index: item for index, item in enumerate(Plots.values(), start=1)}
        for tab, item in Plots.items():
            item.tab_number = tab
        CurrentPlotTab = _resolve_key(_registry.snapshot()["current_plot_id"])
        current = Plots[CurrentPlotTab]
        CurrentPlotId, ConfigPath = current.id, current.configPath
        return removed["plot_id"], str(config_path)


def registrySnapshot():
    with _lock:
        data = _registry.snapshot()
        runtime = {plot.stable_id: plot for plot in Plots.values()}
        for record in data["plots"]:
            plot = runtime.get(record["plot_id"])
            if plot:
                latitude, longitude = plot_reference_location(plot)
                record['reference_location'] = (
                    {'latitude': latitude, 'longitude': longitude}
                    if latitude is not None else None)
                record.update({"tab_number": plot.tab_number, "name": plot.user_given_name,
                               "area": float(getattr(plot, "plot_area_m2", 0.0) or 0.0),
                               "area_unit": getattr(plot, "area_unit", "m2")})
        return data

def getPlots():
    return Plots


def getCurrentConfig():
    return getCurrentPlot().configPath


def getCurrentPlot():
    return Plots[CurrentPlotTab]


def getCurrentPlotNumberWithId(currentPlot):
    return _resolve_key(getattr(currentPlot, "stable_id", currentPlot.id))


def getCurrentPlotWithId(passed_id):
    try:
        return Plots[_resolve_key(passed_id)]
    except KeyError:
        return False


def removePlotWithId(passed_id):
    try:
        removePlot(passed_id)
        return True
    except (KeyError, ValueError):
        return False


def setCurrentConfig(path):
    global ConfigPath
    getCurrentPlot().configPath = path
    ConfigPath = path


def replaceBoundary(**request):
    """Publish new geometry/area on existing workers only after a confirmed commit."""
    from spatial_editing import replace_boundary
    with _lock:
        _registry.snapshot()
        saved, prepared = replace_boundary(_registry._state_store, request, _prepare_runtime)
        # Publish the exact committed snapshot. A later process's commit must not
        # be mixed with runtime objects prepared for this acknowledged revision.
        with _registry._lock:
            _registry.data = deepcopy(saved)
            _registry._committed_data = deepcopy(saved)
        _publish_runtime(prepared, saved)
        for record in saved['plots']:
            runtime = next(p for p in prepared.values() if p.stable_id == record['plot_id'])
            if isinstance(getattr(runtime, 'config', None), dict):
                runtime.config = dict(runtime.config, Plot_area_m2=record['area_m2'])
        record = saved['farm'] if request['entity'] == 'farm' else next(
            p for p in saved['plots'] if p['plot_id'] == request['plot_id'])
        return dict(status='saved', entity=request['entity'], plot_id=request['plot_id'],
                    registry_revision=saved['registry_revision'], boundary=record)
