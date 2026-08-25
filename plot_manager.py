"""Runtime plot objects backed by persistent, stable farm/plot identities."""

from pathlib import Path
import threading

from farm_registry import FarmRegistry
from plot import Plot

Plots = {}
CurrentPlotId = 1
CurrentPlotTab = 1
Config_folder_path = "config/"
ConfigPath = "config/current_config_plot1.json"
Threads = []
ThreadId = 0
_lock = threading.RLock()
_registry = FarmRegistry()


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
        CurrentPlotId, CurrentPlotTab, ConfigPath = plot.id, key, plot.configPath
        _registry.set_current_plot(plot.stable_id)
        return ConfigPath


def loadPlots(registry_path=None, config_folder=None):
    """Load the registry, migrating legacy config files on first use."""
    global Plots, CurrentPlotId, CurrentPlotTab, ConfigPath
    global Config_folder_path, _registry
    with _lock:
        if config_folder is not None:
            Config_folder_path = str(config_folder).rstrip("/") + "/"
        registry_file = registry_path or Path(Config_folder_path) / "farm_registry.json"
        _registry = FarmRegistry(registry_file, Config_folder_path)
        data = _registry.load_or_migrate()
        Plots = {}
        records = sorted(data["plots"], key=lambda item: (item.get("position", 0), item["legacy_id"]))
        for tab, record in enumerate(records, start=1):
            path = str(Path(Config_folder_path) / record["config_file"])
            plot = Plot(tab, path, stable_id=record["plot_id"], farm_id=record["farm_id"])
            plot.user_given_name = record["name"]
            plot.plot_area_m2 = float(record.get("area", 0.0) or 0.0)
            plot.area_unit = record.get("area_unit", "m2")
            Plots[tab] = plot
        CurrentPlotTab = next(key for key, plot in Plots.items() if plot.stable_id == data["current_plot_id"])
        current = Plots[CurrentPlotTab]
        CurrentPlotId, ConfigPath = current.id, current.configPath
        return len(Plots)


def addPlot(tabNumber=None, farm_id=None, name="", area=0.0, area_unit="m2"):
    global Plots
    with _lock:
        farm_id = farm_id or _registry.snapshot()["current_farm_id"]
        record = _registry.add_plot(farm_id, name, area, area_unit)
        tab = len(Plots) + 1
        path = str(Path(Config_folder_path) / record["config_file"])
        plot = Plot(tab, path, stable_id=record["plot_id"], farm_id=farm_id)
        plot.user_given_name, plot.plot_area_m2 = record["name"], record["area"]
        plot.area_unit = record["area_unit"]
        Plots[tab] = plot
        setPlot(record["plot_id"])
        return record["plot_id"], path


def createFarm(name, latitude=0, longitude=0, size=0, area_unit="m2", timezone_name="UTC", owner="", plot_name=""):
    with _lock:
        farm = _registry.create_farm(name, latitude, longitude, size, area_unit, timezone_name, owner)
        plot_id, _ = addPlot(farm_id=farm["farm_id"], name=plot_name or f"{name} Plot 1", area=size, area_unit=area_unit)
        return _registry.get_farm(farm["farm_id"]), _registry.get_plot(plot_id)


def updateFarm(farm_id, **fields):
    return _registry.update_farm(farm_id, **fields)


def updateCurrentPlotMetadata(name=None, area=None, area_unit=None):
    plot = getCurrentPlot()
    fields = {key: value for key, value in {"name": name, "area": area, "area_unit": area_unit}.items() if value is not None}
    record = _registry.update_plot(plot.stable_id, **fields)
    plot.user_given_name, plot.plot_area_m2 = record["name"], record["area"]
    plot.area_unit = record["area_unit"]
    return record


def removePlot(identifier):
    """Remove the exact stable target and archive, rather than delete, its config."""
    global Plots, CurrentPlotId, CurrentPlotTab, ConfigPath
    with _lock:
        key = _resolve_key(identifier)
        plot = Plots[key]
        removed = _registry.remove_plot(plot.stable_id)
        config_path = Path(plot.configPath)
        if config_path.exists():
            archive = config_path.with_suffix(config_path.suffix + ".removed")
            counter = 1
            while archive.exists():
                archive = config_path.with_suffix(config_path.suffix + f".removed.{counter}")
                counter += 1
            config_path.rename(archive)
        del Plots[key]
        Plots = {index: item for index, item in enumerate(Plots.values(), start=1)}
        for tab, item in Plots.items():
            item.tab_number = tab
        CurrentPlotTab = _resolve_key(_registry.snapshot()["current_plot_id"])
        current = Plots[CurrentPlotTab]
        CurrentPlotId, ConfigPath = current.id, current.configPath
        return removed["plot_id"], str(config_path)


def registrySnapshot():
    data = _registry.snapshot()
    runtime = {plot.stable_id: plot for plot in Plots.values()}
    for record in data["plots"]:
        plot = runtime.get(record["plot_id"])
        if plot:
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
