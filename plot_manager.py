"""Runtime plot objects backed by persistent, stable farm/plot identities."""

from pathlib import Path
import threading

from farm_registry import FarmRegistry
from plot import Plot
from removal_recovery import record_removal, recover_removals, clear_journal

Plots = {}
CurrentPlotId = 1
CurrentPlotTab = 1
Config_folder_path = "config/"
ConfigPath = "config/current_config_plot1.json"
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


def _area_in_square_metres(area, area_unit):
    """Convert creation input; persisted plot areas are already square metres."""
    factors = {"m2": 1.0, "ha": 10000.0, "acre": 4046.8564224}
    if area_unit not in factors:
        raise ValueError(f"Unsupported area unit: {area_unit}")
    return float(area) * factors[area_unit]


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
    """Load the registry, migrating legacy config files on first use."""
    global Plots, CurrentPlotId, CurrentPlotTab, ConfigPath
    global Config_folder_path, _registry
    with _lock:
        if config_folder is not None:
            Config_folder_path = str(config_folder).rstrip("/") + "/"
        registry_file = registry_path or Path(Config_folder_path) / "farm_registry.json"
        _registry = FarmRegistry(registry_file, Config_folder_path)
        data = _registry.load_or_migrate()
        farms_by_id = {farm["farm_id"]: farm for farm in data["farms"]}
        recover_removals(Config_folder_path, data)
        Plots = {}
        records = sorted(data["plots"], key=lambda item: (item.get("position", 0), item["legacy_id"]))
        for tab, record in enumerate(records, start=1):
            path = str(Path(Config_folder_path) / record["config_file"])
            plot = Plot(tab, path, stable_id=record["plot_id"], farm_id=record["farm_id"])
            _attach_farm_context(plot, farms_by_id[record["farm_id"]])
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
        record = _registry.add_plot(farm_id, name, _area_in_square_metres(area, area_unit), area_unit)
        tab = len(Plots) + 1
        path = str(Path(Config_folder_path) / record["config_file"])
        plot = Plot(tab, path, stable_id=record["plot_id"], farm_id=farm_id)
        _attach_farm_context(plot, _registry.get_farm(farm_id))
        plot.user_given_name, plot.plot_area_m2 = record["name"], record["area"]
        plot.area_unit = record["area_unit"]
        Plots[tab] = plot
        setPlot(record["plot_id"])
        return record["plot_id"], path


def createFarm(name, latitude=0, longitude=0, size=0, area_unit="m2", timezone_name="UTC", owner="", plot_name="", gateway_id=None):
    global CurrentPlotId, CurrentPlotTab, ConfigPath
    with _lock:
        # Registry identity, ownership, area and selection commit together.
        # Runtime objects are not changed until the registry publish succeeds.
        with _registry.transaction():
            snapshot = _registry.snapshot()
            desired_plot_name = str(plot_name or "Plot 1").strip()
            if _is_pristine_install_scaffold(snapshot):
                farm_record, plot_record = snapshot["farms"][0], snapshot["plots"][0]
                farm = _registry.update_farm(
                    farm_record["farm_id"], name=name, latitude=latitude,
                    longitude=longitude, size=size, area_unit=area_unit,
                    timezone=timezone_name, owner=owner, gateway_id=gateway_id)
                plot = _registry.update_plot(
                    plot_record["plot_id"], name=desired_plot_name,
                    area=_area_in_square_metres(size, area_unit), area_unit=area_unit)
                tab = _resolve_key(plot["plot_id"])
                runtime_plot = Plots[tab]
            else:
                farm = _registry.create_farm(name, latitude, longitude, size, area_unit, timezone_name, owner, gateway_id=gateway_id)
                plot = _registry.add_plot(farm["farm_id"], desired_plot_name,
                                          _area_in_square_metres(size, area_unit), area_unit)
                tab = len(Plots) + 1
                path = str(Path(Config_folder_path) / plot["config_file"])
                runtime_plot = Plot(tab, path, stable_id=plot["plot_id"], farm_id=farm["farm_id"])
            _registry.set_current_plot(plot["plot_id"])
            farm = _registry.get_farm(farm["farm_id"])
        runtime_plot.user_given_name = plot["name"]
        _attach_farm_context(runtime_plot, farm)
        runtime_plot.plot_area_m2 = plot["area"]
        runtime_plot.area_unit = plot["area_unit"]
        Plots[tab] = runtime_plot
        CurrentPlotId, CurrentPlotTab, ConfigPath = runtime_plot.id, tab, runtime_plot.configPath
        return farm, plot


def _is_pristine_install_scaffold(snapshot):
    farms = snapshot.get("farms", [])
    plots = snapshot.get("plots", [])
    if len(farms) != 1 or len(plots) != 1:
        return False
    farm, plot = farms[0], plots[0]
    config_path = Path(Config_folder_path) / plot.get("config_file", "")
    return (
        farm.get("name") == "My Farm"
        and not str(farm.get("owner") or "").strip()
        and float(farm.get("latitude", 0) or 0) == 0
        and float(farm.get("longitude", 0) or 0) == 0
        and float(farm.get("size", 0) or 0) == 0
        and plot.get("name") == "Plot 1"
        and float(plot.get("area", 0) or 0) == 0
        and not config_path.exists()
    )


def updateFarm(farm_id, **fields):
    farm = _registry.update_farm(farm_id, **fields)
    for plot in Plots.values():
        if plot.farm_id == farm_id:
            _attach_farm_context(plot, farm)
    return farm


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
        config_path = Path(plot.configPath)
        archive = None
        journal = None
        archived = False
        try:
            with _registry.transaction():
                removed = _registry.remove_plot(plot.stable_id)
                if config_path.exists():
                    archive = config_path.with_suffix(config_path.suffix + ".removed")
                    counter = 1
                    while archive.exists():
                        archive = config_path.with_suffix(config_path.suffix + f".removed.{counter}")
                        counter += 1
                    journal = record_removal(config_path, archive, plot.stable_id)
                    config_path.rename(archive)
                    archived = True
        except BaseException:
            # Failed registry publication rolls memory back. Restore the file
            # moved by this attempt as well; never overwrite a replacement file.
            if archived:
                if config_path.exists():
                    raise RuntimeError(f'Removal rollback blocked; archived config retained at {archive}')
                archive.rename(config_path)
            clear_journal(journal)
            raise
        del Plots[key]
        Plots = {index: item for index, item in enumerate(Plots.values(), start=1)}
        for tab, item in Plots.items():
            item.tab_number = tab
        CurrentPlotTab = _resolve_key(_registry.snapshot()["current_plot_id"])
        current = Plots[CurrentPlotTab]
        CurrentPlotId, ConfigPath = current.id, current.configPath
        clear_journal(journal)
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
