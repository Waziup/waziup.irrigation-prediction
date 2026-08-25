"""Persistent farm/plot identity and migration for installation-managed data."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import uuid


REGISTRY_VERSION = 1
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_REGISTRY_PATH = DEFAULT_CONFIG_DIR / "farm_registry.json"
AREA_UNITS = {"m2", "ha", "acre"}
_CONFIG_RE = re.compile(r"current_config_plot(\d+)\.json$")


def _stable_id(kind: str, seed: str) -> str:
    value = uuid.uuid5(uuid.NAMESPACE_URL, f"wazifarm:{kind}:{seed}")
    return f"{kind}-{value.hex[:12]}"


def _new_id(kind: str) -> str:
    return f"{kind}-{uuid.uuid4().hex[:12]}"


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


class FarmRegistry:
    """Atomic JSON registry retaining stable IDs across restarts."""

    def __init__(self, path=DEFAULT_REGISTRY_PATH, config_dir=DEFAULT_CONFIG_DIR):
        self.path = Path(path)
        self.config_dir = Path(config_dir)
        self._lock = threading.RLock()
        self.data = None

    def load_or_migrate(self) -> dict:
        with self._lock:
            if self.path.exists():
                data = _read_json(self.path)
                self._validate(data)
                self.data = data
                return deepcopy(data)
            self.data = self._migrate_legacy()
            self._save_unlocked()
            return deepcopy(self.data)

    def reload(self) -> dict:
        with self._lock:
            self.data = None
            return self.load_or_migrate()

    def snapshot(self) -> dict:
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            return deepcopy(self.data)

    def _migrate_legacy(self) -> dict:
        config_files = sorted(
            self.config_dir.glob("current_config_plot*.json"),
            key=lambda path: int(_CONFIG_RE.search(path.name).group(1)),
        )
        farm_id = _stable_id("farm", "legacy-default")
        plots = []
        first_config = _read_json(config_files[0]) if config_files else {}
        first_gps = first_config.get("Gps_info") or {}

        for position, path in enumerate(config_files, start=1):
            match = _CONFIG_RE.search(path.name)
            legacy_id = int(match.group(1))
            config = _read_json(path)
            plot_id = str(config.get("Plot_id") or _stable_id("plot", path.name))
            assigned_farm = str(config.get("Farm_id") or farm_id)
            plots.append({
                "plot_id": plot_id,
                "farm_id": assigned_farm,
                "legacy_id": legacy_id,
                "config_file": path.name,
                "name": str(config.get("Name") or f"Plot {legacy_id}"),
                "area": _number(config.get("Plot_area_m2"), 0.0),
                "area_unit": str(config.get("Plot_area_unit") or "m2"),
                "position": position,
            })

        if not plots:
            plot_id = _stable_id("plot", "initial")
            plots.append({
                "plot_id": plot_id,
                "farm_id": farm_id,
                "legacy_id": 1,
                "config_file": "current_config_plot1.json",
                "name": "Plot 1",
                "area": 0.0,
                "area_unit": "m2",
                "position": 1,
            })

        farm_ids = []
        for plot in plots:
            if plot["farm_id"] not in farm_ids:
                farm_ids.append(plot["farm_id"])
        farms = []
        for index, assigned_farm in enumerate(farm_ids):
            farm_plots = [p for p in plots if p["farm_id"] == assigned_farm]
            farms.append({
                "farm_id": assigned_farm,
                "name": str(first_config.get("Farm_name") or (
                    "My Farm" if index == 0 else f"Farm {index + 1}")),
                "owner": str(first_config.get("Owner") or ""),
                "latitude": _number(first_gps.get(
                    "latitude", first_gps.get("lattitude")), 0.0),
                "longitude": _number(first_gps.get("longitude"), 0.0),
                "size": sum(_number(p["area"], 0.0) for p in farm_plots),
                "area_unit": "m2",
                "timezone": str(first_config.get("Timezone") or "UTC"),
                "plot_ids": [p["plot_id"] for p in farm_plots],
            })

        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": REGISTRY_VERSION,
            "current_farm_id": farms[0]["farm_id"],
            "current_plot_id": plots[0]["plot_id"],
            "farms": farms,
            "plots": plots,
            "created_at": now,
            "updated_at": now,
        }

    def _validate(self, data: dict) -> None:
        if not isinstance(data, dict) or data.get("version") != REGISTRY_VERSION:
            raise ValueError("Unsupported or invalid farm registry version")
        farms = data.get("farms")
        plots = data.get("plots")
        if not isinstance(farms, list) or not isinstance(plots, list):
            raise ValueError("Farm registry must contain farm and plot lists")
        farm_ids = [farm.get("farm_id") for farm in farms]
        plot_ids = [plot.get("plot_id") for plot in plots]
        if not farm_ids or not plot_ids:
            raise ValueError("Farm registry requires at least one farm and plot")
        if None in farm_ids or len(set(farm_ids)) != len(farm_ids):
            raise ValueError("Farm IDs must be present and unique")
        if None in plot_ids or len(set(plot_ids)) != len(plot_ids):
            raise ValueError("Plot IDs must be present and unique")
        known_farms = set(farm_ids)
        if any(plot.get("farm_id") not in known_farms for plot in plots):
            raise ValueError("Every plot must reference an existing farm")
        linked_plot_ids = []
        for farm in farms:
            children = farm.get("plot_ids")
            if not isinstance(children, list):
                raise ValueError("Every farm must contain a plot ID list")
            linked_plot_ids.extend(children)
            expected = {p["plot_id"] for p in plots if p["farm_id"] == farm["farm_id"]}
            if set(children) != expected:
                raise ValueError("Farm plot links do not match plot ownership")
        if len(linked_plot_ids) != len(set(linked_plot_ids)):
            raise ValueError("A plot cannot belong to multiple farms")
        if data.get("current_plot_id") not in set(plot_ids):
            raise ValueError("Current plot must reference an existing plot")
        current = next(p for p in plots if p["plot_id"] == data["current_plot_id"])
        if data.get("current_farm_id") != current["farm_id"]:
            raise ValueError("Current farm must own the current plot")

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.path)

    def save(self) -> None:
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            self._validate(self.data)
            self._save_unlocked()

    def get_farm(self, farm_id: str) -> dict:
        data = self.snapshot()
        for farm in data["farms"]:
            if farm["farm_id"] == farm_id:
                return farm
        raise KeyError(f"Unknown farm_id: {farm_id}")

    def get_plot(self, plot_id: str) -> dict:
        data = self.snapshot()
        for plot in data["plots"]:
            if plot["plot_id"] == plot_id:
                return plot
        raise KeyError(f"Unknown plot_id: {plot_id}")

    def set_current_plot(self, plot_id: str) -> None:
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            plot = next((p for p in self.data["plots"] if p["plot_id"] == plot_id), None)
            if plot is None:
                raise KeyError(f"Unknown plot_id: {plot_id}")
            self.data["current_plot_id"] = plot_id
            self.data["current_farm_id"] = plot["farm_id"]
            self._save_unlocked()

    def create_farm(self, name, latitude, longitude, size, area_unit, timezone_name, owner="") -> dict:
        name = str(name or "").strip()
        if not name:
            raise ValueError("Farm name is required")
        if area_unit not in AREA_UNITS:
            raise ValueError(f"Unsupported area unit: {area_unit}")
        latitude = float(latitude)
        longitude = float(longitude)
        size = float(size)
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Farm coordinates are out of range")
        if size < 0:
            raise ValueError("Farm size cannot be negative")

        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            farm_id = _new_id("farm")
            farm = {
                "farm_id": farm_id,
                "name": name,
                "owner": str(owner or "").strip(),
                "latitude": latitude,
                "longitude": longitude,
                "size": size,
                "area_unit": area_unit,
                "timezone": str(timezone_name or "UTC"),
                "plot_ids": [],
            }
            self.data["farms"].append(farm)
            self._save_unlocked()
            return deepcopy(farm)

    def update_farm(self, farm_id: str, **fields) -> dict:
        allowed = {"name", "owner", "latitude", "longitude", "size", "area_unit", "timezone"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported farm fields: {sorted(unknown)}")
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            farm = next((f for f in self.data["farms"] if f["farm_id"] == farm_id), None)
            if farm is None:
                raise KeyError(f"Unknown farm_id: {farm_id}")
            if "name" in fields and not str(fields["name"]).strip():
                raise ValueError("Farm name is required")
            if "area_unit" in fields and fields["area_unit"] not in AREA_UNITS:
                raise ValueError(f"Unsupported area unit: {fields['area_unit']}")
            for numeric in ("latitude", "longitude", "size"):
                if numeric in fields:
                    fields[numeric] = float(fields[numeric])
            if "latitude" in fields and not -90 <= fields["latitude"] <= 90:
                raise ValueError("Farm latitude is out of range")
            if "longitude" in fields and not -180 <= fields["longitude"] <= 180:
                raise ValueError("Farm longitude is out of range")
            if "size" in fields and fields["size"] < 0:
                raise ValueError("Farm size cannot be negative")
            farm.update(fields)
            self._save_unlocked()
            return deepcopy(farm)

    def add_plot(self, farm_id: str, name="", area=0.0, area_unit="m2") -> dict:
        if area_unit not in AREA_UNITS:
            raise ValueError(f"Unsupported area unit: {area_unit}")
        area = float(area)
        if area < 0:
            raise ValueError("Plot area cannot be negative")
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            farm = next((f for f in self.data["farms"] if f["farm_id"] == farm_id), None)
            if farm is None:
                raise KeyError(f"Unknown farm_id: {farm_id}")
            next_legacy_id = max(
                [int(plot.get("legacy_id", 0)) for plot in self.data["plots"]] or [0]
            ) + 1
            plot_id = _new_id("plot")
            plot = {
                "plot_id": plot_id,
                "farm_id": farm_id,
                "legacy_id": next_legacy_id,
                "config_file": f"current_config_plot{next_legacy_id}.json",
                "name": str(name or f"Plot {next_legacy_id}").strip(),
                "area": area,
                "area_unit": area_unit,
                "position": len(farm["plot_ids"]) + 1,
            }
            self.data["plots"].append(plot)
            farm["plot_ids"].append(plot_id)
            self.data["current_farm_id"] = farm_id
            self.data["current_plot_id"] = plot_id
            self._save_unlocked()
            return deepcopy(plot)

    def update_plot(self, plot_id: str, **fields) -> dict:
        allowed = {"name", "area", "area_unit", "farm_id"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported plot fields: {sorted(unknown)}")
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            plot = next((p for p in self.data["plots"] if p["plot_id"] == plot_id), None)
            if plot is None:
                raise KeyError(f"Unknown plot_id: {plot_id}")
            if "area_unit" in fields and fields["area_unit"] not in AREA_UNITS:
                raise ValueError(f"Unsupported area unit: {fields['area_unit']}")
            if "area" in fields:
                fields["area"] = float(fields["area"])
                if fields["area"] < 0:
                    raise ValueError("Plot area cannot be negative")
            if "farm_id" in fields:
                new_farm = next((f for f in self.data["farms"] if f["farm_id"] == fields["farm_id"]), None)
                if new_farm is None:
                    raise KeyError(f"Unknown farm_id: {fields['farm_id']}")
                old_farm = next(f for f in self.data["farms"] if f["farm_id"] == plot["farm_id"])
                if len(old_farm["plot_ids"]) <= 1:
                    raise ValueError("Cannot move the only plot out of a farm")
                old_farm["plot_ids"].remove(plot_id)
                new_farm["plot_ids"].append(plot_id)
            plot.update(fields)
            self._save_unlocked()
            return deepcopy(plot)

    def remove_plot(self, plot_id: str) -> dict:
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            if len(self.data["plots"]) <= 1:
                raise ValueError("At least one plot must remain")
            plot = next((p for p in self.data["plots"] if p["plot_id"] == plot_id), None)
            if plot is None:
                raise KeyError(f"Unknown plot_id: {plot_id}")
            farm = next(f for f in self.data["farms"] if f["farm_id"] == plot["farm_id"])
            if len(farm["plot_ids"]) <= 1:
                raise ValueError("A farm must retain at least one plot")
            farm["plot_ids"].remove(plot_id)
            self.data["plots"] = [p for p in self.data["plots"] if p["plot_id"] != plot_id]
            self.data["current_plot_id"] = farm["plot_ids"][0]
            self.data["current_farm_id"] = farm["farm_id"]
            self._save_unlocked()
            return deepcopy(plot)
