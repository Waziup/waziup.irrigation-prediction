"""Persistent spatial farm/plot identities for one installation."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import math
from pathlib import Path
import threading
import uuid

from state_store import AppStateStore, configured_database_path
from crops import CROP_PARAMS


REGISTRY_VERSION = 3
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_REGISTRY_PATH = DEFAULT_CONFIG_DIR / "farm_registry.json"
AREA_UNITS = {"m2", "ha", "acre"}


def _new_id(kind: str) -> str:
    return f"{kind}-{uuid.uuid4().hex[:12]}"


class FarmRegistry:
    """Persistent registry retaining stable IDs across restarts."""

    def __init__(self, path=DEFAULT_REGISTRY_PATH, config_dir=DEFAULT_CONFIG_DIR):
        self.path = Path(path)
        self.config_dir = Path(config_dir)
        self._lock = threading.RLock()
        self.data = None
        self._committed_data = None
        self._defer_publish = False
        self._state_store = None

    def load_or_migrate(self) -> dict:
        """Load spatial state only; JSON/config scaffolds are not production data."""
        with self._lock:
            if self._state_store is None:
                self._state_store = AppStateStore(configured_database_path())
            stored = self._state_store.load_registry()
            self._validate(stored)
            self.data = stored
            self._committed_data = deepcopy(stored)
            return deepcopy(stored)

    def reload(self) -> dict:
        with self._lock:
            self.data = None
            return self.load_or_migrate()

    def snapshot(self) -> dict:
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            return deepcopy(self.data)

    def _validate(self, data):
        if data.get('version') != REGISTRY_VERSION:
            raise ValueError('Invalid spatial registry version')
        plots = data.get('plots', [])
        if data.get('farm') is None:
            if plots or data.get('current_plot_id') is not None:
                raise ValueError('An empty installation has no farm or plots')
            return
        if not isinstance(data['farm'], dict):
            raise ValueError('Invalid farm')
        ids = [p.get('plot_id') for p in plots]
        if not ids or None in ids or len(set(ids)) != len(ids):
            raise ValueError('At least one plot with a unique plot ID is required')
        if data.get('current_plot_id') not in ids:
            raise ValueError('Current plot must exist')

    def _save_unlocked(self) -> None:
        if self._defer_publish:
            return
        try:
            self._state_store.save_registry(self.data)
        except BaseException:
            self.data = deepcopy(self._committed_data)
            raise
        self._committed_data = deepcopy(self.data)

    @contextmanager
    def transaction(self):
        """Stage related mutations under one lock and publish one snapshot."""
        with self._lock:
            if self._defer_publish:
                raise RuntimeError('Nested registry transactions are unsupported')
            if self.data is None:
                self.load_or_migrate()
            before = deepcopy(self.data)
            self._defer_publish = True
            try:
                yield self
                self._validate(self.data)
                self._defer_publish = False
                self._save_unlocked()
            except BaseException:
                self.data = before
                raise
            finally:
                self._defer_publish = False

    def save(self) -> None:
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            self._validate(self.data)
            self._save_unlocked()

    def get_farm(self):
        return self.snapshot()['farm']

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
            self._save_unlocked()

    def create_farm(self, name, latitude=None, longitude=None, size=0,
                    area_unit='m2', timezone_name='UTC', owner='', gateway_id=None,
                    *, geometry=None, first_plot=None):
        """Create the farm and first mapped plot in one storage commit."""
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            if self.data['farm'] is not None:
                raise ValueError('This WaziGate already has a farm')
            if geometry is None or not isinstance(first_plot, dict):
                raise ValueError('Setup requires a farm boundary and first plot details/boundary')
            farm = dict(name=name, area_unit=area_unit, timezone=timezone_name,
                        owner=owner, gateway_id=gateway_id, geometry=deepcopy(geometry))
            for key, value in (('latitude', latitude), ('longitude', longitude)):
                if value is not None:
                    farm[key] = value
            plot = self._new_plot(first_plot.get('name'), first_plot.get('geometry'),
                                  first_plot.get('area_unit', 'm2'), 1, first_plot.get('crop_type'))
            before = deepcopy(self.data)
            try:
                self.data.update(farm=farm, plots=[plot], current_plot_id=plot['plot_id'])
                self._save_unlocked()
            except BaseException:
                self.data = before
                raise
            return deepcopy(self.data['farm'])

    @staticmethod
    def _new_plot(name, geometry, area_unit, legacy_id, crop_type):
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Plot name is required')
        if geometry is None:
            raise ValueError('Plot boundary is required')
        if crop_type is None:
            crop_type = ''
        if not isinstance(crop_type, str) or (crop_type and crop_type not in CROP_PARAMS):
            raise ValueError('A supported crop_type is required')
        return dict(plot_id=_new_id('plot'), legacy_id=legacy_id,
                    config_file=f'current_config_plot{legacy_id}.json',
                    name=name.strip(), crop_type=crop_type, geometry=deepcopy(geometry),
                    area_unit=area_unit, area=0, position=1)

    def update_farm(self, **fields):
        allowed = {'name','owner','latitude','longitude','size','area_unit','timezone','gateway_id'}
        if set(fields) - allowed:
            raise ValueError("Unsupported farm fields")
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            if self.data['farm'] is None:
                raise ValueError('Complete spatial setup before editing farm details')
            farm = deepcopy(self.data['farm'])
            if 'size' in fields and float(fields['size']) != float(farm['size']):
                raise ValueError('Farm area is calculated from its boundary')
            farm.update(fields)
            if not str(farm.get('name','')).strip():
                raise ValueError("Farm name is required")
            if farm.get('area_unit') not in AREA_UNITS:
                raise ValueError("Unsupported area unit")
            for key in ('latitude','longitude','size'):
                farm[key] = float(farm[key])
                if not math.isfinite(farm[key]):
                    raise ValueError(f"Invalid {key}")
            if not -90 <= farm['latitude'] <= 90 or not -180 <= farm['longitude'] <= 180 or farm['size'] < 0:
                raise ValueError("Invalid farm coordinates or size")
            from zoneinfo import ZoneInfo
            ZoneInfo(farm['timezone'])
            farm['configured'] = True
            self.data['farm'] = farm
            self._save_unlocked()
            return deepcopy(self.data['farm'])

    def add_plot(self, name='', area=0.0, area_unit='m2', *, geometry=None, crop_type=None):
        with self._lock:
            if self.data is None:
                self.load_or_migrate()
            if self.data['farm'] is None:
                raise ValueError('Complete farm setup before adding plots')
            plot = self._new_plot(name, geometry, area_unit, self.data['next_legacy_id'], crop_type)
            self.data['plots'].append(plot)
            self.data['current_plot_id'] = plot['plot_id']
            self._save_unlocked()
            return deepcopy(next(p for p in self.data['plots'] if p['plot_id'] == plot['plot_id']))

    def update_plot(self, plot_id: str, **fields) -> dict:
        allowed = {"name", "area", "area_unit"}
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
                if not math.isfinite(fields["area"]) or fields["area"] < 0:
                    raise ValueError("Plot area cannot be negative")
            if 'area' in fields and fields['area'] != plot['area']:
                raise ValueError('Plot area is calculated from its boundary')
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
            self.data['plots'] = [p for p in self.data['plots'] if p['plot_id'] != plot_id]
            if self.data['current_plot_id'] == plot_id:
                self.data['current_plot_id'] = self.data['plots'][0]['plot_id']
            self._save_unlocked()
            return deepcopy(plot)
