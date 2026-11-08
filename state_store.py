"""Persistent SQLite storage for application-owned configuration and history."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import sqlite3
import threading
from copy import deepcopy

from geometry_validation import validate_farm_layout
from crops import CROP_PARAMS

_transactions = threading.local()


class SpatialSchemaError(ValueError):
    """Pre-spatial development state must be explicitly replaced, never guessed."""


class RegistryConflictError(ValueError):
    """The installation layout changed after this snapshot was loaded."""


def configured_database_path():
    value = (os.getenv("APP_STATE_DB") or "").strip()
    return Path(value) if value else Path(__file__).resolve().parent / "data/wazifarm.sqlite3"


def database_path(default="data/wazifarm.sqlite3"):
    return configured_database_path() or Path(default)


def _now():
    return datetime.now(timezone.utc).isoformat()


class AppStateStore:
    def __init__(self, path=None):
        self.path = Path(path or database_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self):
        active = getattr(_transactions, 'connections', {}).get(str(self.path.resolve()))
        if active is not None:
            yield active
            return
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA foreign_keys=ON")
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        """Share a transaction across stores pointing at this installation."""
        key = str(self.path.resolve())
        connections = getattr(_transactions, 'connections', {})
        if key in connections:
            raise RuntimeError('Nested state transaction')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            _transactions.connections = {**connections, key: db}
            try:
                yield
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                _transactions.connections = connections

    def _initialize(self):
        with self.connect() as db:
            # Check before DDL: do not rewrite or partially upgrade development
            # records that cannot satisfy mandatory spatial invariants.
            tables = {r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if 'farms' in tables:
                raise SpatialSchemaError('Pre-spatial development database: use a new APP_STATE_DB or explicitly replace development fixtures.')
            for table in ('farm', 'plots'):
                if table in tables:
                    columns = {r[1] for r in db.execute(f'PRAGMA table_info({table})')}
                    if 'geometry_json' not in columns:
                        raise SpatialSchemaError('Pre-spatial development database: use a new APP_STATE_DB or explicitly replace development fixtures.')
                    if table == 'plots' and 'crop_type' not in columns:
                        raise SpatialSchemaError('Incomplete development database schema: use a new APP_STATE_DB or explicitly replace development fixtures.')
            db.executescript("""
                CREATE TABLE IF NOT EXISTS app_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS farm (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    name TEXT NOT NULL, gateway_id TEXT, owner TEXT NOT NULL DEFAULT '',
                    latitude REAL NOT NULL, longitude REAL NOT NULL, size REAL NOT NULL,
                    area_unit TEXT NOT NULL, timezone TEXT NOT NULL,
                    geometry_json TEXT NOT NULL CHECK(json_valid(geometry_json)
                        AND json_extract(geometry_json,'$.type') IS 'Polygon'),
                    geometry_type TEXT NOT NULL DEFAULT 'Polygon' CHECK(geometry_type='Polygon'),
                    geometry_version TEXT NOT NULL,
                    area_m2 REAL NOT NULL CHECK(area_m2 > 0 AND area_m2 <= 1000000000),
                    geometry_updated_at TEXT NOT NULL,
                    geometry_updated_by TEXT
                );
                CREATE TABLE IF NOT EXISTS plots (
                    plot_id TEXT PRIMARY KEY,
                    legacy_id INTEGER NOT NULL UNIQUE,
                    config_file TEXT NOT NULL,
                    name TEXT NOT NULL, crop_type TEXT NOT NULL,
                    area REAL NOT NULL, area_unit TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    farm_singleton INTEGER NOT NULL DEFAULT 1 CHECK(farm_singleton=1)
                        REFERENCES farm(singleton) ON DELETE RESTRICT,
                    geometry_json TEXT NOT NULL CHECK(json_valid(geometry_json)
                        AND json_extract(geometry_json,'$.type') IS 'Polygon'),
                    geometry_type TEXT NOT NULL DEFAULT 'Polygon' CHECK(geometry_type='Polygon'),
                    geometry_version TEXT NOT NULL,
                    area_m2 REAL NOT NULL CHECK(area_m2 > 0 AND area_m2 <= 1000000000),
                    geometry_updated_at TEXT NOT NULL,
                    geometry_updated_by TEXT
                );
                CREATE TABLE IF NOT EXISTS plot_configs (
                    plot_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sensor_registry (
                    sensor_key TEXT PRIMARY KEY,
                    plot_id TEXT,
                    role TEXT,
                    device_sensor_id TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sensors_plot
                    ON sensor_registry(plot_id, role);
                CREATE TABLE IF NOT EXISTS irrigation_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plot_id TEXT NOT NULL,
                    operation_id TEXT,
                    timestamp TEXT NOT NULL,
                    amount_m3 REAL,
                    status TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE UNIQUE INDEX IF NOT EXISTS irrigation_history_operation
                    ON irrigation_history(operation_id)
                    WHERE operation_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS irrigation_history_plot_time
                    ON irrigation_history(plot_id, timestamp);
                CREATE TABLE IF NOT EXISTS alert_history (
                    alert_history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plot_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS alert_history_plot_time
                    ON alert_history(plot_id, alert_history_id);
                CREATE TABLE IF NOT EXISTS runtime_state (
                    state_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    def get_metadata(self, key, default=None):
        with self.connect() as db:
            row = db.execute(
                "SELECT value_json FROM app_metadata WHERE key=?", (str(key),)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except (TypeError, ValueError):
            return default

    def set_metadata(self, key, value):
        timestamp = _now()
        with self.connect() as db:
            db.execute("""INSERT INTO app_metadata(key,value_json,updated_at)
                VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET
                value_json=excluded.value_json,updated_at=excluded.updated_at""",
                (str(key), json.dumps(value, default=str), timestamp))

    def legacy_import_complete(self, scope):
        return self.get_metadata(f"legacy_import:{scope}", False) is True

    def mark_legacy_import_complete(self, scope):
        self.set_metadata(f"legacy_import:{scope}", True)

    @contextmanager
    def _registry_connection(self, write=False):
        """Consistent snapshots, and rollback even when an outer caller catches errors."""
        with self.connect() as db:
            owns_transaction = not db.in_transaction
            if owns_transaction:
                db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            db.execute('SAVEPOINT registry_state')
            try:
                yield db
                db.execute('RELEASE registry_state')
                if owns_transaction:
                    db.commit()
            except BaseException:
                db.execute('ROLLBACK TO registry_state')
                db.execute('RELEASE registry_state')
                if owns_transaction:
                    db.rollback()
                raise

    @staticmethod
    def empty_registry():
        return dict(version=3, farm=None, plots=[], current_plot_id=None,
                    next_legacy_id=1, registry_revision=0,
                    created_at=None, updated_at=None)

    @staticmethod
    def _spatial_record(row):
        record = dict(row)
        record.pop('singleton', None)
        record.pop('farm_singleton', None)
        record['geometry'] = json.loads(record.pop('geometry_json'))
        return record

    def load_registry(self):
        with self._registry_connection() as db:
            meta = {r['key']: json.loads(r['value_json']) for r in db.execute(
                "SELECT key,value_json FROM app_metadata")}
            row = db.execute("SELECT * FROM farm WHERE singleton=1").fetchone()
            plots = [self._spatial_record(r) for r in db.execute(
                "SELECT * FROM plots ORDER BY position,legacy_id")]
            if row is None:
                if plots:
                    raise SpatialSchemaError('Orphan plots in spatial database')
                return self.empty_registry()
            if not plots:
                raise SpatialSchemaError('A saved farm must have at least one plot')
            farm = self._spatial_record(row)
            farm['configured'] = True
            result = dict(version=3, farm=farm, plots=plots,
                current_plot_id=meta.get('current_plot_id'),
                next_legacy_id=meta.get('next_plot_legacy_id'),
                registry_revision=meta.get('registry_revision'),
                created_at=meta.get('registry_created_at'),
                updated_at=meta.get('registry_updated_at'))
            if result['current_plot_id'] not in {p['plot_id'] for p in plots}:
                raise SpatialSchemaError('Selected plot is missing from spatial database')
            return result

    @staticmethod
    def _boundary_columns(boundary, entity_id, previous, timestamp, updated_by):
        geometry = boundary.as_geojson()
        encoded = json.dumps(geometry, separators=(',', ':'), allow_nan=False)
        version = hashlib.sha256((entity_id + ':' + encoded).encode()).hexdigest()
        unchanged = previous is not None and previous['geometry_version'] == version
        return dict(geometry=geometry, geometry_type='Polygon',
                    geometry_version=version, area_m2=boundary.area_m2,
                    geometry_updated_at=previous['geometry_updated_at'] if unchanged else timestamp,
                    geometry_updated_by=previous['geometry_updated_by'] if unchanged else updated_by)

    def save_registry(self, data, *, updated_by=None):
        """Commit a complete single-farm layout with optimistic concurrency.

        registry_revision must come from load_registry (0 for first setup).
        Geometry, metadata, derived areas and initial plot commit together.
        No missing-geometry path or boundary-only delete operation exists.
        """
        candidate = deepcopy(data)
        if candidate.get('version') != 3:
            raise ValueError('Expected spatial registry version 3')
        farm = candidate.get('farm')
        plots = candidate.get('plots')
        if not isinstance(farm, dict) or not isinstance(plots, list) or not plots:
            raise ValueError('Setup requires a farm boundary and at least one plot boundary')
        for key, value in {'gateway_id': None, 'owner': '', 'area_unit': 'm2', 'timezone': 'UTC'}.items():
            farm.setdefault(key, value)
        for plot in plots:
            if isinstance(plot, dict):
                plot.setdefault('area_unit', 'm2')
        ids = [p.get('plot_id') for p in plots if isinstance(p, dict)]
        if len(ids) != len(plots) or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Plots require unique stable IDs')
        if candidate.get('current_plot_id') not in ids:
            raise ValueError('Current plot must exist')
        if type(candidate.get('registry_revision')) is not int:
            raise RegistryConflictError('Reload the installation before saving')
        if updated_by is not None and (not isinstance(updated_by, str) or not updated_by.strip()):
            raise ValueError('updated_by must be a nonempty actor identifier or None')
        from zoneinfo import ZoneInfo
        for record in [farm, *plots]:
            if not isinstance(record.get('name'), str) or not record['name'].strip():
                raise ValueError('Farm and plot names are required')
            if record.get('area_unit', 'm2') not in {'m2', 'ha', 'acre'}:
                raise ValueError('Unsupported area unit')
        ZoneInfo(farm.get('timezone', 'UTC'))
        legacy_ids = [p.get('legacy_id') for p in plots]
        if any(type(i) is not int or i < 1 for i in legacy_ids) or len(set(legacy_ids)) != len(legacy_ids):
            raise ValueError('Plots require distinct positive legacy compatibility IDs')
        for p in plots:
            if not isinstance(p.get('crop_type'), str) or (p['crop_type'] and p['crop_type'] not in CROP_PARAMS):
                raise ValueError('A supported crop_type is required for each plot')
            if p.get('config_file') != f"current_config_plot{p['legacy_id']}.json":
                raise ValueError('Invalid plot configuration filename')
        with self._registry_connection(write=True) as db:
            row = db.execute("SELECT value_json FROM app_metadata WHERE key='registry_revision'").fetchone()
            revision = json.loads(row[0]) if row else 0
            if candidate['registry_revision'] != revision:
                raise RegistryConflictError('Farm layout changed; reload before saving')
            # Validation is inside the write lock; every plot belongs to the
            # singleton farm. Callers submit the full proposed layout.
            farm_boundary, boundaries = validate_farm_layout(
                farm.get('geometry'), {p['plot_id']: p.get('geometry') for p in plots})
            timestamp = _now()
            previous_farm = db.execute('SELECT * FROM farm WHERE singleton=1').fetchone()
            previous_plots = {r['plot_id']: r for r in db.execute('SELECT * FROM plots')}
            farm.update(self._boundary_columns(farm_boundary, 'installation-farm', previous_farm, timestamp, updated_by))
            # Retain display units, but derive the actual area from the polygon.
            factor = {'m2': 1, 'ha': 10000, 'acre': 4046.8564224}[farm.get('area_unit', 'm2')]
            farm['size'] = farm_boundary.area_m2 / factor
            # A supplied reference point is metadata, never a substitute boundary.
            from math import isfinite
            shape_lons, shape_lats = zip(*farm_boundary.positions[:-1])
            for key, default, limit in (('latitude', sum(shape_lats)/len(shape_lats), 90),
                                        ('longitude', sum(shape_lons)/len(shape_lons), 180)):
                value = farm.get(key, default)
                if type(value) not in (int, float) or not isfinite(value) or abs(value) > limit:
                    raise ValueError('Invalid farm reference coordinates')
                farm[key] = value
            values = (farm['name'], farm.get('gateway_id'), farm.get('owner', ''),
                farm['latitude'], farm['longitude'], farm['size'], farm.get('area_unit', 'm2'),
                farm.get('timezone', 'UTC'), json.dumps(farm['geometry']), 'Polygon',
                farm['geometry_version'], farm['area_m2'], farm['geometry_updated_at'], farm['geometry_updated_by'])
            db.execute("""INSERT INTO farm(singleton,name,gateway_id,owner,latitude,longitude,size,area_unit,timezone,
                geometry_json,geometry_type,geometry_version,area_m2,geometry_updated_at,geometry_updated_by)
                VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET
                name=excluded.name,gateway_id=excluded.gateway_id,owner=excluded.owner,
                latitude=excluded.latitude,longitude=excluded.longitude,size=excluded.size,
                area_unit=excluded.area_unit,timezone=excluded.timezone,
                geometry_json=excluded.geometry_json,geometry_type=excluded.geometry_type,
                geometry_version=excluded.geometry_version,area_m2=excluded.area_m2,
                geometry_updated_at=excluded.geometry_updated_at,geometry_updated_by=excluded.geometry_updated_by""", values)
            for removed in previous_plots.keys() - set(ids):
                db.execute('DELETE FROM plot_configs WHERE plot_id=?', (removed,))
                db.execute('DELETE FROM sensor_registry WHERE plot_id=?', (removed,))
            db.execute('DELETE FROM plots')
            for position, plot in enumerate(plots, start=1):
                prior = previous_plots.get(plot['plot_id'])
                config = db.execute('SELECT config_json FROM plot_configs WHERE plot_id=?', (plot['plot_id'],)).fetchone()
                if config:
                    # Once configured, settings is the authoritative crop source.
                    # An older registry cache must not undo a settings crop edit.
                    crop = json.loads(config[0]).get('Crop_type', plot['crop_type'])
                    if not isinstance(crop, str) or (crop and crop not in CROP_PARAMS):
                        raise ValueError('Configuration contains an unsupported crop_type')
                    plot['crop_type'] = crop
                if prior is not None and (plot['legacy_id'] != prior['legacy_id'] or plot['config_file'] != prior['config_file']):
                    raise ValueError('Existing plot compatibility identity cannot change')
                if prior is None:
                    # The persisted high-water mark is authoritative.
                    row = db.execute("SELECT value_json FROM app_metadata WHERE key='next_plot_legacy_id'").fetchone()
                    if row and plot['legacy_id'] < json.loads(row[0]):
                        raise ValueError('Removed plot numbers cannot be reused')
                plot.update(self._boundary_columns(boundaries[plot['plot_id']], plot['plot_id'], prior, timestamp, updated_by))
                plot['area'] = plot['area_m2']
                plot['position'] = position
                db.execute("""INSERT INTO plots(plot_id,legacy_id,config_file,name,crop_type,area,area_unit,position,
                    farm_singleton,geometry_json,geometry_type,geometry_version,area_m2,geometry_updated_at,geometry_updated_by)
                    VALUES(?,?,?,?,?,?,?,?,1,?,?,?,?,?,?)""", (
                    plot['plot_id'], plot['legacy_id'], plot['config_file'], plot['name'], plot['crop_type'], plot['area'],
                    plot.get('area_unit', 'm2'), position, json.dumps(plot['geometry']), 'Polygon',
                    plot['geometry_version'], plot['area_m2'], plot['geometry_updated_at'], plot['geometry_updated_by']))
                db.execute('INSERT OR REPLACE INTO app_metadata VALUES(?,?,?)',
                    (f"legacy_import:plot_config:{plot['plot_id']}", 'true', timestamp))
            counter = db.execute("SELECT value_json FROM app_metadata WHERE key='next_plot_legacy_id'").fetchone()
            candidate['next_legacy_id'] = max(max(legacy_ids)+1, json.loads(counter[0]) if counter else 1)
            candidate['created_at'] = (db.execute("SELECT value_json FROM app_metadata WHERE key='registry_created_at'").fetchone() or [None])[0]
            candidate['created_at'] = json.loads(candidate['created_at']) if candidate['created_at'] else timestamp
            candidate.update(updated_at=timestamp, registry_revision=revision+1)
            farm['configured'] = True
            for key, value in {'next_plot_legacy_id': candidate['next_legacy_id'],
                'farm_setup_complete': True, 'registry_version': 3,
                'registry_revision': candidate['registry_revision'],
                'current_plot_id': candidate['current_plot_id'],
                'registry_created_at': candidate['created_at'], 'registry_updated_at': timestamp}.items():
                db.execute('INSERT OR REPLACE INTO app_metadata VALUES(?,?,?)', (key, json.dumps(value), timestamp))
            # Area used by operational settings must never diverge from geometry.
            for plot in plots:
                row = db.execute('SELECT config_json FROM plot_configs WHERE plot_id=?', (plot['plot_id'],)).fetchone()
                if row:
                    config = json.loads(row[0])
                    config['Plot_area_m2'] = plot['area_m2']
                    db.execute('UPDATE plot_configs SET config_json=?,updated_at=? WHERE plot_id=?',
                               (json.dumps(config), timestamp, plot['plot_id']))
        data.clear()
        data.update(candidate)
        return deepcopy(candidate)

    def load_plot_config(self, plot_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT config_json FROM plot_configs WHERE plot_id=?",
                (str(plot_id),)).fetchone()
        return {k: v for k, v in json.loads(row["config_json"]).items() if k != "Farm_id"} if row else None

    def plot_config_version(self, plot_id):
        """Opaque content version; compare inside the same transaction as a save.

        Includes the stable resource ID and the unconfigured state. Existing
        writers automatically invalidate tokens without a database migration.
        """
        snapshot = {"plot_id": str(plot_id), "config": self.load_plot_config(plot_id)}
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                             default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save_plot_config(self, plot_id, config):
        config = {k: v for k, v in config.items() if k != "Farm_id"}
        with self._registry_connection(write=True) as db:
            row = db.execute('SELECT area_m2 FROM plots WHERE plot_id=?', (str(plot_id),)).fetchone()
            if row is None:
                raise KeyError(f'Unknown plot_id: {plot_id}')
            config['Plot_area_m2'] = row['area_m2']
            if 'Crop_type' in config:
                crop = config['Crop_type']
                if not isinstance(crop, str) or (crop and crop not in CROP_PARAMS):
                    raise ValueError('A supported crop_type is required')
                db.execute('UPDATE plots SET crop_type=? WHERE plot_id=?', (crop, str(plot_id)))
            db.execute("""INSERT INTO plot_configs(plot_id,config_json,updated_at)
                VALUES(?,?,?) ON CONFLICT(plot_id) DO UPDATE SET
                config_json=excluded.config_json,updated_at=excluded.updated_at""",
                (str(plot_id), json.dumps(config, default=str), _now()))

    def delete_plot_config(self, plot_id):
        with self.connect() as db:
            db.execute("DELETE FROM plot_configs WHERE plot_id=?", (str(plot_id),))

    def load_sensor_registry(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload_json FROM sensor_registry ORDER BY sensor_key").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def save_sensor_registry(self, records):
        with self.connect() as db:
            owns_transaction = not db.in_transaction
            if owns_transaction:
                db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM sensor_registry")
            for record in records:
                db.execute("""INSERT INTO sensor_registry
                    (sensor_key,plot_id,role,device_sensor_id,payload_json,updated_at)
                    VALUES(?,?,?,?,?,?)""", (
                    record["sensor_key"], str(record.get("plot_id", "")),
                    record.get("role"), record.get("device_sensor_id"),
                    json.dumps(record, default=str), _now()))
            if owns_transaction:
                db.commit()

    def load_irrigation_history(self, plot_id):
        with self.connect() as db:
            rows = db.execute("""SELECT timestamp,amount_m3,status,operation_id,
                detail_json FROM irrigation_history WHERE plot_id=?
                ORDER BY history_id""", (str(plot_id),)).fetchall()
        result = []
        for row in rows:
            item = {"timestamp": row["timestamp"], "amount": row["amount_m3"],
                    "status": row["status"]}
            if row["operation_id"]:
                item["operation_id"] = row["operation_id"]
            detail = json.loads(row["detail_json"] or "{}")
            if detail:
                item.update(detail)
            result.append(item)
        return result

    def save_irrigation_history(self, plot_id, records):
        with self.connect() as db:
            owns_transaction = not db.in_transaction
            if owns_transaction:
                db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM irrigation_history WHERE plot_id=?",
                       (str(plot_id),))
            for record in records:
                detail = {
                    key: value for key, value in record.items()
                    if key not in {"timestamp", "amount", "status", "operation_id"}
                }
                db.execute("""INSERT INTO irrigation_history
                    (plot_id,operation_id,timestamp,amount_m3,status,detail_json)
                    VALUES(?,?,?,?,?,?)""", (
                    str(plot_id), record.get("operation_id"),
                    str(record.get("timestamp", "")), record.get("amount"),
                    str(record.get("status", "commanded")),
                    json.dumps(detail, default=str)))
            if owns_transaction:
                db.commit()

    def append_alert(self, plot_id, payload):
        timestamp = str(payload.get("timestamp_utc") or _now())
        serialized = json.dumps(payload, default=str)
        with self.connect() as db:
            owns_transaction = not db.in_transaction
            if owns_transaction:
                db.execute("BEGIN IMMEDIATE")
            db.execute("""INSERT INTO alert_history(plot_id,timestamp,payload_json)
                VALUES(?,?,?)""", (str(plot_id), timestamp, serialized))
            db.execute("""INSERT INTO runtime_state(state_key,payload_json,updated_at)
                VALUES(?,?,?) ON CONFLICT(state_key) DO UPDATE SET
                payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (f"latest_alert:{plot_id}", serialized, _now()))
            if owns_transaction:
                db.commit()

    def load_latest_alert(self, plot_id):
        return self.load_runtime_state(f"latest_alert:{plot_id}")

    def load_runtime_state(self, key, default=None):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload_json FROM runtime_state WHERE state_key=?",
                (str(key),)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["payload_json"])
        except (TypeError, ValueError):
            return default

    def save_runtime_state(self, key, payload):
        with self.connect() as db:
            db.execute("""INSERT INTO runtime_state(state_key,payload_json,updated_at)
                VALUES(?,?,?) ON CONFLICT(state_key) DO UPDATE SET
                payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (str(key), json.dumps(payload, default=str), _now()))


_store = None
_store_path = None


def get_app_state_store():
    global _store, _store_path
    path = database_path()
    if _store is None or _store_path != path:
        _store = AppStateStore(path)
        _store_path = path
    return _store
