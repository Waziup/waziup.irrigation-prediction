"""Persistent SQLite storage for application-owned configuration and history."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3


def configured_database_path():
    value = (os.getenv("APP_STATE_DB") or "").strip()
    return Path(value) if value else None


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
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA foreign_keys=ON")
            yield db
        finally:
            db.close()

    def _initialize(self):
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS app_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS farms (
                    farm_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    gateway_id TEXT,
                    owner TEXT NOT NULL DEFAULT '',
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    size REAL NOT NULL,
                    area_unit TEXT NOT NULL,
                    timezone TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plots (
                    plot_id TEXT PRIMARY KEY,
                    farm_id TEXT NOT NULL REFERENCES farms(farm_id),
                    legacy_id INTEGER NOT NULL UNIQUE,
                    config_file TEXT NOT NULL,
                    name TEXT NOT NULL,
                    area REAL NOT NULL,
                    area_unit TEXT NOT NULL,
                    position INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS plots_farm_position
                    ON plots(farm_id, position);
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

    def load_registry(self):
        with self.connect() as db:
            meta = {row["key"]: json.loads(row["value_json"])
                    for row in db.execute("SELECT key,value_json FROM app_metadata")}
            farms = [dict(row) for row in db.execute(
                "SELECT * FROM farms ORDER BY rowid")]
            plots = [dict(row) for row in db.execute(
                "SELECT * FROM plots ORDER BY position,legacy_id")]
        if not farms or not plots:
            return None
        for farm in farms:
            farm["plot_ids"] = [plot["plot_id"] for plot in plots
                                if plot["farm_id"] == farm["farm_id"]]
        return {
            "version": int(meta.get("registry_version", 1)),
            "current_farm_id": meta.get("current_farm_id"),
            "current_plot_id": meta.get("current_plot_id"),
            "farms": farms,
            "plots": plots,
            "created_at": meta.get("registry_created_at"),
            "updated_at": meta.get("registry_updated_at"),
        }

    def save_registry(self, data):
        timestamp = _now()
        data["updated_at"] = timestamp
        data.setdefault("created_at", timestamp)
        metadata = {
            "registry_version": data["version"],
            "current_farm_id": data["current_farm_id"],
            "current_plot_id": data["current_plot_id"],
            "registry_created_at": data["created_at"],
            "registry_updated_at": data["updated_at"],
            "legacy_json_imported": True,
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM plots")
            db.execute("DELETE FROM farms")
            for farm in data["farms"]:
                db.execute("""INSERT INTO farms
                    (farm_id,name,gateway_id,owner,latitude,longitude,size,area_unit,timezone)
                    VALUES(?,?,?,?,?,?,?,?,?)""", (
                    farm["farm_id"], farm["name"], farm.get("gateway_id"),
                    farm.get("owner", ""), farm.get("latitude", 0),
                    farm.get("longitude", 0), farm.get("size", 0),
                    farm.get("area_unit", "m2"), farm.get("timezone", "UTC")))
            for plot in data["plots"]:
                db.execute("""INSERT INTO plots
                    (plot_id,farm_id,legacy_id,config_file,name,area,area_unit,position)
                    VALUES(?,?,?,?,?,?,?,?)""", (
                    plot["plot_id"], plot["farm_id"], plot["legacy_id"],
                    plot["config_file"], plot["name"], plot.get("area", 0),
                    plot.get("area_unit", "m2"), plot.get("position", 1)))
            for key, value in metadata.items():
                db.execute("""INSERT INTO app_metadata(key,value_json,updated_at)
                    VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,updated_at=excluded.updated_at""",
                    (key, json.dumps(value), timestamp))
            db.commit()

    def load_plot_config(self, plot_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT config_json FROM plot_configs WHERE plot_id=?",
                (str(plot_id),)).fetchone()
        return json.loads(row["config_json"]) if row else None

    def save_plot_config(self, plot_id, config):
        with self.connect() as db:
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
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM sensor_registry")
            for record in records:
                db.execute("""INSERT INTO sensor_registry
                    (sensor_key,plot_id,role,device_sensor_id,payload_json,updated_at)
                    VALUES(?,?,?,?,?,?)""", (
                    record["sensor_key"], str(record.get("plot_id", "")),
                    record.get("role"), record.get("device_sensor_id"),
                    json.dumps(record, default=str), _now()))
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
            db.commit()

    def append_alert(self, plot_id, payload):
        timestamp = str(payload.get("timestamp_utc") or _now())
        serialized = json.dumps(payload, default=str)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""INSERT INTO alert_history(plot_id,timestamp,payload_json)
                VALUES(?,?,?)""", (str(plot_id), timestamp, serialized))
            db.execute("""INSERT INTO runtime_state(state_key,payload_json,updated_at)
                VALUES(?,?,?) ON CONFLICT(state_key) DO UPDATE SET
                payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (f"latest_alert:{plot_id}", serialized, _now()))
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
