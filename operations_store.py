"""Persistent irrigation operations, schedules, events, and active alerts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid


MODES = {"automatic", "approval_required", "manual", "advisory_only"}
STATUSES = {"planned", "pending_approval", "approved", "active", "completed",
            "declined", "failed", "verified"}
TERMINAL_STATUSES = {"declined", "failed", "verified"}
ACTIVE_ALERT_URGENCIES = {"watch", "advise", "critical"}
TRANSITIONS = {
    "planned": {"pending_approval", "approved", "active", "declined", "failed"},
    "pending_approval": {"approved", "declined"},
    "approved": {"active", "declined", "failed"},
    "active": {"completed", "verified", "failed"},
    "completed": {"verified", "failed"},
    "declined": set(), "failed": set(), "verified": set(),
}


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(value):
    return json.dumps(value or {}, separators=(",", ":"), default=str)


class OperationsStore:
    def __init__(self, path="data/operations.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    farm_id TEXT,
                    plot_id TEXT NOT NULL,
                    plot_name TEXT,
                    source TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    amount_m3 REAL,
                    planned_start TEXT,
                    planned_end TEXT,
                    recommendation_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS operations_plot_status
                    ON operations(plot_id, status, updated_at);
                CREATE INDEX IF NOT EXISTS operations_farm_time
                    ON operations(farm_id, created_at);
                CREATE TABLE IF NOT EXISTS operation_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL REFERENCES operations(operation_id),
                    event_type TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    operation_id TEXT REFERENCES operations(operation_id),
                    farm_id TEXT,
                    plot_id TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS alerts_active_farm
                    ON alerts(active, farm_id, urgency, updated_at);
            """)

    @staticmethod
    def _operation(row):
        if row is None:
            return None
        item = dict(row)
        item["recommendation"] = json.loads(item.pop("recommendation_json") or "{}")
        return item

    @staticmethod
    def _alert(row):
        item = dict(row)
        item["active"] = bool(item["active"])
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def create_operation(self, *, idempotency_key, plot_id, farm_id=None,
                         plot_name="", source="recommendation", mode="manual",
                         status="planned", amount_m3=None, planned_start=None,
                         planned_end=None, recommendation=None):
        if mode not in MODES:
            raise ValueError(f"Unsupported irrigation mode: {mode}")
        if status not in STATUSES:
            raise ValueError(f"Unsupported operation status: {status}")
        if amount_m3 is not None and float(amount_m3) <= 0:
            raise ValueError("Irrigation amount must be greater than zero")
        operation_id = f"op-{uuid.uuid4().hex[:16]}"
        timestamp = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM operations WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing is not None:
                db.commit()
                return self._operation(existing), False
            db.execute("""
                INSERT INTO operations(operation_id,idempotency_key,farm_id,plot_id,plot_name,
                    source,mode,status,amount_m3,planned_start,planned_end,recommendation_json,
                    created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (operation_id, idempotency_key, farm_id, str(plot_id), plot_name,
                  source, mode, status, amount_m3, planned_start, planned_end,
                  _json(recommendation), timestamp, timestamp))
            db.execute("""INSERT INTO operation_events
                (operation_id,event_type,from_status,to_status,detail_json,timestamp)
                VALUES(?,?,?,?,?,?)""",
                       (operation_id, "created", None, status, "{}", timestamp))
            row = db.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
            db.commit()
        return self._operation(row), True

    def get_operation(self, operation_id):
        with self._connect() as db:
            return self._operation(db.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone())

    def transition(self, operation_id, to_status, detail=None):
        if to_status not in STATUSES:
            raise ValueError(f"Unsupported operation status: {to_status}")
        timestamp = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
            if row is None:
                db.rollback()
                raise KeyError(f"Unknown operation_id: {operation_id}")
            current = row["status"]
            if current == to_status:
                db.commit()
                return self._operation(row), False
            if to_status not in TRANSITIONS[current]:
                db.rollback()
                raise ValueError(f"Invalid irrigation transition: {current} -> {to_status}")
            error = (detail or {}).get("error") if to_status == "failed" else row["error"]
            db.execute("UPDATE operations SET status=?,error=?,updated_at=? WHERE operation_id=?",
                       (to_status, error, timestamp, operation_id))
            db.execute("""INSERT INTO operation_events
                (operation_id,event_type,from_status,to_status,detail_json,timestamp)
                VALUES(?,?,?,?,?,?)""",
                       (operation_id, "transition", current, to_status, _json(detail), timestamp))
            updated = db.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
            db.commit()
        return self._operation(updated), True

    def latest_for_plot(self, plot_id, statuses=None):
        clauses, values = ["plot_id=?"], [str(plot_id)]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            values.extend(statuses)
        query = "SELECT * FROM operations WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC LIMIT 1"
        with self._connect() as db:
            return self._operation(db.execute(query, values).fetchone())

    def list_operations(self, *, farm_id=None, plot_id=None, status=None,
                        search=None, limit=100, today=None):
        clauses, values = [], []
        if farm_id:
            clauses.append("farm_id=?"); values.append(farm_id)
        if plot_id:
            clauses.append("plot_id=?"); values.append(str(plot_id))
        if status:
            statuses = [status] if isinstance(status, str) else list(status)
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})"); values.extend(statuses)
        if search:
            clauses.append("(plot_name LIKE ? OR source LIKE ? OR status LIKE ?)")
            term = f"%{search}%"; values.extend([term, term, term])
        if today:
            clauses.append("date(COALESCE(planned_start,created_at))=date(?)"); values.append(today)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(max(1, min(int(limit), 500)))
        with self._connect() as db:
            rows = db.execute("SELECT * FROM operations" + where +
                              " ORDER BY COALESCE(planned_start,created_at) DESC LIMIT ?", values).fetchall()
        return [self._operation(row) for row in rows]

    def events(self, operation_id):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM operation_events WHERE operation_id=? ORDER BY event_id",
                              (operation_id,)).fetchall()
        result = []
        for row in rows:
            item = dict(row); item["detail"] = json.loads(item.pop("detail_json") or "{}")
            result.append(item)
        return result

    def record_alert(self, *, idempotency_key, plot_id, urgency, payload,
                     farm_id=None, operation_id=None):
        if urgency not in {"none", "watch", "advise", "critical", "error", "unknown"}:
            raise ValueError(f"Unsupported alert urgency: {urgency}")
        timestamp = _now(); active = int(urgency in ACTIVE_ALERT_URGENCIES)
        alert_id = f"alert-{uuid.uuid4().hex[:16]}"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM alerts WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing is not None:
                db.commit(); return self._alert(existing), False
            # The newest evaluated recommendation owns the active state for a
            # plot; older Watch/Advise/Critical alerts become history.
            db.execute("UPDATE alerts SET active=0,updated_at=? WHERE plot_id=? AND active=1",
                       (timestamp, str(plot_id)))
            db.execute("""INSERT INTO alerts(alert_id,idempotency_key,operation_id,farm_id,
                plot_id,urgency,active,payload_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (alert_id, idempotency_key, operation_id, farm_id, str(plot_id), urgency,
                        active, _json(payload), timestamp, timestamp))
            row = db.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            db.commit()
        return self._alert(row), True

    def active_alerts(self, farm_id=None):
        query = "SELECT * FROM alerts WHERE active=1"
        values = []
        if farm_id:
            query += " AND farm_id=?"; values.append(farm_id)
        query += " ORDER BY CASE urgency WHEN 'critical' THEN 1 WHEN 'advise' THEN 2 ELSE 3 END, updated_at DESC"
        with self._connect() as db:
            rows = db.execute(query, values).fetchall()
        return [self._alert(row) for row in rows]


_default_store = None


def get_operations_store():
    global _default_store
    if _default_store is None:
        _default_store = OperationsStore()
    return _default_store
