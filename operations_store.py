"""Persistent irrigation operations, schedules, events, and active alerts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import uuid
from zoneinfo import ZoneInfo


MODES = {"automatic", "approval_required", "manual", "advisory_only"}
STATUSES = {"planned", "pending_approval", "approved", "active", "completed",
            "declined", "failed", "verified"}
TERMINAL_STATUSES = {"declined", "failed", "verified"}
ACTIVE_ALERT_URGENCIES = {"watch", "advise", "critical"}
TRANSITIONS = {
    "planned": {"pending_approval", "approved", "active", "declined", "failed"},
    "pending_approval": {"approved", "declined"},
    "approved": {"pending_approval", "active", "declined", "failed"},
    "active": {"completed", "verified", "failed"},
    "completed": {"verified", "failed"},
    "declined": set(), "failed": set(), "verified": set(),
}


def _now():
    # Preserve sub-second ordering so a command completed immediately after a
    # calculation is not accidentally timestamped before that calculation.
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value or {}, separators=(",", ":"), default=str)


class OperationsStore:
    def __init__(self, path="data/operations.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

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
        if amount_m3 is not None:
            amount_m3 = float(amount_m3)
            if not math.isfinite(amount_m3) or amount_m3 <= 0:
                raise ValueError("Irrigation amount must be finite and greater than zero")
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
                        search=None, limit=100, today=None, timezone_name="UTC"):
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
            day = date.fromisoformat(str(today))
            zone = ZoneInfo(timezone_name)
            start = datetime.combine(day, time.min, tzinfo=zone).astimezone(timezone.utc)
            end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone).astimezone(timezone.utc)
            # Half-open local-day bounds also handle 23/25-hour DST days.
            # SQLite normalizes stored offsets; comparing date strings would
            # instead select the UTC day regardless of the farm's timezone.
            clauses.extend([
                "julianday(COALESCE(planned_start,created_at)) >= julianday(?)",
                "julianday(COALESCE(planned_start,created_at)) < julianday(?)",
            ])
            values.extend([start.isoformat(), end.isoformat()])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(max(1, min(int(limit), 500)))
        with self._connect() as db:
            rows = db.execute("SELECT * FROM operations" + where +
                              " ORDER BY COALESCE(planned_start,created_at) DESC LIMIT ?", values).fetchall()
        return [self._operation(row) for row in rows]

    def due_schedules(self, now, *, after_rowid=0, limit=100):
        """Page executable schedules without dashboard limits or future starvation."""
        with self._connect() as db:
            rows = db.execute("""SELECT rowid AS dispatch_cursor, * FROM operations
                WHERE rowid > ? AND source='schedule'
                AND status IN ('planned','approved')
                AND (julianday(planned_start) IS NULL OR julianday(planned_start) <= julianday(?))
                ORDER BY rowid LIMIT ?""",
                (after_rowid, str(now), max(1, min(int(limit), 500)))).fetchall()
        return [(row['dispatch_cursor'], self._operation(row)) for row in rows]

    def events(self, operation_id):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM operation_events WHERE operation_id=? ORDER BY event_id",
                              (operation_id,)).fetchall()
        result = []
        for row in rows:
            item = dict(row); item["detail"] = json.loads(item.pop("detail_json") or "{}")
            result.append(item)
        return result

    def has_pending_flow_verification(self, plot_id):
        """Return whether an accepted command still awaits meter evidence."""
        with self._connect() as db:
            row = db.execute("""
                SELECT 1
                FROM operations AS o
                JOIN operation_events AS e
                  ON e.operation_id = o.operation_id
                WHERE o.plot_id = ?
                  AND o.status = 'completed'
                  AND e.to_status = 'completed'
                  AND json_type(e.detail_json, '$.flow_verification') IS NOT NULL
                LIMIT 1
            """, (str(plot_id),)).fetchone()
        return row is not None

    def applied_irrigation_since(self, plot_id, since, until=None):
        """Return recorded applied volume after a calculation checkpoint.

        An operation contributes once: at command completion when no meter is
        required, or at the later verification event when a meter is used.
        Failed, declined, merely planned, approved, or active operations are
        excluded. A completed operation with flow-verification context is not
        credited until it reaches ``verified``; ordinary completed operations
        retain compatibility for installations without a confirmation meter.
        Verified operations use measured delivered volume when it is present.
        """
        if since is None:
            return {"volume_m3": 0.0, "operation_count": 0,
                    "since": None, "until": None}

        def normalized_timestamp(value):
            if isinstance(value, datetime):
                parsed = value
            else:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()

        start = normalized_timestamp(since)
        end = normalized_timestamp(until or datetime.now(timezone.utc))
        if end < start:
            raise ValueError("until must not be earlier than since")

        with self._connect() as db:
            row = db.execute("""
                SELECT COALESCE(SUM(
                           CASE WHEN o.status = 'verified'
                             THEN COALESCE(applied.delivered_m3, o.amount_m3)
                             ELSE o.amount_m3
                           END), 0.0) AS volume_m3,
                       COUNT(*) AS operation_count
                FROM operations AS o
                JOIN (
                    SELECT operation_id,
                           COALESCE(
                             MIN(CASE WHEN to_status = 'verified'
                                      THEN timestamp END),
                             MIN(CASE WHEN to_status = 'completed'
                                      THEN timestamp END)
                           ) AS applied_at,
                           MAX(CASE WHEN to_status = 'verified'
                               THEN CAST(json_extract(
                                 detail_json, '$.delivered_m3') AS REAL)
                               END) AS delivered_m3
                    FROM operation_events
                    WHERE to_status IN ('completed', 'verified')
                    GROUP BY operation_id
                ) AS applied ON applied.operation_id = o.operation_id
                WHERE o.plot_id = ?
                  AND o.status IN ('completed', 'verified')
                  AND (
                    o.status = 'verified'
                    OR NOT EXISTS (
                      SELECT 1 FROM operation_events AS verification_pending
                      WHERE verification_pending.operation_id = o.operation_id
                        AND verification_pending.to_status = 'completed'
                        AND json_type(
                          verification_pending.detail_json,
                          '$.flow_verification') IS NOT NULL
                    )
                  )
                  AND o.amount_m3 IS NOT NULL
                  AND o.amount_m3 > 0
                  AND applied.applied_at > ?
                  AND applied.applied_at <= ?
            """, (str(plot_id), start, end)).fetchone()
        return {
            "volume_m3": float(row["volume_m3"] or 0.0),
            "operation_count": int(row["operation_count"] or 0),
            "since": start,
            "until": end,
        }

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
        _default_store = OperationsStore(
            os.getenv("IRRIGATION_OPERATIONS_DB", "data/operations.sqlite3"))
    return _default_store
