"""Raw soil-tension observation capture and automatic-control safety checks.

Modelled/interpolated rows are deliberately excluded.  Automatic irrigation
must be justified by the timestamp and value of the physical observations that
anchor the forecast, not merely by the last row in a resampled dataframe.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sensor_roles import soil_sensor_groups


# Conservative application safeguards. These are not agronomic calibration
# inputs and ordinary users must not be able to weaken them from the UI.
SENSOR_STALE_HOURS = 6.0
TENSION_MIN_CBAR = 0.0
TENSION_MAX_CBAR = 200.0
MAX_SENSOR_SPREAD_CBAR = 20.0
AUTOMATIC_MINIMUM_SENSORS = 2


def capture_tension_observations(
    plot,
    raw_frame: pd.DataFrame,
    sensor_ids: Optional[Iterable[str]] = None,
) -> list[dict]:
    """Store the latest non-null raw reading for each configured tension sensor."""
    sensor_ids, _, _ = soil_sensor_groups(plot, sensor_ids)
    observations: list[dict] = []
    if not isinstance(raw_frame, pd.DataFrame) or raw_frame.empty:
        plot.tension_sensor_observations = observations
        return observations

    for sensor_id in sensor_ids:
        if sensor_id not in raw_frame.columns:
            continue
        values = pd.to_numeric(raw_frame[sensor_id], errors="coerce").dropna()
        # API responses need not be chronological. Normalize known timezones
        # before ordering, and never turn an ambiguous timestamp into evidence.
        values.index = pd.DatetimeIndex([_utc(value) for value in values.index])
        values = values[values.index.notna()].sort_index(kind="stable")
        if values.empty:
            continue
        timestamp = pd.Timestamp(values.index[-1])
        observations.append({
            "sensor_id": str(sensor_id),
            "timestamp": timestamp.isoformat(),
            "value_cbar": float(values.iloc[-1]),
        })

    plot.tension_sensor_observations = observations
    return observations


def _utc(value) -> Optional[pd.Timestamp]:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        # WaziGate API timestamps are UTC.  Naive timestamps are not accepted
        # from arbitrary sources because their age would be ambiguous.
        return None
    return timestamp.tz_convert("UTC")


def evaluate_tension_sensor_safety(plot, current_value=None, now=None) -> dict:
    """Evaluate raw tension evidence for advisory and automatic use.

    One fresh, plausible observation can anchor a warning or
    advisory forecast.  Automatic irrigation remains stricter and requires at
    least two independent passing observations.  Keeping the two decisions
    separate avoids both unsafe actuation and the opposite error of making a
    single-sensor advisory deployment look completely unusable.
    """
    observations = list(getattr(plot, "tension_sensor_observations", []) or [])
    now_ts = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    stale_hours = SENSOR_STALE_HOURS
    # One sensor can support advisory modelling, but cannot distinguish slowly
    # changing soil from a plausible stuck-sensor failure. Automatic control
    # therefore requires at least two independent observations.
    minimum = AUTOMATIC_MINIMUM_SENSORS
    lower = TENSION_MIN_CBAR
    upper = TENSION_MAX_CBAR
    max_spread = MAX_SENSOR_SPREAD_CBAR

    valid, rejected = [], []
    if now_ts is None:
        return {
            "safe_for_advisory": False,
            "safe_for_automatic": False,
            "advisory_reasons": ["reference_time_missing_or_not_timezone_aware"],
            "reasons": ["reference_time_missing_or_not_timezone_aware"],
            "valid_sensor_count": 0,
            "required_sensor_count": minimum,
            "latest_age_hours": None,
            "spread_cbar": None,
            "plausibility_range_cbar": [lower, upper],
            "max_spread_cbar": max_spread,
            "valid_observations": [],
            "rejected_observations": [],
        }

    by_sensor = {}
    for item in observations:
        timestamp = _utc(item.get("timestamp"))
        try:
            value = float(item.get("value_cbar"))
        except (TypeError, ValueError):
            value = np.nan
        sensor_id = str(item.get("sensor_id") or "").strip()
        reasons = []
        if not sensor_id:
            reasons.append("sensor_identity_missing")
        age_hours = None
        if timestamp is None:
            reasons.append("timestamp_missing_or_not_timezone_aware")
        else:
            age_hours = (now_ts - timestamp).total_seconds() / 3600.0
            if age_hours < -(5.0 / 60.0):
                reasons.append("timestamp_is_in_the_future")
            elif age_hours > stale_hours:
                reasons.append("stale")
            age_hours = max(0.0, age_hours)
        if not np.isfinite(value) or value < lower or value > upper:
            reasons.append("outside_configured_plausibility_range")
        record = {
            "sensor_id": sensor_id,
            "timestamp": timestamp.isoformat() if timestamp is not None else None,
            "age_hours": age_hours,
            "value_cbar": value if np.isfinite(value) else None,
            "reasons": reasons,
        }
        by_sensor.setdefault(sensor_id, []).append(record)

    # Defend the gate independently of capture: restored/caller-supplied
    # evidence must also represent distinct physical sensor identities.
    # Identical repeats count once; conflicting repeats fail closed for that
    # sensor rather than allowing input order to choose the trusted reading.
    for records in by_sensor.values():
        record = records[0]
        if any(other != record for other in records[1:]):
            for other in records:
                other["reasons"].append("conflicting_duplicate_sensor_observations")
                rejected.append(other)
        else:
            (valid if not record["reasons"] else rejected).append(record)

    values = [item["value_cbar"] for item in valid]
    spread = max(values) - min(values) if len(values) > 1 else 0.0
    advisory_reasons = []
    if not valid:
        advisory_reasons.append(
            "no_fresh_plausible_sensor")
    if spread > max_spread:
        advisory_reasons.append("sensor_disagreement_exceeds_limit")
    try:
        current = float(current_value)
    except (TypeError, ValueError):
        current = np.nan
    if current_value is not None and (
            not np.isfinite(current) or current < lower or current > upper):
        advisory_reasons.append(
            "current_tension_outside_configured_plausibility_range")
    if current_value is not None and np.isfinite(current) and values:
        raw_median = float(np.median(values))
        if abs(current - raw_median) > max_spread:
            advisory_reasons.append(
                "model_current_tension_disagrees_with_raw_sensors")

    advisory_warnings = list(advisory_reasons)
    automatic_reasons = list(advisory_reasons)
    if len(valid) < minimum:
        automatic_reasons.append(
            "insufficient_fresh_plausible_sensors")

    latest_age = min(
        (item["age_hours"] for item in valid if item["age_hours"] is not None),
        default=None,
    )
    return {
        "safe_for_advisory": not advisory_reasons,
        "safe_for_automatic": not automatic_reasons,
        "advisory_reasons": advisory_reasons,
        "advisory_warnings": advisory_warnings,
        # ``reasons`` remains the automatic-control reason list for backward
        # compatibility with existing API consumers.
        "reasons": automatic_reasons,
        "valid_sensor_count": len(valid),
        "required_sensor_count": minimum,
        "latest_age_hours": latest_age,
        "spread_cbar": float(spread),
        "plausibility_range_cbar": [lower, upper],
        "max_spread_cbar": max_spread,
        "valid_observations": valid,
        "rejected_observations": rejected,
    }
