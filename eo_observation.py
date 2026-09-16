
from __future__ import annotations

import numpy as np
import pandas as pd


def _analyse_ndvi_history(history: pd.DataFrame) -> dict:
    """Describe NDVI direction and a conservative greening-onset interval."""
    if not isinstance(history, pd.DataFrame) or history.empty:
        return {
            "available": False,
            "reason": "No EO history was returned for this plot.",
            "valid_observations": 0,
            "greening": {"detected": False, "status": "insufficient_data"},
        }

    frame = history.copy()
    if "timestamp" not in frame or "sat_ndvi" not in frame:
        return {
            "available": False,
            "reason": "EO history did not contain timestamped NDVI values.",
            "valid_observations": 0,
            "greening": {"detected": False, "status": "insufficient_data"},
        }

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce")
    frame["sat_ndvi"] = pd.to_numeric(frame["sat_ndvi"], errors="coerce")
    frame = frame[
        frame["timestamp"].notna()
        & frame["sat_ndvi"].between(-1.0, 1.0, inclusive="both")
    ].sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    if frame.empty:
        return {
            "available": False,
            "reason": "EO scenes were present but none contained valid NDVI pixels.",
            "valid_observations": 0,
            "greening": {"detected": False, "status": "insufficient_data"},
        }

    values = frame["sat_ndvi"].to_numpy(dtype=float)
    times = pd.DatetimeIndex(frame["timestamp"])
    elapsed_days = (times - times[0]).total_seconds() / 86400.0
    slope = (
        float(np.polyfit(elapsed_days, values, 1)[0])
        if len(values) >= 2 and float(np.ptp(elapsed_days)) > 0
        else None
    )
    direction = "unknown"
    if slope is not None:
        direction = "increasing" if slope > 0.003 else (
            "decreasing" if slope < -0.003 else "stable")

    greening = {
        "detected": False,
        "status": "insufficient_data" if len(values) < 4 else "not_detected",
        "estimated_start": None,
        "estimated_end": None,
        "confidence": None,
        "label": "Vegetation greening was not established from the available scenes.",
    }
    if len(values) >= 4:
        baseline_count = max(2, min(len(values) // 3, 4))
        baseline = values[:baseline_count]
        baseline_median = float(np.median(baseline))
        threshold = max(0.08, 2.0 * float(np.std(baseline)))
        for index in range(baseline_count, len(values) - 1):
            first_rise = values[index] - baseline_median
            sustained_rise = values[index + 1] - baseline_median
            if first_rise >= threshold and sustained_rise >= threshold * 0.75:
                previous = times[index - 1]
                detected = times[index]
                confidence = "high" if len(values) >= 6 else "moderate"
                greening = {
                    "detected": True,
                    "status": "estimated_unconfirmed",
                    "estimated_start": previous.date().isoformat(),
                    "estimated_end": detected.date().isoformat(),
                    "confidence": confidence,
                    "label": (
                        "A sustained vegetation greening signal was detected. "
                        "This is not yet confirmed as crop emergence."
                    ),
                }
                break

    latest = frame.iloc[-1]
    records = [
        {
            "timestamp": row.timestamp.isoformat(),
            "ndvi": round(float(row.sat_ndvi), 4),
            "source": getattr(row, "source", None),
        }
        for row in frame.itertuples(index=False)
    ]
    return {
        "available": True,
        "interpretation": "vegetation_observation_not_crop_phenology",
        "valid_observations": int(len(frame)),
        "first_observation": times[0].isoformat(),
        "latest_observation": times[-1].isoformat(),
        "latest_ndvi": round(float(latest["sat_ndvi"]), 4),
        "trend": {
            "direction": direction,
            "slope_ndvi_per_day": round(slope, 6) if slope is not None else None,
        },
        "greening": greening,
        "observations": records,
    }


def analyse_vegetation_history(history: pd.DataFrame) -> dict:
    """Expose both index histories without claiming NDRE is crop emergence."""
    ndvi = _analyse_ndvi_history(history)
    red_edge = history.drop(columns=["sat_ndvi"], errors="ignore").rename(
        columns={"sat_ndre": "sat_ndvi"}) if isinstance(history, pd.DataFrame) else history
    ndre = _analyse_ndvi_history(red_edge)
    if ndre.get("available"):
        ndre["latest_ndre"] = ndre.pop("latest_ndvi")
        ndre["trend"]["slope_ndre_per_day"] = ndre["trend"].pop("slope_ndvi_per_day")
        for observation in ndre["observations"]:
            observation["ndre"] = observation.pop("ndvi")
    elif "reason" in ndre:
        ndre["reason"] = ndre["reason"].replace("NDVI", "NDRE")
    # The existing greening detector's amplitude criteria were for NDVI;
    # applying them to the narrower NDRE scale would invent a calibration.
    ndre["greening"] = {"detected": False, "status": "not_assessed"}
    result = dict(ndvi if ndvi.get("available") else ndre)
    result["indices"] = {"ndvi": ndvi, "ndre": ndre}
    result["latest_ndvi"] = ndvi.get("latest_ndvi")
    result["latest_ndre"] = ndre.get("latest_ndre")
    result["index_source"] = "+".join(
        name for name, item in result["indices"].items() if item.get("available")) or "none"
    return result
