"""Shared timestamp uniqueness contract for normalized weather observations."""

import pandas as pd
import numpy as np


def complete_daily_et0(frame: pd.DataFrame, timezone_name="UTC") -> pd.Series:
    """Daily ET0 totals; an absent/invalid hour makes the local day unknown.

    Normalized provider ET0 is an hourly depth in mm, not a daily rate.
    Calendar-day boundaries also preserve 23/25-hour daylight-saving days.
    """
    if frame.empty or "Et0_evapotranspiration" not in frame:
        return pd.Series(dtype=float)
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = deduplicate_weather_frame(frame.loc[~frame.index.isna()].sort_index())
    frame.index = frame.index.tz_convert(timezone_name)
    values = pd.to_numeric(frame["Et0_evapotranspiration"], errors="coerce")
    totals = {}
    for day, group in values.groupby(values.index.normalize()):
        expected = pd.date_range(day, day + pd.DateOffset(days=1),
                                 freq="h", inclusive="left")
        hourly = group.reindex(expected)
        valid = (len(group) == len(expected)
                 and np.isfinite(hourly.to_numpy(dtype=float)).all()
                 and (hourly >= 0).all())
        totals[day] = float(hourly.sum()) if valid else np.nan
    return pd.Series(totals, dtype=float).sort_index()


def deduplicate_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Count identical repeats once; reject conflicting rows at the same time.

    Call after timestamp normalization. Without revision or source-priority
    metadata, row order cannot establish which conflicting value is correct.
    Missing versus present values are also a conflict, not implicit authority.
    """
    duplicates = frame.loc[frame.index.duplicated(keep=False)]
    if not duplicates.empty:
        distinct = duplicates.groupby(level=0, sort=False).nunique(dropna=False)
        if distinct.gt(1).any().any():
            raise ValueError("Conflicting weather observations at the same timestamp")
    return frame.loc[~frame.index.duplicated(keep="first")].copy()
