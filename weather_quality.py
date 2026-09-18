"""Shared timestamp uniqueness contract for normalized weather observations."""

import pandas as pd


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
