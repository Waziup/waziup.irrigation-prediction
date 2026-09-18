"""Stable JSON normalization shared by backend API result contracts."""

from datetime import datetime

import numpy as np
import pandas as pd


def json_safe(value):
    """Normalize pandas/NumPy values into JSON-compatible primitives."""
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
