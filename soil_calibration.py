"""Shared validation for soil-curve calibration, independent of ML imports."""
import math
from datetime import date
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def original_soil_presets():
    return json.loads(Path(__file__).with_name('soil_presets.json').read_text())


def is_original_soil_preset(points, fc, wp, saturation, metadata):
    if metadata.get('source') != 'estimate' or metadata.get('reference') != 'WaziFarm original soil presets':
        return False
    preset = original_soil_presets().get(str(metadata.get('preset', '')))
    return bool(preset and fc == preset['FieldCapacity'][0]
                and wp == preset['PermanentWiltingPoint']
                and saturation == preset['Saturation']
                and points == list(zip(preset['soilTension'], preset['waterRetention'])))



def validate_soil_calibration(curve, fc, wp, saturation=0, metadata=None):
    metadata = metadata or {}
    source = metadata.get('source', 'estimate')
    if source not in {'estimate', 'manual', 'laboratory', 'field'}:
        raise ValueError('Unknown soil calibration source')
    if metadata.get('date'):
        date.fromisoformat(str(metadata['date']))
    if metadata.get('sample_depth_cm') not in (None, ''):
        depth = float(metadata['sample_depth_cm'])
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError('Sample depth must be positive')
    points = []
    for row in curve:
        try:
            t, v = ((row.get('Soil tension', row.get('tension_cbar')),
                     row.get('VWC', row.get('vwc'))) if isinstance(row, dict) else row)
            t, v = float(t), float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError('Each curve row needs cbar and VWC') from exc
        if not math.isfinite(t) or t < 0 or not math.isfinite(v) or not 0 < v < 1:
            raise ValueError('Use finite cbar and VWC fractions between 0 and 1')
        points.append((t, v))
    points.sort()
    if len(points) < 3 or any(b[0] <= a[0] or b[1] >= a[1]
                              for a, b in zip(points, points[1:])):
        raise ValueError('Curve needs at least three points: increasing cbar, decreasing VWC')
    fc, wp, saturation = float(fc), float(wp), float(saturation)
    if not all(math.isfinite(v) for v in (fc, wp, saturation)) or not 0 <= saturation < fc < wp:
        raise ValueError('Require saturation < field capacity < wilting tension')
    original = is_original_soil_preset(points, fc, wp, saturation, metadata)
    # The original table starts at field capacity; only those exact presets
    # retain the original saturation endpoint convention.
    if not original and (points[0][0] > saturation or points[-1][0] < wp):
        raise ValueError('Curve must cover saturation through wilting tension')
    if wp != 1500 and source not in {'laboratory', 'field'} and not original:
        raise ValueError('Nonstandard PWP requires a field or laboratory source; use 1500 cbar for estimates')
    return points
