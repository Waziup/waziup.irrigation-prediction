"""Shared validation for soil-curve calibration, independent of ML imports."""
import math
from datetime import date


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
    if points[0][0] > saturation or points[-1][0] < wp:
        raise ValueError('Curve must cover saturation through wilting tension')
    if wp != 1500 and not (source in {'laboratory', 'field'} and
                           str(metadata.get('reference', '')).strip() and metadata.get('date')):
        raise ValueError('Nonstandard PWP requires a measurement source, reference and date; use 1500 cbar for estimates')
    return points
