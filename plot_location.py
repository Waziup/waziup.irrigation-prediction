"""Resolve a plot's satellite reference point from GPS or its saved boundary."""
import math
from shapely.geometry import Polygon
from geometry_validation import BoundaryValidationError, validate_polygon


def plot_reference_location(plot):
    """Return (latitude, longitude); never substitute farm GPS for plot GPS."""
    gps = getattr(plot, 'gps_info', None)
    if isinstance(gps, dict):
        try:
            latitude = float(gps.get('latitude', gps.get('lattitude')))
            longitude = float(gps.get('longitude'))
            if (math.isfinite(latitude) and math.isfinite(longitude)
                    and -90 <= latitude <= 90 and -180 <= longitude <= 180):
                return latitude, longitude
        except (TypeError, ValueError, OverflowError):
            pass
    geometry = getattr(plot, 'geometry', None)
    if geometry is not None:
        try:
            boundary = validate_polygon(geometry)
            # A centroid can lie outside a concave plot; this point stays inside.
            point = Polygon(boundary.positions).representative_point()
            return float(point.y), float(point.x)
        except BoundaryValidationError:
            pass
    return None, None
