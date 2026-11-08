"""Boundary validation independent of storage, HTTP, and ML dependencies.

Topology uses GeoJSON straight segments in longitude/latitude space. Area uses
WGS84 geodesic edges. Only local, non-polar polygons without holes are supported.
"""
import math
from dataclasses import dataclass
from typing import Mapping

from geographiclib.geodesic import Geodesic
from shapely.geometry import Polygon
from shapely.validation import explain_validity


# Explicit initial product limits; not an agronomic claim about farm sizes.
MAX_AREA_M2 = 1_000_000_000  # 100,000 hectares
MAX_POSITIONS = 2000
MAX_SPAN_DEGREES = 1


class BoundaryValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ValidatedBoundary:
    positions: tuple
    area_m2: float

    def as_geojson(self):
        """Return a fresh JSON-ready value, without retaining caller metadata."""
        return {"type": "Polygon", "coordinates": [
            [list(position) for position in self.positions]]}


def _fail(code, message):
    raise BoundaryValidationError(code, message)


def validate_polygon(geometry):
    """Validate a decoded JSON geometry; never repair or auto-close a boundary."""
    if not isinstance(geometry, dict) or geometry.get('type') != 'Polygon':
        _fail('geometry_type', 'A GeoJSON Polygon geometry is required.')
    rings = geometry.get('coordinates')
    if not isinstance(rings, list) or len(rings) != 1:
        _fail('rings', 'Provide exactly one exterior ring; holes are not supported yet.')
    ring = rings[0]
    if not isinstance(ring, list) or not 4 <= len(ring) <= MAX_POSITIONS:
        _fail('positions', f'A closed ring needs 4 to {MAX_POSITIONS} positions.')
    positions = []
    for point in ring:
        if not isinstance(point, list) or len(point) != 2:
            _fail('coordinate', 'Each position must be [longitude, latitude].')
        for value in point:
            if type(value) not in (int, float):
                _fail('coordinate', 'Coordinates must be finite JSON numbers.')
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                _fail('coordinate', 'Coordinates must be finite JSON numbers.')
        lon, lat = point
        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            _fail('coordinate_range', 'Longitude/latitude are outside their valid ranges.')
        positions.append((float(lon), float(lat)))
    if positions[0] != positions[-1]:
        _fail('closure', 'Close the ring by repeating its first position.')
    if len(set(positions[:-1])) < 3:
        _fail('vertices', 'Provide at least three distinct vertices.')
    if len(set(positions[:-1])) != len(positions) - 1:
        _fail('vertices', 'Only the closing vertex may be repeated.')
    lons, lats = zip(*positions)
    if (max(lons) - min(lons) > MAX_SPAN_DEGREES or
            max(lats) - min(lats) > MAX_SPAN_DEGREES or
            min(lats) <= -89 or max(lats) >= 89):
        _fail('extent', 'Use a local boundary spanning at most one degree, away from the poles and date line.')
    polygon = Polygon(positions)
    if not polygon.is_valid or polygon.area <= 0:
        _fail('topology', f'Invalid polygon: {explain_validity(polygon)}')
    accumulator = Geodesic.WGS84.Polygon()
    for lon, lat in positions[:-1]:
        accumulator.AddPoint(lat, lon)
    area = abs(accumulator.Compute()[2])
    if not math.isfinite(area) or not 0 < area <= MAX_AREA_M2:
        _fail('area', f'Boundary area must be positive and at most {MAX_AREA_M2} square metres.')
    return ValidatedBoundary(tuple(positions), area)


def validate_farm_layout(farm_geometry, plot_geometries: Mapping):
    """Validate an entire proposed single-farm layout, including farm edits.

    Mapping keys are stable plot IDs. This validates relationships, not database
    ownership: persistence must resolve those IDs and call this within its write
    transaction. Empty mappings are allowed for validation of a farm draft only.
    """
    if not isinstance(plot_geometries, Mapping):
        _fail('plots', 'Plot geometries must be keyed by stable plot ID.')
    farm = validate_polygon(farm_geometry)
    farm_shape = Polygon(farm.positions)
    plots = {}
    shapes = []
    for plot_id, geometry in plot_geometries.items():
        if not isinstance(plot_id, str) or not plot_id.strip():
            _fail('plot_id', 'A nonempty stable plot ID is required.')
        boundary = validate_polygon(geometry)
        shape = Polygon(boundary.positions)
        if not farm_shape.covers(shape):
            _fail('containment', f'Plot {plot_id} must be entirely inside the farm boundary.')
        for other_id, other_shape in shapes:
            # Interior/interior intersection rejects equality and containment too;
            # shared boundary edges/corners remain permitted, with no buffer fudge.
            if shape.relate_pattern(other_shape, 'T********'):
                _fail('overlap', f'Plots {plot_id} and {other_id} have overlapping interiors.')
        shapes.append((plot_id, shape))
        plots[plot_id] = boundary
    return farm, plots
