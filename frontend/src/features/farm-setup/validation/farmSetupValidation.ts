import { area } from '@turf/area';
import { difference } from '@turf/difference';
import { intersect } from '@turf/intersect';
import { kinks } from '@turf/kinks';
import type {
  GeoJsonPosition,
  PolygonGeometry,
} from '../../../domain/geometry';
import type { JsonObject } from '../../../domain/api';
import type { SetupDraft } from '../types';

export function geometryJson(
  geometry: PolygonGeometry,
): JsonObject & { type: 'Polygon'; coordinates: number[][][] } {
  return {
    type: 'Polygon',
    coordinates: geometry.coordinates.map((ring) =>
      ring.map((p) => [p[0], p[1]]),
    ),
  };
}
export function polygonFromVertices(
  vertices: readonly GeoJsonPosition[],
): PolygonGeometry {
  const [a, b, c, ...rest] = vertices;
  if (!a || !b || !c) throw new Error('Add at least three distinct corners.');
  const positions = [a, b, c, ...rest];
  if (positions.length > 1999) throw new Error('Use at most 1,999 corners.');
  if (
    positions.some(
      (p) =>
        p.length !== 2 ||
        !p.every(Number.isFinite) ||
        Math.abs(p[0]) > 180 ||
        Math.abs(p[1]) >= 89,
    )
  )
    throw new Error(
      'Use valid longitude and latitude coordinates away from the poles.',
    );
  if (
    new Set(positions.map((p) => `${p[0]},${p[1]}`)).size !== positions.length
  )
    throw new Error('Each corner must be distinct; closing is automatic.');
  const lons = positions.map((p) => p[0]),
    lats = positions.map((p) => p[1]);
  if (
    Math.max(...lons) - Math.min(...lons) > 1 ||
    Math.max(...lats) - Math.min(...lats) > 1
  )
    throw new Error('Keep the boundary local, spanning at most one degree.');
  const [d, ...tail] = rest;
  const geometry: PolygonGeometry = {
    type: 'Polygon',
    coordinates: [d ? [a, b, c, d, ...tail, a] : [a, b, c, a]],
  };
  const json = geometryJson(geometry);
  if (kinks(json).features.length)
    throw new Error('Boundary edges must not cross or touch themselves.');
  const measured = area(json);
  // Spherical preview differs slightly from the authoritative WGS84 area.
  if (!(measured > 0 && measured <= 1_010_000_000))
    throw new Error(
      'Draw a positive area no larger than approximately 100,000 hectares.',
    );
  return geometry;
}
export function parseGeometry(value: unknown): PolygonGeometry {
  if (
    !value ||
    typeof value !== 'object' ||
    !('type' in value) ||
    value.type !== 'Polygon' ||
    !('coordinates' in value)
  )
    throw new Error('A polygon boundary is required.');
  const coordinates = value.coordinates;
  if (
    !Array.isArray(coordinates) ||
    coordinates.length !== 1 ||
    !Array.isArray(coordinates[0])
  )
    throw new Error('Use one exterior boundary without holes.');
  const ring: GeoJsonPosition[] = coordinates[0].map((p: unknown) => {
    if (
      !Array.isArray(p) ||
      p.length !== 2 ||
      typeof p[0] !== 'number' ||
      typeof p[1] !== 'number'
    )
      throw new Error('Coordinates must be [longitude, latitude].');
    return [p[0], p[1]];
  });
  const first = ring[0],
    last = ring.at(-1);
  if (
    ring.length < 4 ||
    !first ||
    !last ||
    first[0] !== last[0] ||
    first[1] !== last[1]
  )
    throw new Error('The boundary ring must be closed.');
  return polygonFromVertices(ring.slice(0, -1));
}
function pair(a: PolygonGeometry, b: PolygonGeometry) {
  return {
    type: 'FeatureCollection' as const,
    features: [a, b].map((g) => ({
      type: 'Feature' as const,
      properties: {},
      geometry: geometryJson(g),
    })),
  };
}
/** Exact outside area, including an escaping edge whose corners are inside. */
export function plotOutsideFarm(plot: PolygonGeometry, farm: PolygonGeometry) {
  return difference(pair(plot, farm));
}
export function validatePlotBoundary(
  plot: PolygonGeometry,
  farm: PolygonGeometry,
  others: readonly PolygonGeometry[] = [],
) {
  if (plotOutsideFarm(plot, farm))
    throw new Error(
      'Part of this plot is outside the farm boundary. Move it inside; shared edges are allowed.',
    );
  for (const other of others)
    if (intersect(pair(plot, other)))
      throw new Error(
        'This plot overlaps an existing plot. Shared edges are allowed.',
      );
}
export function previewArea(geometry: PolygonGeometry): number {
  return area(geometryJson(geometry));
}
export function areaLabel(value: number): string {
  return value >= 10000
    ? `${(value / 10000).toLocaleString(undefined, { maximumFractionDigits: 2 })} ha`
    : `${value.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²`;
}
export function validateDetails(draft: SetupDraft) {
  if (!draft.name.trim() || draft.name.trim().length > 160)
    throw new Error('Enter a farm name up to 160 characters.');
  if (draft.owner.trim().length > 200)
    throw new Error('Keep the owner name within 200 characters.');
  try {
    new Intl.DateTimeFormat('en', { timeZone: draft.timezone }).format();
  } catch {
    throw new Error('Enter a valid timezone, such as Africa/Nairobi.');
  }
}
export function centerPosition(
  latitude: string,
  longitude: string,
): GeoJsonPosition {
  const lat = Number(latitude),
    lon = Number(longitude);
  if (
    !latitude.trim() ||
    !longitude.trim() ||
    !Number.isFinite(lat) ||
    !Number.isFinite(lon) ||
    Math.abs(lat) >= 89 ||
    Math.abs(lon) > 180
  )
    throw new Error('Enter valid latitude and longitude to locate the farm.');
  return [lon, lat];
}

/** Snap map clicks to a nearby saved corner/edge; coordinates entered manually stay exact. */
export function snapToFarmBoundary(
  position: GeoJsonPosition,
  farm: PolygonGeometry,
  project: (position: GeoJsonPosition) => { x: number; y: number },
  tolerance = 12,
): GeoJsonPosition {
  const point = project(position),
    ring = farm.coordinates[0];
  let closest = position,
    distance = tolerance * tolerance;
  // Prefer the nearest saved corner, keeping adjoining boundaries identical.
  let closestCorner: GeoJsonPosition | null = null;
  for (const corner of ring.slice(0, -1)) {
    const pixel = project(corner);
    const separation = (point.x - pixel.x) ** 2 + (point.y - pixel.y) ** 2;
    if (separation <= distance) {
      distance = separation;
      closestCorner = corner;
    }
  }
  if (closestCorner) return closestCorner;
  for (let i = 1; i < ring.length; i++) {
    const start = ring[i - 1]!,
      end = ring[i]!,
      a = project(start),
      b = project(end);
    const dx = b.x - a.x,
      dy = b.y - a.y,
      length = dx * dx + dy * dy;
    if (!length) continue;
    const t = Math.max(
      0,
      Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / length),
    );
    const separation =
      (point.x - a.x - t * dx) ** 2 + (point.y - a.y - t * dy) ** 2;
    if (separation <= distance) {
      distance = separation;
      closest = [
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1]),
      ];
    }
  }
  return closest;
}

export function boundaryCoordinateBounds(boundary: PolygonGeometry) {
  const ring = boundary.coordinates[0];
  return {
    minLongitude: Math.min(...ring.map((point) => point[0])),
    maxLongitude: Math.max(...ring.map((point) => point[0])),
    minLatitude: Math.min(...ring.map((point) => point[1])),
    maxLatitude: Math.max(...ring.map((point) => point[1])),
  };
}

/** Boundary-inclusive point check, including concave farms and shared corners. */
export function pointInBoundary(
  point: GeoJsonPosition,
  boundary: PolygonGeometry,
): boolean {
  if (!point.every(Number.isFinite)) return false;
  const [x, y] = point,
    ring = boundary.coordinates[0];
  let inside = false;
  for (let index = 1; index < ring.length; index++) {
    const [ax, ay] = ring[index - 1]!,
      [bx, by] = ring[index]!;
    const left = (x - ax) * (by - ay),
      right = (y - ay) * (bx - ax);
    const roundoff = Number.EPSILON * (Math.abs(left) + Math.abs(right)) * 8;
    if (
      Math.abs(left - right) <= roundoff &&
      x >= Math.min(ax, bx) &&
      x <= Math.max(ax, bx) &&
      y >= Math.min(ay, by) &&
      y <= Math.max(ay, by)
    )
      return true;
    if (ay > y !== by > y && x < ax + ((y - ay) * (bx - ax)) / (by - ay))
      inside = !inside;
  }
  return inside;
}
