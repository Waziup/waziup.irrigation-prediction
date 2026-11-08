/**
 * Two-dimensional GeoJSON position in WGS84 decimal degrees.
 */
export type GeoJsonPosition = readonly [longitude: number, latitude: number];

/**
 * A ring has at least four positions, including the repeated closing position.
 * Closure, three distinct vertices, finite coordinates and topology require
 * runtime validation;
 */
export type LinearRing = readonly [
  GeoJsonPosition,
  GeoJsonPosition,
  GeoJsonPosition,
  GeoJsonPosition,
  ...GeoJsonPosition[],
];

/** First ring is the exterior; any subsequent rings are interior holes. */
export type PolygonCoordinates = readonly [
  exterior: LinearRing,
  ...holes: LinearRing[],
];

/**
 * Required boundary shape for saved farms and plots. No Point, MultiPolygon,
 * GeometryCollection or null alternative is part of this contract.
 * Accept external JSON as unknown and validate it before using this type.
 */
export interface PolygonGeometry {
  readonly type: 'Polygon';
  readonly coordinates: PolygonCoordinates;
}
