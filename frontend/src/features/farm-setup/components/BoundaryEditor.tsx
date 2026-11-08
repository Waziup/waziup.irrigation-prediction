import { MapTiles } from '../../farm-map/MapTiles';
import { useEffect, useId, useMemo, useState } from 'react';
import {
  MapContainer,
  GeoJSON,
  Polygon,
  Polyline,
  CircleMarker,
  Marker,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet';
import { divIcon } from 'leaflet';
import type { LatLngTuple } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type {
  GeoJsonPosition,
  PolygonGeometry,
} from '../../../domain/geometry';
import {
  areaLabel,
  polygonFromVertices,
  previewArea,
  validatePlotBoundary,
  plotOutsideFarm,
  snapToFarmBoundary,
  pointInBoundary,
  boundaryCoordinateBounds,
} from '../validation/farmSetupValidation';

const cornerIcon = divIcon({
  className: '',
  html: '<span style="display:block;width:18px;height:18px;border:3px solid #345f3d;border-radius:50%;background:white;box-shadow:0 1px 5px #0003"></span>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});
const button =
  'min-h-11 rounded-lg border border-[#cbd8ca] bg-white px-4 py-2 text-sm font-semibold text-[#2d5a27] hover:bg-[#edf5e9] disabled:opacity-40';
function latLng(p: GeoJsonPosition): LatLngTuple {
  return [p[1], p[0]];
}
function View({
  center,
  boundary,
}: {
  center: GeoJsonPosition;
  boundary: PolygonGeometry | null;
}) {
  const map = useMap();
  const [longitude, latitude] = center;
  useEffect(() => {
    const fit = () => {
      map.invalidateSize({ pan: false });
      if (boundary)
        map.fitBounds(boundary.coordinates[0].map(latLng), {
          padding: [28, 28],
          maxZoom: 19,
          animate: false,
        });
      else map.setView([latitude, longitude], 17, { animate: false });
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map, boundary, longitude, latitude]);
  return null;
}
function snapDistance() {
  return window.matchMedia('(pointer: coarse)').matches ? 20 : 12;
}
function Clicks({
  drawing,
  add,
  farm,
}: {
  drawing: boolean;
  add: (p: GeoJsonPosition) => void;
  farm: PolygonGeometry | null;
}) {
  const map = useMapEvents({
    click: (e) => {
      if (!drawing) return;
      const point: GeoJsonPosition = [e.latlng.lng, e.latlng.lat];
      add(
        farm
          ? snapToFarmBoundary(
              point,
              farm,
              (p) => map.latLngToContainerPoint(latLng(p)),
              snapDistance(),
            )
          : point,
      );
    },
  });
  return null;
}
function EditableCorner({
  position,
  index,
  farm,
  select,
  move,
}: {
  position: GeoJsonPosition;
  index: number;
  farm: PolygonGeometry | null;
  select: () => void;
  move: (point: GeoJsonPosition) => boolean;
}) {
  const map = useMap();
  return (
    <Marker
      position={latLng(position)}
      icon={cornerIcon}
      draggable
      title={`Move corner ${index + 1}`}
      eventHandlers={{
        click: select,
        dragstart: select,
        dragend: (event) => {
          const location = event.target.getLatLng();
          const point: GeoJsonPosition = [location.lng, location.lat];
          const snapped = farm
            ? snapToFarmBoundary(
                point,
                farm,
                (p) => map.latLngToContainerPoint(latLng(p)),
                snapDistance(),
              )
            : point;
          event.target.setLatLng(latLng(move(snapped) ? snapped : position));
        },
      }}
    />
  );
}
function CoordinatePreview({ position }: { position: GeoJsonPosition | null }) {
  const map = useMap();
  const longitude = position?.[0],
    latitude = position?.[1];
  useEffect(() => {
    if (longitude !== undefined && latitude !== undefined)
      map.panInside([latitude, longitude], {
        padding: [35, 35],
        animate: false,
      });
  }, [map, longitude, latitude]);
  return position ? (
    <CircleMarker
      interactive={false}
      center={latLng(position)}
      radius={7}
      pathOptions={{ color: '#b88000', fillColor: '#ffe19a', fillOpacity: 1 }}
    >
      <Tooltip direction="top" permanent>
        Corner preview
      </Tooltip>
    </CircleMarker>
  ) : null;
}
export default function BoundaryEditor({
  label,
  value,
  center,
  farm = null,
  others = [],
  onChange,
  onDrawingChange,
  startEditing = false,
  onFinished,
}: {
  onFinished?: () => void;
  startEditing?: boolean;
  label: string;
  value: PolygonGeometry | null;
  center: GeoJsonPosition;
  farm?: PolygonGeometry | null;
  others?: readonly PolygonGeometry[];
  onChange: (geometry: PolygonGeometry) => void;
  onDrawingChange: (drawing: boolean) => void;
}) {
  const [drawing, setDrawing] = useState(startEditing && !!value);
  const [editingPoints, setEditingPoints] = useState(startEditing && !!value);
  const [history, setHistory] = useState<(readonly GeoJsonPosition[])[]>([]);
  const [selectedCorner, setSelectedCorner] = useState<number | null>(null);
  useEffect(() => {
    onDrawingChange(drawing);
  }, [drawing, onDrawingChange]);

  const [vertices, setVertices] = useState<readonly GeoJsonPosition[]>(
    value?.coordinates[0].slice(0, -1) ?? [],
  );
  const [longitude, setLongitude] = useState(''),
    [latitude, setLatitude] = useState('');
  const [imagery, setImagery] = useState<'satellite' | 'street' | 'boundaries'>(
    'satellite',
  );
  const [tileError, setTileError] = useState(false);
  const [tilesLoaded, setTilesLoaded] = useState(false);
  useEffect(() => {
    if (imagery === 'boundaries' || tilesLoaded || tileError) return;
    const timeout = window.setTimeout(() => setTileError(true), 12000);
    return () => window.clearTimeout(timeout);
  }, [imagery, tilesLoaded, tileError]);
  const [error, setError] = useState('');
  const hintId = useId();
  const bounds = farm ? boundaryCoordinateBounds(farm) : null;
  const lon = Number(longitude),
    lat = Number(latitude);
  const candidate: GeoJsonPosition | null =
    longitude.trim() &&
    latitude.trim() &&
    Number.isFinite(lon) &&
    Number.isFinite(lat) &&
    Math.abs(lon) <= 180 &&
    Math.abs(lat) < 89
      ? [lon, lat]
      : null;
  const outsideFarm = !!(
    farm &&
    candidate &&
    !pointInBoundary(candidate, farm)
  );
  const preview =
    drawing &&
    (!editingPoints || selectedCorner !== null) &&
    candidate &&
    !outsideFarm
      ? candidate
      : null;
  const formatCoordinate = (value: number) =>
    value.toLocaleString(undefined, { maximumFractionDigits: 8 });
  const outside = useMemo(() => {
    if (!farm || vertices.length < 3) return null;
    try {
      return plotOutsideFarm(polygonFromVertices(vertices), farm);
    } catch {
      // Invalid/incomplete rings retain their points for editing; the finish
      // action reports their topology error without inventing an outside area.
      return null;
    }
  }, [farm, vertices]);
  function allowedPoint(point: GeoJsonPosition) {
    if (
      !point.every(Number.isFinite) ||
      Math.abs(point[0]) > 180 ||
      Math.abs(point[1]) >= 89
    ) {
      setError('Enter valid longitude and latitude.');
      return false;
    }
    if (farm && !pointInBoundary(point, farm)) {
      setError(
        'This corner is outside the farm boundary. Choose a point inside or on its edge.',
      );
      return false;
    }
    return true;
  }
  function changePoints(next: readonly GeoJsonPosition[]) {
    setHistory([...history, vertices]);
    setVertices(next);
    setError('');
  }
  function moveCorner(index: number, point: GeoJsonPosition) {
    if (!allowedPoint(point)) return false;
    changePoints(vertices.map((old, i) => (i === index ? point : old)));
    if (selectedCorner === index) {
      setLongitude(String(point[0]));
      setLatitude(String(point[1]));
    }
    return true;
  }
  function selectCorner(index: number) {
    setSelectedCorner(index);
    setLongitude(String(vertices[index]![0]));
    setLatitude(String(vertices[index]![1]));
  }
  function insertCorner(after: number, point: GeoJsonPosition) {
    // A midpoint on an escaping edge may itself be outside. Keep it as an
    // editable draft point so the farmer can drag it back in; never save it.
    if (vertices.length >= 1999) return;
    changePoints([
      ...vertices.slice(0, after + 1),
      point,
      ...vertices.slice(after + 1),
    ]);
    setSelectedCorner(after + 1);
    setLongitude(String(point[0]));
    setLatitude(String(point[1]));
  }
  function mapCorner(point: GeoJsonPosition) {
    if (!editingPoints) {
      add(point);
    } else if (selectedCorner !== null) {
      moveCorner(selectedCorner, point);
    } else {
      setError(
        'Select a corner to move it, or tap a small dot between corners to add one.',
      );
    }
  }
  function add(p: GeoJsonPosition) {
    if (!allowedPoint(p)) return false;
    setError('');
    if (vertices.length >= 1999) {
      setError('Use at most 1,999 corners.');
      return false;
    }
    setVertices((current) => [...current, p]);
    return true;
  }
  function finish() {
    try {
      const geometry = polygonFromVertices(vertices);
      if (farm) validatePlotBoundary(geometry, farm, others);
      onChange(geometry);
      setDrawing(false);
      onDrawingChange(false);
      setError('');
      onFinished?.();
    } catch (cause) {
      setEditingPoints(true);
      setError(
        outside
          ? ''
          : cause instanceof Error
            ? cause.message
            : 'Check the boundary.',
      );
    }
  }
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-2">
          {!drawing ? (
            <>
              {value && (
                <button
                  type="button"
                  className={button}
                  onClick={() => {
                    setVertices(value.coordinates[0].slice(0, -1));
                    setHistory([]);
                    setSelectedCorner(null);
                    setEditingPoints(true);
                    setDrawing(true);
                    onDrawingChange(true);
                    setError('');
                  }}
                >
                  Edit corner points
                </button>
              )}
              <button
                type="button"
                className={button}
                onClick={() => {
                  setVertices([]);
                  setEditingPoints(false);
                  setHistory([]);
                  setDrawing(true);
                  onDrawingChange(true);
                  setError('');
                }}
              >
                {value ? 'Redraw boundary' : 'Start drawing'}
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className={button}
                disabled={vertices.length < 3}
                onClick={finish}
              >
                {editingPoints ? 'Apply changes' : 'Finish boundary'}
              </button>
              {!editingPoints && vertices.length >= 3 && (
                <button
                  type="button"
                  className={button}
                  onClick={() => setEditingPoints(true)}
                >
                  Adjust corners
                </button>
              )}
              {editingPoints ? (
                <button
                  type="button"
                  className={button}
                  disabled={!history.length}
                  onClick={() => {
                    const previous = history.at(-1);
                    if (previous) {
                      setVertices(previous);
                      setHistory(history.slice(0, -1));
                      setSelectedCorner(null);
                      setError('');
                    }
                  }}
                >
                  Undo change
                </button>
              ) : (
                <button
                  type="button"
                  className={button}
                  disabled={!vertices.length}
                  onClick={() => setVertices(vertices.slice(0, -1))}
                >
                  Undo corner
                </button>
              )}
              <button
                type="button"
                className={button}
                onClick={() => {
                  setVertices(value?.coordinates[0].slice(0, -1) ?? []);
                  setDrawing(false);
                  onDrawingChange(false);
                  setError('');
                }}
              >
                {editingPoints ? 'Cancel changes' : 'Cancel drawing'}
              </button>
              {editingPoints && (
                <button
                  type="button"
                  className={button}
                  onClick={() => {
                    setVertices([]);
                    setEditingPoints(false);
                    setHistory([]);
                    setSelectedCorner(null);
                    setError('');
                  }}
                >
                  Redraw boundary
                </button>
              )}
            </>
          )}
        </div>
        <div
          role="group"
          aria-label="Boundary map view"
          className="flex flex-wrap gap-1 rounded-xl border border-[#dce5d7] bg-white p-1"
        >
          {(
            [
              ['satellite', 'Satellite'],
              ['street', 'Street map'],
              ['boundaries', 'Boundaries only'],
            ] as const
          ).map(([mode, name]) => (
            <button
              key={mode}
              type="button"
              aria-pressed={imagery === mode}
              className="min-h-10 rounded-lg px-3 text-xs font-semibold text-[#52675b] aria-pressed:bg-[#315f48] aria-pressed:text-white"
              onClick={() => {
                if (mode === imagery && !tileError) return;
                setImagery(mode);
                setTileError(false);
                setTilesLoaded(false);
              }}
            >
              {name}
            </button>
          ))}
        </div>
      </div>
      <p className="text-sm text-[#53645a]" role="status">
        {drawing
          ? editingPoints
            ? 'Drag a corner, or select it and tap its new position. Tap a small dot between corners to add a point along a bend.'
            : `${vertices.length} corners · Add points along every bend. Shared farm edges are allowed. Choose Finish boundary when done.`
          : value
            ? `Boundary ready · Approximately ${areaLabel(previewArea(value))}`
            : 'Position the map, then choose Start drawing.'}
      </p>
      {tileError && (
        <p className="text-xs text-[#806536]" role="status">
          Map imagery unavailable.{' '}
          {farm || value ? 'Saved boundaries are still shown. ' : ''}Use known
          corner coordinates, or try another map view.
        </p>
      )}
      {imagery === 'boundaries' && (
        <p role="status" className="text-xs text-[#78896e]">
          Boundaries only: no aerial or street imagery. Use saved outlines or
          known corner coordinates.
        </p>
      )}
      <div
        className="relative z-0 overflow-hidden rounded-xl border border-[#d5dfd0]"
        aria-label={`${label} map`}
      >
        <MapContainer
          center={latLng(center)}
          zoom={17}
          minZoom={3}
          maxZoom={22}
          scrollWheelZoom={false}
          doubleClickZoom={false}
          fadeAnimation={false}
          zoomAnimation={false}
          markerZoomAnimation={false}
          className="h-88 w-full bg-[#edf3e8]! sm:h-112"
        >
          <View center={center} boundary={farm ?? value} />
          <Clicks drawing={drawing} add={mapCorner} farm={farm} />
          <CoordinatePreview position={preview} />
          {preview && !editingPoints && vertices.length > 0 && (
            <Polyline
              interactive={false}
              smoothFactor={0}
              positions={[
                latLng(vertices[vertices.length - 1]!),
                latLng(preview),
              ]}
              pathOptions={{
                color: '#b88000',
                dashArray: '5 5',
                interactive: false,
              }}
            />
          )}
          {imagery !== 'boundaries' && !tileError && (
            <MapTiles
              key={imagery}
              satellite={imagery === 'satellite'}
              onError={() => setTileError(true)}
              onLoad={() => setTilesLoaded(true)}
            />
          )}
          {farm && (
            <Polygon
              interactive={false}
              smoothFactor={0}
              positions={farm.coordinates[0].map(latLng)}
              pathOptions={{
                color: '#356b30',
                fillOpacity: 0.06,
                weight: 3,
                dashArray: '8 5',
                interactive: false,
              }}
            />
          )}
          {others.map((g, i) => (
            <Polygon
              interactive={false}
              smoothFactor={0}
              key={i}
              positions={g.coordinates[0].map(latLng)}
              pathOptions={{
                color: '#64748b',
                fillOpacity: 0.15,
                interactive: false,
              }}
            />
          ))}
          {vertices.length >= 3 ? (
            <Polygon
              interactive={false}
              smoothFactor={0}
              positions={vertices.map(latLng)}
              pathOptions={{
                color: '#356b30',
                fillOpacity: 0.25,
                interactive: false,
              }}
            />
          ) : (
            <Polyline
              interactive={false}
              smoothFactor={0}
              positions={vertices.map(latLng)}
              pathOptions={{ color: '#356b30', interactive: false }}
            />
          )}
          {outside && (
            <GeoJSON
              key={JSON.stringify(outside.geometry)}
              data={outside}
              interactive={false}
              onEachFeature={(_, layer) => {
                Object.assign(layer.options, { smoothFactor: 0 });
              }}
              pathOptions={{
                color: '#bd352c',
                fillColor: '#ef6b5c',
                fillOpacity: 0.45,
                weight: 3,
                dashArray: '5 4',
              }}
            />
          )}
          {drawing &&
            editingPoints &&
            vertices.length >= 3 &&
            vertices.map((point, index) => {
              const next = vertices[(index + 1) % vertices.length]!;
              const midpoint: GeoJsonPosition = [
                (point[0] + next[0]) / 2,
                (point[1] + next[1]) / 2,
              ];
              return (
                <CircleMarker
                  key={`insert-${index}`}
                  center={latLng(midpoint)}
                  radius={5}
                  bubblingMouseEvents={false}
                  pathOptions={{
                    color: '#356b30',
                    fillColor: '#fff',
                    fillOpacity: 1,
                    weight: 1,
                  }}
                  eventHandlers={{ click: () => insertCorner(index, midpoint) }}
                >
                  <Tooltip>Add corner along this edge</Tooltip>
                </CircleMarker>
              );
            })}
          {drawing &&
            vertices.map((point, index) =>
              editingPoints ? (
                <EditableCorner
                  key={index}
                  index={index}
                  position={point}
                  farm={farm}
                  select={() => selectCorner(index)}
                  move={(next) => moveCorner(index, next)}
                />
              ) : (
                <CircleMarker
                  key={index}
                  center={latLng(point)}
                  radius={5}
                  pathOptions={{
                    color: '#174e25',
                    fillColor: '#fff',
                    fillOpacity: 1,
                    interactive: false,
                  }}
                />
              ),
            )}
        </MapContainer>
      </div>
      <p className="text-xs text-[#53645a]">
        {farm
          ? 'Dashed green: farm boundary · Gray: existing plots · Solid green: your draft. Corners near the farm edge snap to it.'
          : 'Solid green: your farm draft'}{' '}
        · Areas are previews until saved.
      </p>
      {outside && (
        <p
          role="status"
          className="rounded-lg border border-[#e7b5af] bg-[#fff3f1] px-3 py-2 text-sm text-[#92362d]"
        >
          The red area is outside your farm. Move a corner, or tap a small dot
          on that edge to add a corner and move it inside. Your other points are
          kept.
        </p>
      )}
      {drawing && (
        <fieldset className="grid min-w-0 gap-3 rounded-xl border border-[#dce5d7] p-4 sm:grid-cols-[1fr_1fr_auto]">
          <legend className="px-1 text-sm font-semibold">
            {editingPoints
              ? 'Change a corner using coordinates'
              : 'Add a corner using coordinates'}
          </legend>
          {editingPoints && (
            <label className="grid gap-1 text-sm sm:col-span-3">
              Corner
              <select
                aria-label="Choose corner"
                className="min-h-11 rounded-lg border border-gray-300 px-3"
                value={selectedCorner ?? ''}
                onChange={(event) => selectCorner(Number(event.target.value))}
              >
                <option value="" disabled>
                  Choose a corner
                </option>
                {vertices.map((_, index) => (
                  <option value={index} key={index}>
                    Corner {index + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="grid gap-1 text-sm">
            Longitude
            <input
              className="min-h-11 min-w-0 rounded-lg border border-gray-300 px-3"
              type="number"
              step="any"
              aria-label="Longitude"
              min={bounds?.minLongitude ?? -180}
              max={bounds?.maxLongitude ?? 180}
              aria-describedby={`${hintId}-longitude`}
              aria-invalid={
                outsideFarm ||
                (longitude.trim() !== '' &&
                  bounds !== null &&
                  (lon < bounds.minLongitude || lon > bounds.maxLongitude))
              }
              value={longitude}
              onChange={(e) => {
                setLongitude(e.target.value);
                setError('');
              }}
            />
            <span id={`${hintId}-longitude`} className="text-xs text-[#78896e]">
              {bounds
                ? `Farm range: ${formatCoordinate(bounds.minLongitude)} to ${formatCoordinate(bounds.maxLongitude)}`
                : '−180 to 180'}
            </span>
          </label>
          <label className="grid gap-1 text-sm">
            Latitude
            <input
              className="min-h-11 min-w-0 rounded-lg border border-gray-300 px-3"
              type="number"
              step="any"
              aria-label="Latitude"
              min={bounds?.minLatitude ?? -89}
              max={bounds?.maxLatitude ?? 89}
              aria-describedby={`${hintId}-latitude`}
              aria-invalid={
                outsideFarm ||
                (latitude.trim() !== '' &&
                  bounds !== null &&
                  (lat < bounds.minLatitude || lat > bounds.maxLatitude))
              }
              value={latitude}
              onChange={(e) => {
                setLatitude(e.target.value);
                setError('');
              }}
            />
            <span id={`${hintId}-latitude`} className="text-xs text-[#78896e]">
              {bounds
                ? `Farm range: ${formatCoordinate(bounds.minLatitude)} to ${formatCoordinate(bounds.maxLatitude)}`
                : '−89 to 89'}
            </span>
          </label>
          <button
            type="button"
            className={`${button} self-end`}
            disabled={
              !candidate ||
              outsideFarm ||
              (editingPoints && selectedCorner === null)
            }
            onClick={() => {
              const lon = Number(longitude),
                lat = Number(latitude);
              if (
                !longitude.trim() ||
                !latitude.trim() ||
                !Number.isFinite(lon) ||
                !Number.isFinite(lat) ||
                Math.abs(lon) > 180 ||
                Math.abs(lat) >= 89
              ) {
                setError('Enter valid longitude and latitude.');
                return;
              }
              if (editingPoints) {
                if (selectedCorner === null) {
                  setError('Choose a corner first.');
                  return;
                }
                if (!moveCorner(selectedCorner, [lon, lat])) return;
              } else if (!add([lon, lat])) return;
              setLongitude('');
              setLatitude('');
            }}
          >
            {editingPoints ? 'Update corner' : 'Add corner'}
          </button>
          {outsideFarm && (
            <p role="status" className="text-xs text-red-700 sm:col-span-3">
              This location is outside the farm boundary. Shared edges are
              allowed.
            </p>
          )}
          <ol className="max-h-36 overflow-auto text-xs sm:col-span-3">
            {vertices.map((p, i) => (
              <li key={i}>
                {i + 1}. Longitude {p[0]}, latitude {p[1]}
              </li>
            ))}
          </ol>
        </fieldset>
      )}
      {error && (
        <p
          role="alert"
          className="rounded-lg bg-red-50 p-3 text-sm text-red-800"
        >
          {error}
        </p>
      )}
    </div>
  );
}
