import { MapTiles } from './MapTiles';
import { useEffect, useRef, useState } from 'react';
import {
  GeoJSON,
  MapContainer,
  Pane,
  SVGOverlay,
  ZoomControl,
  useMap,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { PolygonGeometry } from '../../domain/geometry';
import type { FarmMap as FarmMapData } from '../../domain/map';
import { geometryJson } from '../farm-setup/validation/farmSetupValidation';
import { PlotLayer } from './PlotLayer';
import { MapControls, type Imagery } from './MapControls';

function View({
  geometry,
  fit,
  compact,
}: {
  geometry: PolygonGeometry;
  fit: number;
  compact: boolean;
}) {
  const map = useMap();
  useEffect(() => {
    const refit = () => {
      map.invalidateSize({ pan: false });
      map.fitBounds(
        geometry.coordinates[0].map((p) => [p[1], p[0]]),
        {
          padding:
            map.getContainer().clientWidth < 600
              ? [22, 60]
              : compact
                ? [35, 45]
                : [70, 85],
          maxZoom: 19,
          animate: false,
        },
      );
    };
    refit();
    const observer = new ResizeObserver(refit);
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map, geometry, fit, compact]);
  return null;
}
function BoundaryGround({ geometry }: { geometry: PolygonGeometry }) {
  const positions = geometry.coordinates[0];
  const minLon = Math.min(...positions.map((p) => p[0])),
    maxLon = Math.max(...positions.map((p) => p[0]));
  const minLat = Math.min(...positions.map((p) => p[1])),
    maxLat = Math.max(...positions.map((p) => p[1]));
  const dx = maxLon - minLon,
    dy = maxLat - minLat;
  return (
    <SVGOverlay
      pane="tilePane"
      bounds={[
        [minLat - dy * 3, minLon - dx * 3],
        [maxLat + dy * 3, maxLon + dx * 3],
      ]}
      attributes={{ viewBox: '0 0 1400 1400', preserveAspectRatio: 'none' }}
      interactive={false}
    >
      <defs>
        <pattern
          id="farm-grid"
          width="40"
          height="40"
          patternUnits="userSpaceOnUse"
        >
          <path
            d="M40 0H0V40"
            fill="none"
            stroke="#a5b7a0"
            strokeWidth=".6"
            opacity=".35"
          />
          <circle cx="0" cy="0" r="1" fill="#8da185" opacity=".4" />
        </pattern>
        <pattern
          id="farm-hatch"
          width="8"
          height="8"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(35)"
        >
          <path d="M0 0V8" stroke="#b2c0a8" strokeWidth=".6" opacity=".13" />
        </pattern>
      </defs>
      <rect width="1400" height="1400" fill="#e9eee1" />
      <rect width="1400" height="1400" fill="url(#farm-hatch)" />
      <rect width="1400" height="1400" fill="url(#farm-grid)" />
    </SVGOverlay>
  );
}
export function FarmMap({
  data,
  selected,
  onInspect,
  compact = false,
}: {
  data: FarmMapData;
  compact?: boolean;
  selected: string | null;
  onInspect: (id: string) => void;
}) {
  const [imagery, setImagery] = useState<Imagery>('satellite');
  const [failed, setFailed] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [fit, setFit] = useState(0);
  useEffect(() => {
    if (imagery === 'illustrated' || failed) return;
    timer.current = setTimeout(() => setFailed(true), 12_000);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [imagery, failed]);
  if (!data.setup.complete || !data.farm) return null;
  const farm = data.farm,
    schematic = imagery === 'illustrated' || failed;
  return (
    <div
      className="relative isolate overflow-hidden bg-[#e9eee1]"
      aria-label="Farm boundary map"
    >
      <MapContainer
        preferCanvas
        fadeAnimation={false}
        zoomAnimation={false}
        markerZoomAnimation={false}
        center={[
          farm.geometry.coordinates[0][0][1],
          farm.geometry.coordinates[0][0][0],
        ]}
        zoom={16}
        zoomSnap={0.25}
        maxZoom={22}
        zoomControl={false}
        scrollWheelZoom={false}
        className={`${compact ? 'h-100 sm:h-116' : 'h-116 sm:h-150 xl:h-[min(70vh,760px)] xl:min-h-150'} w-full bg-[#e9eee1]! [&_.leaflet-control-zoom]:overflow-hidden [&_.leaflet-control-zoom]:rounded-xl! [&_.leaflet-control-zoom]:border-white/80! [&_.leaflet-control-zoom]:shadow-lg! [&_.leaflet-control-zoom_a]:size-10! [&_.leaflet-control-zoom_a]:leading-10!`}
      >
        <View geometry={farm.geometry} fit={fit} compact={compact} />
        <ZoomControl position="bottomright" />
        {schematic && <BoundaryGround geometry={farm.geometry} />}
        {!schematic && (
          <MapTiles
            key={imagery}
            satellite={imagery === 'satellite'}
            onError={() => setFailed(true)}
            onLoad={() => {
              if (timer.current) clearTimeout(timer.current);
            }}
          />
        )}
        <Pane name="farm-outlines" className="z-450! bg-transparent!">
          <GeoJSON
            key={farm.geometryVersion}
            data={geometryJson(farm.geometry)}
            onEachFeature={(_, layer) => {
              Object.assign(layer.options, { smoothFactor: 0 });
            }}
            interactive={false}
            pathOptions={{
              color: '#4f6850',
              weight: 2,
              fill: false,
              fillOpacity: schematic ? 0.58 : 0.08,
              dashArray: '7 6',
            }}
          />
          {data.plots.map((plot) => (
            <PlotLayer
              key={plot.plotId}
              plot={plot}
              selected={plot.plotId === selected}
              onInspect={onInspect}
            />
          ))}
        </Pane>
      </MapContainer>
      {failed && (
        <p
          role="status"
          className="absolute inset-x-4 top-4 z-500 w-fit max-w-[calc(100%-2rem)] rounded-lg border border-[#e1d6b5] bg-[#fffaf0]/95 px-3 py-2 text-xs text-[#806222] shadow-sm"
        >
          Map imagery unavailable. Boundaries remain usable.
        </p>
      )}
      <div className="absolute bottom-7 left-4 z-500 sm:left-5">
        <MapControls
          imagery={schematic ? 'illustrated' : imagery}
          onImagery={(mode) => {
            setFailed(false);
            setImagery(mode);
          }}
          onFit={() => setFit((n) => n + 1)}
        />
      </div>
      <span className="sr-only">
        Dashed outline: farm boundary. Tap a plot to open its overview. Boundary
        view is a schematic of saved geometry, not satellite imagery.
      </span>
    </div>
  );
}
