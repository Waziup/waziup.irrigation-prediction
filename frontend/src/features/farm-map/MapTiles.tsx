import { useState } from 'react';
import { TileLayer } from 'react-leaflet';

export function MapTiles({
  satellite,
  onLoad,
  onError,
}: {
  satellite: boolean;
  onLoad: () => void;
  onError: () => void;
}) {
  const [nativeZoom, setNativeZoom] = useState(19);
  return (
    <TileLayer
      key={nativeZoom}
      url={
        satellite
          ? 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}?blankTile=false'
          : 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
      }
      maxNativeZoom={nativeZoom}
      maxZoom={22}
      attribution={
        satellite
          ? '&copy; <a href="https://goto.arcgisonline.com/maps/World_Imagery">Esri, Vantor, Earthstar Geographics &amp; GIS User Community</a>'
          : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      }
      eventHandlers={{
        tileload: onLoad,
        tileerror: (event) => {
          // Ignore events from a higher-resolution layer already being replaced.
          if (event.coords.z > nativeZoom) return;
          if (!satellite || event.coords.z <= 12) {
            onError();
            return;
          }
          setNativeZoom((current) => Math.min(current, event.coords.z - 1));
        },
      }}
    />
  );
}
