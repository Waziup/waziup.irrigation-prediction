import { GeoJSON, Tooltip } from 'react-leaflet';
import type { FarmMapPlot } from '../../domain/map';
import { geometryJson } from '../farm-setup/validation/farmSetupValidation';
import { mapStatus, mapStatuses } from './mapStyles';
export function PlotLayer({
  plot,
  selected,
  onInspect,
}: {
  plot: FarmMapPlot;
  selected: boolean;
  onInspect: (id: string) => void;
}) {
  const status = mapStatuses[mapStatus(plot)];
  return (
    <GeoJSON
      key={`${plot.plotId}:${plot.geometryVersion}`}
      data={geometryJson(plot.geometry)}
      onEachFeature={(_, layer) => {
        Object.assign(layer.options, { smoothFactor: 0 });
      }}
      pathOptions={{
        color: status.color,
        fillColor: status.color,
        weight: selected ? 3 : 2,
        fillOpacity: selected ? 0.27 : 0.14,
        dashArray: plot.dataState === 'fresh' ? undefined : '6 4',
      }}
      eventHandlers={{ click: () => onInspect(plot.plotId) }}
    >
      <Tooltip
        permanent
        direction="center"
        className="border-0! bg-transparent! p-0! shadow-none! before:hidden!"
      >
        <span
          className={`flex items-center gap-2 rounded-lg border bg-white/95 px-2 py-2 text-[10px] sm:px-3 sm:text-xs font-semibold text-[#233e30] shadow-md ${selected ? 'border-[#809883]' : 'border-white/80'}`}
        >
          <span className={`size-2 rounded-full ${status.dot}`} />
          <span
            title={`${plot.name} · ${plot.cropType ?? 'Crop unavailable'} · ${status.label}`}
            className="max-w-20 truncate"
          >
            {plot.name}
          </span>
        </span>
      </Tooltip>
    </GeoJSON>
  );
}
