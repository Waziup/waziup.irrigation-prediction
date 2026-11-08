import type { FarmPlotCard } from '../../../domain/farm';
import { mapStatus, mapStatuses } from '../../farm-map/mapStyles';
export function PlotList({
  plots,
  currentPlotId,
  onSelectPlot,
  disabled = false,
}: {
  plots: FarmPlotCard[];
  currentPlotId: string;
  onSelectPlot: (id: string) => void;
  timezone: string;
  disabled?: boolean;
}) {
  return (
    <section aria-labelledby="farm-plots-title">
      <h2 id="farm-plots-title" className="sr-only">
        Plot status
      </h2>
      {plots.length ? (
        <ul className="divide-y divide-[#e5eadf]">
          {plots.map((plot) => {
            const status = mapStatuses[mapStatus(plot)];
            return (
              <li
                key={plot.plotId}
                className="flex flex-wrap items-center gap-3 py-3"
              >
                <span
                  aria-hidden="true"
                  className={`size-2 rounded-full ${status.dot}`}
                />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-semibold text-[#304b3b]">
                    {plot.name}
                  </span>
                  <span className="text-xs text-[#849079]">
                    {plot.cropType ?? 'Crop not set'} · {status.label}
                  </span>
                  {plot.plotId === currentPlotId && (
                    <span className="sr-only">Selected plot</span>
                  )}
                </span>
                <button
                  type="button"
                  className="min-h-10 rounded-lg px-3 py-2 text-xs font-medium text-[#3e6743] hover:bg-[#f1f5ec] disabled:opacity-45"
                  disabled={disabled}
                  onClick={() => onSelectPlot(plot.plotId)}
                  aria-label={`Open plot ${plot.name}`}
                >
                  View plot →
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="text-sm text-[#849079]">
          No plots are configured for this farm.
        </p>
      )}
    </section>
  );
}
