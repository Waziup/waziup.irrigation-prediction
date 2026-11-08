import type { Plot } from '../../../domain/plot';
import { fromSquareMetres } from '../../../domain/plot';
export function PlotHeader({ plot }: { plot: Plot }) {
  return (
    <header className="min-w-0">
      <h2 className="text-2xl font-semibold tracking-tight text-[#294531]">
        {plot.name}
      </h2>
      <p className="mt-2 text-xs text-[#8a967e]">
        {plot.areaM2 > 0
          ? `${fromSquareMetres(plot.areaM2, plot.areaUnit).toLocaleString(undefined, { maximumFractionDigits: 2 })} ${plot.areaUnit === 'm2' ? 'm²' : plot.areaUnit}`
          : 'Area not measured'}
      </p>
    </header>
  );
}
