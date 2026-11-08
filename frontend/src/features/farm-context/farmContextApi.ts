import { farmApi } from '../../api/farmApi';
import { plotApi } from '../../api/plotApi';
import { ApiError } from '../../api/errors';
import type { RequestOptions } from '../../domain/api';
import type { PlotContext } from '../../domain/plot';

function agree(registry: PlotContext, plots: PlotContext): boolean {
  if (
    registry.currentPlotId !== plots.currentPlotId ||
    JSON.stringify(registry.farm) !== JSON.stringify(plots.farm)
  )
    return false;
  const byId = new Map(registry.plots.map((plot) => [plot.id, plot]));
  return (
    byId.size === plots.plots.length &&
    plots.plots.every((plot) => {
      const registered = byId.get(plot.id);
      return (
        registered?.name === plot.name &&
        registered.areaM2 === plot.areaM2 &&
        registered.areaUnit === plot.areaUnit
      );
    })
  );
}
export const farmContextApi = {
  async load(options?: RequestOptions): Promise<PlotContext> {
    // Both endpoints carry selection. A concurrent registry change must not
    // splice farm/plot metadata from different snapshots into one UI context.
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const registry = await farmApi.getRegistry(options);
      const plots = await plotApi.getPlots(options);
      if (agree(registry, plots))
        return {
          farm: registry.farm,
          plots: plots.plots,
          currentPlotId: plots.currentPlotId,
        };
    }
    throw new ApiError(
      'Farm or plot selection changed while loading. Refresh to try again.',
      { kind: 'context' },
    );
  },
  persistSelection: (plotId: string) => plotApi.selectPlot(plotId),
};
