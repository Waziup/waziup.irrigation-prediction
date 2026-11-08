import type { FarmMap } from '../../domain/map';
import type { FarmDashboard } from '../../domain/farm';
import { ApiError } from '../../api/errors';

/** Geometry is spatial state; all dashboard visuals share one status snapshot. */
export function dashboardMap(
  geometry: FarmMap,
  dashboard: FarmDashboard,
): FarmMap {
  if (!geometry.farm) return geometry;
  const cards = new Map(dashboard.plots.map((plot) => [plot.plotId, plot]));
  if (
    geometry.farm.name !== dashboard.farm.name ||
    cards.size !== geometry.plots.length ||
    geometry.plots.some((plot) => !cards.has(plot.plotId))
  )
    throw new ApiError('Farm boundaries changed. Refresh the dashboard.', {
      kind: 'context',
    });
  return {
    ...geometry,
    generatedAt: dashboard.generatedAt,
    summary: dashboard.summary,
    plots: geometry.plots.map((plot) => ({
      ...plot,
      ...cards.get(plot.plotId)!,
    })),
  };
}
