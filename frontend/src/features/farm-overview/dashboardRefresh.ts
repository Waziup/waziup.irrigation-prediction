import type { FarmDashboard } from '../../domain/farm';

/** Keep the map mounted after refresh failure without presenting old evidence as live. */
export function failedDashboardRefresh(data: FarmDashboard): FarmDashboard {
  return {
    ...data,
    summary: {
      ...data.summary,
      equipmentActive: 0,
      staleOrUnavailablePlots: data.plots.length,
    },
    plots: data.plots.map((plot) => ({
      ...plot,
      dataState: plot.dataState === 'unavailable' ? 'unavailable' : 'stale',
      dataReason: 'The latest farm status could not be loaded.',
      equipment: {
        ...plot.equipment,
        state: plot.equipment.configured ? 'ready' : 'not_configured',
      },
    })),
  };
}
