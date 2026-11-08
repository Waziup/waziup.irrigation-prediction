import { useQuery } from '@tanstack/react-query';
import { farmApi } from '../../../api/farmApi';
import { ApiError } from '../../../api/errors';
import type { PlotContext } from '../../../domain/plot';
import { useFarmContext } from '../../farm-context/FarmContext';

export const farmDashboardKey = (plotId: string) =>
  ['farm-dashboard', plotId] as const;
export async function loadFarmDashboard(
  context: PlotContext,
  signal?: AbortSignal,
) {
  const dashboard = await farmApi.getFarmDashboard(
    signal ? { signal } : undefined,
  );
  signal?.throwIfAborted();
  const expectedIds = new Set(context.plots.map((plot) => plot.id));
  const returnedIds = new Set(dashboard.plots.map((plot) => plot.plotId));
  // This endpoint has no farm_id. Compare the identity fields it does return
  // and require the same plot set before displaying the snapshot.
  if (
    dashboard.farm.name !== context.farm.name ||
    dashboard.farm.gatewayId !== context.farm.gatewayId ||
    returnedIds.size !== dashboard.plots.length ||
    returnedIds.size !== expectedIds.size ||
    [...returnedIds].some((id) => !expectedIds.has(id))
  ) {
    throw new ApiError(
      'Farm details changed while loading the dashboard. Refresh the farm context to continue.',
      { kind: 'context' },
    );
  }
  return dashboard;
}
export function useFarmDashboard() {
  const context = useFarmContext();
  return useQuery({
    queryKey: farmDashboardKey(context.currentPlot?.id ?? 'unselected'),
    queryFn: ({ signal }) => {
      if (!context.snapshot || context.status !== 'ready')
        throw new ApiError(
          'Confirm the selected farm before loading its dashboard.',
          { kind: 'context' },
        );
      return loadFarmDashboard(context.snapshot, signal);
    },
    enabled: context.status === 'ready',
    retry: false,
    staleTime: 0,
    refetchOnWindowFocus: true,
    refetchInterval: 60_000,
    refetchIntervalInBackground: false,
  });
}
