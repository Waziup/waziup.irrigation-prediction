import { useQuery } from '@tanstack/react-query';
import { farmMapApi } from '../api/farmMapApi';
import { useFarmContext } from '../../farm-context/FarmContext';
import { ApiError } from '../../../api/errors';
import type { FarmMap } from '../../../domain/map';
import type { PlotContext } from '../../../domain/plot';
export const farmMapKey = ['farm-map', 'installation'] as const;
export async function loadFarmMap(context: PlotContext, signal?: AbortSignal) {
  const data = await farmMapApi.load(signal ? { signal } : undefined);
  signal?.throwIfAborted();
  if (data.setup.complete && data.farm) {
    const ids = new Set(context.plots.map((plot) => plot.id));
    if (
      data.farm.name !== context.farm.name ||
      data.farm.gatewayId !== context.farm.gatewayId ||
      data.plots.length !== ids.size ||
      data.plots.some((plot) => !ids.has(plot.plotId))
    )
      throw new ApiError(
        'Farm details changed. Refresh the farm context to load its map.',
        { kind: 'context' },
      );
  } else if (context.plots.length) {
    throw new ApiError(
      'Farm details changed. Refresh the farm context to continue.',
      { kind: 'context' },
    );
  }
  return data;
}
export function useFarmMap() {
  const context = useFarmContext();
  return useQuery({
    queryKey: [
      ...farmMapKey,
      context.snapshot?.farm.name ?? '',
      context.snapshot?.plots
        .map((plot) => plot.id)
        .sort()
        .join(',') ?? '',
    ],
    queryFn: ({ signal }) => {
      if (!context.snapshot || context.status !== 'ready')
        throw new ApiError('Confirm farm context before loading the map.', {
          kind: 'context',
        });
      return loadFarmMap(context.snapshot, signal);
    },
    enabled: context.status === 'ready' && !context.contextHeld,
    retry: false,
    staleTime: 0,
    refetchOnWindowFocus: true,
    refetchInterval: 60_000,
    refetchIntervalInBackground: false,
  });
}

/** Cached geometry remains useful; cached readings must never appear current. */
export function staleMap(data: FarmMap): FarmMap {
  if (data.farm === null) return data;
  return {
    ...data,
    summary: { ...data.summary, staleOrUnavailablePlots: data.plots.length },
    plots: data.plots.map((plot) => ({
      ...plot,
      dataState: plot.dataState === 'unavailable' ? 'unavailable' : 'stale',
    })),
  };
}
