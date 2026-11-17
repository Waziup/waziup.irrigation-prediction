import { useQuery } from '@tanstack/react-query';
import { ApiError } from '../../../api/errors';
import { useFarmContext } from '../../farm-context/FarmContext';

export function usePlotQuery<T>(
  resource: string,
  load: (signal: AbortSignal, plotId: string) => Promise<T>,
  interval: number | false = 60_000,
  staleTime = 0,
) {
  const context = useFarmContext();
  const plotId =
    context.currentPlot?.id ??
    (context.status === 'refreshing'
      ? context.lastConfirmedSnapshot?.currentPlotId
      : undefined);
  return useQuery({
    queryKey: ['plot', plotId ?? 'unselected', resource],
    enabled: context.status === 'ready' && !!plotId,
    queryFn: ({ signal }) => {
      if (!plotId || context.status !== 'ready')
        throw new ApiError(
          'Confirm the selected plot before loading its data.',
          { kind: 'context' },
        );
      return load(signal, plotId);
    },
    retry: false,
    staleTime,
    refetchOnWindowFocus: true,
    refetchInterval: (query) => {
      if (interval === false) return false;
      if (query.state.status === 'error') return Math.max(interval, 30_000);
      const value = query.state.data;
      return value &&
        typeof value === 'object' &&
        'loading' in value &&
        value.loading === true
        ? 3_000
        : interval;
    },
    refetchIntervalInBackground: false,
  });
}
