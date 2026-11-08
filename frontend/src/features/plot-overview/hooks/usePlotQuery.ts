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
  const plotId = context.currentPlot?.id;
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
    refetchInterval: interval,
    refetchIntervalInBackground: false,
  });
}
