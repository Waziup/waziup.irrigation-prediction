import { useQuery } from '@tanstack/react-query';
import type { Coordinates } from '../../domain/plot';
import { eoApi } from '../../api/eoApi';

export function usePlotSatellite(
  plotId: string | undefined,
  enabled: boolean,
  coordinates?: Coordinates | null,
) {
  return useQuery({
    queryKey: ['plot', plotId ?? 'unselected', 'eo', coordinates ?? null],
    queryFn: ({ signal }) => eoApi.getCurrentObservation({ signal }, plotId!),
    enabled: enabled && !!plotId,
    retry: false,
    staleTime: (query) =>
      query.state.data?.available ? 24 * 60 * 60_000 : 5 * 60_000,
    gcTime: 24 * 60 * 60_000,
    refetchOnWindowFocus: true,
    refetchInterval: 24 * 60 * 60_000,
    refetchIntervalInBackground: false,
  });
}
