import { useQuery } from '@tanstack/react-query';
import { loadSettings } from '../settingsActions';
export function usePlotSettings(plotId: string) {
  return useQuery({
    queryKey: ['plot', plotId, 'settings-editor'],
    queryFn: ({ signal }) => loadSettings(plotId, signal),
    staleTime: Infinity,
    refetchOnMount: 'always',
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
}
