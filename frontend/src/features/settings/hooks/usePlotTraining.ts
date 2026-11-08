import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import type { SettingsSnapshot } from '../types';
import { startPlotTraining } from '../trainingActions';
import { trainingApi } from '../../../api/trainingApi';
export function usePlotTraining(plotId: string) {
  const client = useQueryClient();
  const lock = useRef(false);
  const status = useQuery({
    queryKey: ['plot', plotId, 'training-status'],
    queryFn: ({ signal }) => trainingApi.getTrainingStatus(plotId, { signal }),
    retry: false,
    staleTime: 0,
    refetchOnWindowFocus: false,
    refetchInterval: 10_000,
    refetchIntervalInBackground: false,
  });
  useEffect(() => {
    if (status.data?.status === 'finished' || status.data?.status === 'error')
      void client.invalidateQueries({
        predicate: (q) =>
          q.queryKey[0] === 'plot' &&
          q.queryKey[1] === plotId &&
          !['settings-editor', 'training-status'].includes(
            String(q.queryKey[2]),
          ),
        refetchType: 'none',
      });
  }, [status.data?.status, client, plotId]);
  const start = useMutation({
    retry: false,
    mutationFn: ({ baseline }: { baseline: SettingsSnapshot }) =>
      startPlotTraining(plotId, baseline),
    onSettled: async () => {
      await status.refetch();
      await client.invalidateQueries({
        predicate: (q) =>
          q.queryKey[0] === 'plot' &&
          q.queryKey[1] === plotId &&
          !['settings-editor', 'training-status'].includes(
            String(q.queryKey[2]),
          ),
        refetchType: 'none',
      });
    },
  });
  async function startTraining(baseline: SettingsSnapshot) {
    // Lock before mutateAsync: a second observer must not replace the pending
    // state of the first start while its response/status reconciliation is pending.
    if (lock.current) return;
    lock.current = true;
    try {
      await start.mutateAsync({ baseline });
    } finally {
      lock.current = false;
    }
  }
  return { status, start, startTraining };
}
