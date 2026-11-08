import { useCallback, useMemo, useRef, useState } from 'react';
import { ApiError } from '../../api/errors';
import type { ReactNode } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { FarmContext } from './FarmContext';
import {
  createContextActions,
  farmContextKey,
  PlotSelectionError,
} from './selection';
import type { FarmContextValue } from './types';

export function FarmContextProvider({ children }: { children: ReactNode }) {
  const client = useQueryClient();
  const selectionLock = useRef(false);
  const holds = useRef(0);
  const [held, setHeld] = useState(false);
  const holdContext = useCallback(() => {
    holds.current += 1;
    setHeld(true);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      holds.current -= 1;
      setHeld(holds.current > 0);
    };
  }, []);
  const actions = useMemo(() => createContextActions(client), [client]);
  const selection = useMutation({ mutationFn: actions.select, retry: false });
  const context = useQuery({
    queryKey: farmContextKey,
    queryFn: ({ signal }) => actions.load(signal),
    enabled: !selection.isPending && !held,
    retry: false,
    staleTime: 30_000,
    refetchOnWindowFocus: true,
    refetchInterval: 60_000,
    refetchIntervalInBackground: false,
  });
  const unconfirmed =
    selection.error instanceof PlotSelectionError &&
    !selection.error.contextConfirmed;
  const status = selection.isPending
    ? 'switching'
    : context.isFetching
      ? context.data
        ? 'refreshing'
        : 'loading'
      : context.isError || unconfirmed
        ? 'error'
        : context.data
          ? 'ready'
          : 'loading';
  // Never expose an older snapshot as usable while the backend selection is uncertain.
  const snapshot = status === 'ready' ? (context.data ?? null) : null;
  const value: FarmContextValue = {
    snapshot,
    farm: snapshot?.farm ?? null,
    currentPlot:
      snapshot?.plots.find((plot) => plot.id === snapshot.currentPlotId) ??
      null,
    plots: snapshot?.plots ?? [],
    status,
    error: context.error ?? (unconfirmed ? selection.error : null),
    selectionError: selection.error,
    revision: context.dataUpdatedAt,
    contextHeld: held,
    holdContext,
    async selectPlot(plotId) {
      // Guard before mutateAsync: a second mutation observer must not replace
      // the pending state of a still-running first POST.
      if (holds.current)
        throw new ApiError('Close the settings editor before changing plots.', {
          kind: 'context',
        });
      if (selectionLock.current)
        throw new ApiError('Wait for the current plot change to finish.', {
          kind: 'context',
        });
      selectionLock.current = true;
      try {
        await selection.mutateAsync(plotId);
      } finally {
        selectionLock.current = false;
      }
    },
    async refresh() {
      if (selectionLock.current || holds.current) return;
      await context.refetch({ throwOnError: true });
      if (!selectionLock.current) selection.reset();
    },
  };
  return <FarmContext.Provider value={value}>{children}</FarmContext.Provider>;
}
