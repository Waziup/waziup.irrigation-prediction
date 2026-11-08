import { useState } from 'react';
import { useNavigate } from 'react-router';
import { ApiError } from '../../../api/errors';
import { useFarmContext } from '../../farm-context/FarmContext';
import type { FarmContextValue } from '../../farm-context/types';

/** Reuse selection acknowledgement, readback, cache clearing and reconciliation. */
export async function openMapPlot(
  context: FarmContextValue,
  navigate: (path: string) => void,
  plotId: string,
): Promise<void> {
  if (context.status !== 'ready' || context.contextHeld)
    throw new ApiError('Wait for the farm context before opening a plot.', {
      kind: 'context',
    });
  if (!context.plots.some((plot) => plot.id === plotId))
    throw new ApiError(
      'This plot is no longer in the farm. Refresh before continuing.',
      { kind: 'context' },
    );
  await context.selectPlot(plotId);
  // Only the confirmed existing selection flow can unlock the plot page.
  navigate('/plot');
}
export function useMapSelection() {
  const context = useFarmContext();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  return {
    disabled: context.status !== 'ready' || Boolean(context.contextHeld),
    error,
    openPlot(plotId: string) {
      setError(null);
      void openMapPlot(
        context,
        (path) => {
          void navigate(path);
        },
        plotId,
      ).catch((cause) => {
        setError(
          cause instanceof Error
            ? cause.message
            : 'Plot could not be opened. Refresh and try again.',
        );
      });
    },
  };
}
