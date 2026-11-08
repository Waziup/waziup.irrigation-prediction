import type { QueryClient } from '@tanstack/react-query';
import { ApiError } from '../../api/errors';
import type { PlotContext } from '../../domain/plot';
import { farmContextApi } from './farmContextApi';

export const farmContextKey = ['farm-context'] as const;
export class PlotSelectionError extends Error {
  readonly contextConfirmed: boolean;
  constructor(cause: unknown, contextConfirmed: boolean) {
    const message =
      cause instanceof Error ? cause.message : 'Plot selection failed.';
    super(
      contextConfirmed
        ? message
        : `${message} The current plot could not be verified. Refresh before continuing.`,
      { cause },
    );
    this.name = 'PlotSelectionError';
    this.contextConfirmed = contextConfirmed;
  }
}
// Until individual features arrive, all other query data may depend on the
// server-selected plot. Cancel and discard it before publishing a new context.
export async function clearDependentQueries(client: QueryClient) {
  const filters = {
    predicate: (query: { queryKey: readonly unknown[] }) =>
      query.queryKey[0] !== farmContextKey[0],
  };
  await client.cancelQueries(filters);
  client.removeQueries(filters);
}
export function createContextActions(client: QueryClient) {
  let selecting = false;
  return {
    async load(signal?: AbortSignal): Promise<PlotContext> {
      signal?.throwIfAborted();
      if (selecting)
        throw new ApiError('Plot selection is in progress.', {
          kind: 'context',
        });
      const context = await farmContextApi.load(
        signal ? { signal } : undefined,
      );
      signal?.throwIfAborted();
      const previous = client.getQueryData<PlotContext>(farmContextKey);
      if (previous && JSON.stringify(previous) !== JSON.stringify(context))
        await clearDependentQueries(client);
      return context;
    },
    async select(plotId: string): Promise<void> {
      if (selecting)
        throw new ApiError('Wait for the current plot change to finish.', {
          kind: 'context',
        });
      const previous = client.getQueryData<PlotContext>(farmContextKey);
      if (!previous?.plots.some((plot) => plot.id === plotId))
        throw new ApiError(
          'This plot is not in the loaded registry. Refresh the plot list.',
          { kind: 'context' },
        );
      selecting = true;
      try {
        await client.cancelQueries({ queryKey: farmContextKey });
        await clearDependentQueries(client);
        // Do not abort a POST then assume it was rolled back. Even a lost
        // response may have committed the backend's persistent selection.
        try {
          await farmContextApi.persistSelection(plotId);
          const confirmed = await farmContextApi.load();
          client.setQueryData(farmContextKey, confirmed);
          if (confirmed.currentPlotId !== plotId)
            throw new ApiError(
              'Another session changed the selected plot. The current backend selection has been reloaded.',
              { kind: 'context' },
            );
        } catch (cause) {
          let contextConfirmed = false;
          try {
            const reconciled = await farmContextApi.load();
            client.setQueryData(farmContextKey, reconciled);
            contextConfirmed = true;
          } catch {
            /* Keep dependent pages blocked until an explicit refresh succeeds. */
          }
          throw new PlotSelectionError(cause, contextConfirmed);
        }
      } finally {
        selecting = false;
      }
    },
  };
}
