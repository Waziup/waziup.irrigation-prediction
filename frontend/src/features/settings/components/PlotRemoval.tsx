import {
  useMutation,
  useMutationState,
  useQueryClient,
} from '@tanstack/react-query';
import { plotApi } from '../../../api/plotApi';
import { useFarmContext } from '../../farm-context/FarmContext';

export function PlotRemoval({
  disabled = false,
  onBegin,
  onEnd,
}: {
  disabled?: boolean;
  onBegin?: () => void;
  onEnd?: () => void;
}) {
  const context = useFarmContext();
  const client = useQueryClient();
  const removal = useMutation({
    mutationKey: ['plot-removal'],
    mutationFn: (plotId: string) => plotApi.removePlot(plotId),
    retry: false,
  });
  // Keep a failed request visible even if context reconciliation remounts the editor.
  const failures = useMutationState({
    filters: { mutationKey: ['plot-removal'], status: 'error' },
    select: (mutation) => mutation.state.error,
  });
  const failure = failures.at(-1);
  const plot = context.currentPlot;
  const unavailable =
    disabled ||
    removal.isPending ||
    context.status !== 'ready' ||
    !plot ||
    context.plots.length <= 1;
  async function remove() {
    if (unavailable || !plot) return;
    if (
      !window.confirm(
        `Remove ${plot.name}? Its local configuration will be archived.`,
      )
    )
      return;
    for (const mutation of client
      .getMutationCache()
      .findAll({ mutationKey: ['plot-removal'], status: 'error' }))
      client.getMutationCache().remove(mutation);
    onBegin?.();
    try {
      await removal.mutateAsync(plot.id);
    } catch {
      // A failed response may still have committed. Always reread the context.
    } finally {
      await context.refresh().catch(() => undefined);
      onEnd?.();
    }
  }
  return (
    <section
      aria-label="Remove selected plot"
      className="rounded-2xl border border-[#e0e0e0] bg-white p-6"
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <strong className="text-sm text-[#374151]">
            Remove {plot?.name ?? 'this plot'}
          </strong>
          <p className="mt-1 text-xs text-[#64748b]">
            This action applies only to the selected plot. Its local
            configuration will be archived.
          </p>
        </div>
        <button
          type="button"
          disabled={unavailable}
          onClick={() => void remove()}
          className="min-h-10 rounded-lg border border-[#e6cccc] px-4 py-2 text-sm font-semibold text-[#a33d3d] disabled:cursor-default disabled:opacity-45"
        >
          {removal.isPending ? 'Removing…' : 'Remove selected plot'}
        </button>
      </div>
      {failure instanceof Error && (
        <p role="alert" className="mt-3 text-sm text-red-700">
          {failure.message}
        </p>
      )}
    </section>
  );
}
