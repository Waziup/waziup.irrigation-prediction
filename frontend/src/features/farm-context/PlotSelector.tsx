import { useId } from 'react';
import { useFarmContext } from './FarmContext';

export function PlotSelector() {
  const context = useFarmContext();
  const labelId = useId();
  return (
    <div className="w-full min-w-0 [&>label]:mb-1 [&>label]:block [&>label]:text-xs [&>label]:text-gray-600">
      <label htmlFor={labelId}>Current plot</label>
      <div className="flex gap-2">
        <select
          className="min-h-11 min-w-0 rounded-lg border border-gray-200 bg-white px-3 py-2 text-[#2d5a27] disabled:cursor-wait disabled:opacity-70 w-full"
          id={labelId}
          value={context.currentPlot?.id ?? ''}
          disabled={context.status !== 'ready' || context.contextHeld}
          onChange={(event) => {
            void context.selectPlot(event.target.value).catch(() => {
              /* Rendered below from mutation state. */
            });
          }}
        >
          {!context.currentPlot && (
            <option value="">
              {context.status === 'switching'
                ? 'Switching plot…'
                : 'Plot unavailable'}
            </option>
          )}
          {context.plots.map((plot) => (
            <option key={plot.id} value={plot.id}>
              {plot.name}
            </option>
          ))}
        </select>
        <button
          className="min-h-11 min-w-0 rounded-lg border border-gray-200 bg-white px-3 py-2 text-[#2d5a27] disabled:cursor-wait disabled:opacity-70"
          type="button"
          disabled={
            context.contextHeld ||
            context.status === 'loading' ||
            context.status === 'refreshing' ||
            context.status === 'switching'
          }
          onClick={() => {
            void context.refresh().catch(() => {
              /* Query error is rendered in the shell. */
            });
          }}
        >
          Refresh
        </button>
      </div>
      {context.contextHeld && (
        <p className="my-3 leading-relaxed text-gray-600">
          Close settings to change or refresh the current plot.
        </p>
      )}
      {context.selectionError && (
        <p className="mt-2 text-sm text-[#a52020] wrap-anywhere" role="alert">
          {context.selectionError.message}
        </p>
      )}
    </div>
  );
}
