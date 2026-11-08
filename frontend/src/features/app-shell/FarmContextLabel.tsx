import { StatusBadge } from '../../components/StatusBadge';
import type { ShellFarmContext } from './context';

export function FarmContextLabel({ context }: { context: ShellFarmContext }) {
  if (context.status === 'loading')
    return (
      <p
        className="mt-2 flex flex-wrap items-center gap-2 text-sm text-gray-600 wrap-anywhere"
        role="status"
      >
        Loading farm and plot…
      </p>
    );
  if (context.status === 'unavailable')
    return (
      <div
        className="mt-2 flex flex-wrap items-center gap-2 text-sm text-gray-600 wrap-anywhere"
        role="status"
      >
        <StatusBadge tone="warning">Context unavailable</StatusBadge>
        <span>{context.message}</span>
      </div>
    );
  if (context.status === 'idle')
    return (
      <p className="mt-2 flex flex-wrap items-center gap-2 text-sm text-gray-600 wrap-anywhere">
        Farm not loaded <span aria-hidden="true">·</span> No plot selected
      </p>
    );
  return (
    <p className="mt-2 flex flex-wrap items-center gap-2 text-sm text-gray-600 wrap-anywhere">
      <span className="font-semibold">{context.farmName}</span>
      <span aria-hidden="true">·</span>
      <span>{context.plotName ?? 'No plot selected'}</span>
    </p>
  );
}
