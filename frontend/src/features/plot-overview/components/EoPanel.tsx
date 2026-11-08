import type { EOObservation } from '../../../domain/eo';
import type { ResourceState } from '../types';
import { ResourcePanel } from './ResourcePanel';
import {
  formatAmount,
  formatDashboardTime,
} from '../../../shared/presentation';
export function EoPanel({
  state,
  timezone,
  onRetry,
}: {
  state: ResourceState<EOObservation>;
  timezone: string;
  onRetry: () => void;
}) {
  if (state.status !== 'ready' || !state.data.available)
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-5"
        aria-labelledby="eo-empty-title"
      >
        <h2
          id="eo-empty-title"
          className="text-base font-semibold text-[#253247]"
        >
          Earth observation
        </h2>
        <p className="mt-1 text-xs text-slate-500">
          Satellite vegetation indices for this plot.
        </p>
        <dl className="mt-4 grid grid-cols-3 gap-2">
          {['NDVI', 'NDRE', 'Vegetation trend'].map((label) => (
            <div key={label} className="rounded-xl bg-[#f3f7f2] p-3">
              <dt className="text-xs text-slate-500">{label}</dt>
              <dd className="mt-2 font-semibold">—</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-xs text-slate-500">
          {state.status === 'loading'
            ? 'Loading satellite observations…'
            : state.status === 'ready' && !state.data.available
              ? state.data.reason
              : 'Satellite observations couldn’t be loaded.'}
        </p>
        {state.status === 'error' && (
          <button
            type="button"
            className="mt-2 text-xs text-[#345c3d] underline"
            onClick={onRetry}
          >
            Refresh observations
          </button>
        )}
      </section>
    );
  return (
    <ResourcePanel title="Earth observation" state={state} onRetry={onRetry}>
      {(eo) =>
        !eo.available ? (
          <p className="my-3 leading-relaxed text-gray-600">
            Earth observation unavailable. {eo.reason}
          </p>
        ) : (
          <>
            <dl className="my-4 grid grid-cols-[repeat(auto-fit,minmax(min(100%,180px),1fr))] gap-3 [&_dt]:text-xs [&_dt]:text-slate-500 [&_dd]:mt-2 [&_dd]:font-semibold [&_dd]:text-[#253c30] [&_dd]:wrap-anywhere [&>div]:rounded-xl [&>div]:bg-[#f3f7f2] [&>div]:p-4">
              <div>
                <dt>NDVI</dt>
                <dd>{formatAmount(eo.latestNdvi, '', 3)}</dd>
              </div>
              <div>
                <dt>NDRE</dt>
                <dd>{formatAmount(eo.latestNdre, '', 3)}</dd>
              </div>
              <div>
                <dt>Vegetation trend</dt>
                <dd>{eo.ndvi.available ? eo.ndvi.trend : eo.ndre.trend}</dd>
              </div>
              <div>
                <dt>Last observation</dt>
                <dd>{formatDashboardTime(eo.latestObservation, timezone)}</dd>
              </div>
            </dl>
          </>
        )
      }
    </ResourcePanel>
  );
}
