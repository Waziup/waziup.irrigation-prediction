import { lazy, Suspense, useState } from 'react';
import { Link } from 'react-router';
import { FarmMap } from './FarmMap';
import { MapLegend } from './MapLegend';
import { mapStatus, mapStatuses } from './mapStyles';
import { mapButton } from './MapControls';
import { staleMap, useFarmMap } from './hooks/useFarmMap';
import { useMapSelection } from './hooks/useMapSelection';
import { useFarmContext } from '../farm-context/FarmContext';
import type { BoundaryTarget } from './boundaryActions';
const BoundaryEditPanel = lazy(() =>
  import('./BoundaryEditPanel').then((module) => ({
    default: module.BoundaryEditPanel,
  })),
);
import { ApiError } from '../../api/errors';
import type { FarmMap as FarmMapData, FarmMapPlot } from '../../domain/map';

export function PlotDetails({ plot }: { plot: FarmMapPlot }) {
  const presentation = mapStatuses[mapStatus(plot)];
  return (
    <section aria-label={`${plot.name} details`} className="min-w-0">
      <h3 className="text-sm font-semibold text-[#304b3b]">{plot.name}</h3>
      <p className="mt-1 text-xs text-[#89947e]">
        {plot.cropType ?? 'Crop unavailable'} ·{' '}
        {(plot.areaM2 / 10000).toLocaleString(undefined, {
          maximumFractionDigits: 2,
        })}{' '}
        ha
      </p>
      <p className="mt-4 flex w-fit items-center gap-1.5 rounded-md border border-[#e5eadd] bg-[#fafbf7] px-2 py-1 text-[11px] font-medium text-[#66735d]">
        <span
          aria-hidden="true"
          className={`size-1.5 rounded-full ${presentation.dot}`}
        />
        {presentation.label}
      </p>
      {plot.dataState !== 'fresh' && (
        <p className="mt-2 text-sm text-amber-900">
          {plot.dataReason ||
            (plot.dataState === 'stale'
              ? 'Irrigation advice needs updating.'
              : 'Irrigation advice is not available yet.')}
        </p>
      )}
      <dl className="mt-5 grid gap-4 text-xs">
        <div>
          <dt className="text-[#64748b]">Soil tension</dt>
          <dd>
            {plot.dataState === 'fresh' && plot.tensionCbar !== null
              ? `${plot.tensionCbar} cbar`
              : 'Unavailable'}
          </dd>
        </div>
        <div>
          <dt className="text-[#64748b]">
            Recommended action
            {plot.dataState === 'stale' ? ' (last known)' : ''}
          </dt>
          <dd>
            {plot.dataState === 'unavailable'
              ? 'Await current readings'
              : plot.nextAction}
          </dd>
        </div>
        <div>
          <dt className="text-[#64748b]">Irrigation</dt>
          <dd>
            {plot.dataState !== 'fresh'
              ? 'Current status unavailable'
              : (plot.operation?.status.replaceAll('_', ' ') ??
                'No current operation')}
          </dd>
        </div>
      </dl>
      <details className="mt-4 text-[11px] text-[#89947e]">
        <summary className="cursor-pointer">More measurements</summary>
        <p className="mt-2">
          Vegetation indices (NDVI / NDRE): Unavailable in current map data
        </p>
      </details>
    </section>
  );
}
export function FarmMapContent({
  data,
  onOpenPlot,
  disabled = false,
  onEditBoundary,
}: {
  data: FarmMapData;
  onOpenPlot?: (plotId: string) => void;
  onEditBoundary?: (target: BoundaryTarget) => void;
  disabled?: boolean;
}) {
  const [selected, setSelected] = useState<string | null>(
    data.plots[0]?.plotId ?? null,
  );
  if (!data.setup.complete || !data.farm || !data.summary)
    return (
      <section className="rounded-2xl border border-[#e2e8de] bg-white p-8">
        <p className="text-xs font-semibold tracking-widest text-[#73816e] uppercase">
          A place to begin
        </p>
        <h2 className="mt-3 text-2xl font-semibold text-[#243d2d]">
          Create your farm map
        </h2>
        <p className="mt-2 mb-5 text-sm text-[#75816f]">
          Draw the farm boundary and first plot to get started.
        </p>
        <Link to="/setup" className={mapButton}>
          Set up farm
        </Link>
      </section>
    );
  const plot = data.plots.find((p) => p.plotId === selected) ?? data.plots[0];
  const healthy = data.plots.filter((p) => mapStatus(p) === 'healthy').length;
  return (
    <div className="grid gap-6">
      <header className="flex flex-wrap items-center justify-between gap-5">
        <div>
          <p className="mb-2 text-[11px] font-semibold tracking-[.18em] text-[#84917f] uppercase">
            Your land, at a glance
          </p>
          <h2 className="text-2xl font-semibold tracking-tight text-[#223e2c] sm:text-3xl">
            {data.farm.name}
          </h2>
          <p className="mt-2 flex flex-wrap items-center gap-2 text-sm text-[#7a8675]">
            <span>
              {(data.farm.areaM2 / 10000).toLocaleString(undefined, {
                maximumFractionDigits: 1,
              })}{' '}
              hectares
            </span>
            <span aria-hidden="true">·</span>
            <span>{data.summary.totalPlots} plots</span>
            <span aria-hidden="true">·</span>
            <span>Farm overview</span>
          </p>
        </div>
        {onEditBoundary && (
          <button
            type="button"
            className={mapButton}
            disabled={disabled}
            onClick={() => onEditBoundary({ entity: 'farm' })}
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              aria-hidden="true"
            >
              <path d="m4 16 12-12 4 4-12 12-5 1Zm10-10 4 4" />
            </svg>
            Edit farm boundary
          </button>
        )}
      </header>
      <div className="grid grid-cols-3 divide-x divide-[#e4e9df] rounded-2xl border border-[#e4e9df] bg-white py-4 sm:py-5">
        <div className="px-4 sm:px-6">
          <p className="text-[11px] font-medium text-[#7c8975] sm:text-xs">
            Needing attention
          </p>
          <p className="mt-1.5 text-2xl font-semibold tracking-tight text-[#b35d37]">
            {data.summary.plotsNeedingAttention}
            <span className="ml-1.5 text-xs font-normal text-[#899480]">
              plots
            </span>
          </p>
        </div>
        <div className="px-4 sm:px-6">
          <p className="text-[11px] font-medium text-[#7c8975] sm:text-xs">
            Healthy plots
          </p>
          <p className="mt-1.5 text-2xl font-semibold tracking-tight text-[#356044]">
            {healthy}
            <span className="ml-1.5 text-xs font-normal text-[#899480]">
              of {data.plots.length}
            </span>
          </p>
        </div>
        <div className="px-4 sm:px-6">
          <p className="text-[11px] font-medium text-[#7c8975] sm:text-xs">
            Current irrigation plan
          </p>
          <p className="mt-1.5 text-2xl font-semibold tracking-tight text-[#304b3b]">
            {data.summary.plannedWaterM3.toLocaleString()}
            <span className="ml-1.5 text-xs font-normal text-[#899480]">
              m³
            </span>
          </p>
        </div>
      </div>
      {data.summary.staleOrUnavailablePlots > 0 && (
        <p
          role="status"
          className="rounded-xl border border-[#ecdfbc] bg-[#fffaf0] px-4 py-3 text-sm text-[#8b723c]"
        >
          {data.summary.staleOrUnavailablePlots} plots have out-of-date or
          unavailable irrigation data. Their boundaries remain visible.
        </p>
      )}
      <div className="grid overflow-hidden rounded-2xl border border-[#dde5d8] bg-white shadow-sm shadow-[#243d2d]/3 xl:grid-cols-[minmax(0,1fr)_300px] min-[1500px]:grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-w-0">
          <FarmMap
            data={data}
            selected={plot?.plotId ?? null}
            onInspect={(plotId) => {
              if (disabled) return;
              if (onOpenPlot) onOpenPlot(plotId);
              else setSelected(plotId);
            }}
          />
          <div className="border-t border-[#e7ebe3] bg-white px-5 py-4">
            <MapLegend />
          </div>
        </div>
        <aside
          className="min-w-0 border-t border-[#e4e9df] xl:border-t-0 xl:border-l"
          aria-label="Plot explorer"
        >
          <div className="flex items-center justify-between border-b border-[#edf0e9] px-5 py-4">
            <h3 className="text-sm font-semibold text-[#304b3b]">Your plots</h3>
            <span className="rounded-md bg-[#f1f5ed] px-2 py-0.5 text-xs font-medium text-[#75836b]">
              {data.plots.length}
            </span>
          </div>
          <section
            aria-label="Farm plots"
            className="divide-y divide-[#edf0e9]"
          >
            {data.plots.map((p) => (
              <div
                key={p.plotId}
                className={`border-l-2 transition ${plot?.plotId === p.plotId ? 'border-l-[#416c48] bg-[#f4f7ef]' : 'border-l-transparent hover:bg-[#fafbf7]'}`}
              >
                <button
                  type="button"
                  aria-pressed={plot?.plotId === p.plotId}
                  onClick={() => setSelected(p.plotId)}
                  className="flex min-h-22 w-full items-center gap-3 px-4 py-4 text-left"
                >
                  <span
                    className={`flex size-10 shrink-0 items-center justify-center rounded-xl ${plot?.plotId === p.plotId ? 'bg-[#e5ecd9]' : 'bg-[#f1f4eb]'}`}
                  >
                    <svg
                      width="21"
                      height="21"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.2"
                      className="text-[#71835e]"
                      aria-hidden="true"
                    >
                      <path d="m3 7 8-4 10 4-8 5Zm0 5 10 5 8-5M3 17l10 5 8-5" />
                    </svg>
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold text-[#304b3b]">
                      {p.name}
                    </span>
                    <span className="mt-1 flex items-center gap-1.5 text-[11px] text-[#7b8772]">
                      <span
                        className={`size-1.5 rounded-full ${mapStatuses[mapStatus(p)].dot}`}
                      />
                      {p.cropType ?? 'Crop unavailable'} ·{' '}
                      {mapStatuses[mapStatus(p)].label}
                    </span>
                  </span>
                  <span aria-hidden="true" className="text-[#9aa68e]">
                    ›
                  </span>
                </button>
              </div>
            ))}
          </section>
          {plot && (
            <div
              className="border-t border-[#e4e9df] px-5 py-5"
              aria-live="polite"
            >
              <PlotDetails plot={plot} />
              <div className="mt-5 grid gap-2">
                {onOpenPlot && (
                  <button
                    type="button"
                    className="min-h-11 rounded-xl bg-[#294e38] px-4 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-[#1d3d29] disabled:opacity-45"
                    disabled={disabled}
                    onClick={() => onOpenPlot(plot.plotId)}
                    aria-label={`Open plot ${plot.name}`}
                  >
                    View plot
                    <svg
                      className="ml-2 inline-block"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      aria-hidden="true"
                    >
                      <path d="M7 17 17 7M7 7h10v10" />
                    </svg>
                  </button>
                )}
                {onEditBoundary && (
                  <button
                    type="button"
                    className="min-h-10 rounded-lg text-xs font-medium text-[#7a8a71] hover:bg-[#f4f7ef] disabled:opacity-45"
                    disabled={disabled}
                    onClick={() =>
                      onEditBoundary({ entity: 'plot', plotId: plot.plotId })
                    }
                    aria-label={`Edit boundary ${plot.name}`}
                  >
                    Edit boundary
                  </button>
                )}
              </div>
            </div>
          )}
        </aside>
      </div>
      <footer className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#939d8c]">
        <p>Made for a better view of your farm.</p>
        <p>
          Data checked {new Date(data.generatedAt ?? '').toLocaleString()} ·
          Freshness shown per plot
        </p>
      </footer>
    </div>
  );
}

export function FarmMapPage() {
  const query = useFarmMap();
  const selection = useMapSelection();
  const context = useFarmContext();
  const [editing, setEditing] = useState<BoundaryTarget | null>(null);
  if (editing && query.data)
    return (
      <Suspense fallback={<p role="status">Loading boundary editor…</p>}>
        <BoundaryEditPanel
          data={query.data}
          target={editing}
          onClose={() => setEditing(null)}
        />
      </Suspense>
    );
  if (context.status === 'error')
    return (
      <section className="grid gap-3">
        <h2 className="text-xl font-bold">Plot selection needs confirmation</h2>
        <p role="alert">
          {context.error?.message ??
            context.selectionError?.message ??
            'Plot selection could not be confirmed. Refresh the farm context before continuing.'}
        </p>
        <button
          type="button"
          className={mapButton}
          onClick={() => {
            void context.refresh().catch(() => {});
          }}
        >
          Refresh farm context
        </button>
      </section>
    );
  if (context.status !== 'ready')
    return (
      <p role="status">
        {context.status === 'switching'
          ? 'Opening plot…'
          : 'Confirming farm context…'}
      </p>
    );
  if (query.isError)
    return (
      <section className="grid gap-3">
        <h2 className="text-xl font-bold">Farm map unavailable</h2>
        <p role="alert">{query.error.message}</p>
        {query.data &&
          !(
            query.error instanceof ApiError && query.error.kind === 'context'
          ) && (
            <>
              <p className="text-sm text-amber-900">
                Showing saved boundaries. Irrigation data could not be
                refreshed.
              </p>
              <FarmMapContent
                data={staleMap(query.data)}
                onOpenPlot={selection.openPlot}
                onEditBoundary={setEditing}
                disabled={selection.disabled}
              />
            </>
          )}
        <button
          type="button"
          className={mapButton}
          onClick={() => {
            void (
              query.error instanceof ApiError && query.error.kind === 'context'
                ? context.refresh()
                : query.refetch()
            ).catch(() => {});
          }}
        >
          Retry
        </button>
      </section>
    );
  // Keep geometry usable during refresh, but never label the cached status current.
  if (!query.data) return <p role="status">Loading farm boundaries…</p>;
  return (
    <>
      {query.isFetching && (
        <p role="status" className="mb-3 text-sm">
          Refreshing irrigation data…
        </p>
      )}
      {(selection.error || context.selectionError) && (
        <p
          role="alert"
          className="rounded-lg bg-red-50 p-3 text-sm text-red-800"
        >
          {selection.error ?? context.selectionError?.message}
        </p>
      )}
      <FarmMapContent
        data={query.isFetching ? staleMap(query.data) : query.data}
        onOpenPlot={selection.openPlot}
        onEditBoundary={setEditing}
        disabled={selection.disabled}
      />
    </>
  );
}
