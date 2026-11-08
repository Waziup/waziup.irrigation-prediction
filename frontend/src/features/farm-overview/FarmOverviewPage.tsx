import { lazy, Suspense } from 'react';
import { useFarmMap } from '../farm-map/hooks/useFarmMap';
import { MapLegend } from '../farm-map/MapLegend';
import { dashboardMap } from '../farm-map/dashboardMap';
const FarmMap = lazy(() =>
  import('../farm-map/FarmMap').then((module) => ({ default: module.FarmMap })),
);
import { EmptyFarmOverview } from './components/EmptyFarmOverview';
import { useNavigate } from 'react-router';
import { ApiError } from '../../api/errors';
import { FarmWeather } from './components/FarmWeather';
import { useFarmContext } from '../farm-context/FarmContext';
import { useFarmDashboard } from './hooks/useFarmDashboard';
import type { FarmOverviewContentProps } from './types';
import { PlotList } from './components/PlotList';
import { AlertSummary } from './components/AlertSummary';
import { DataFreshnessNotice } from './components/DataFreshnessNotice';
import { IrrigationPlanSummary } from './components/IrrigationPlanSummary';

export function FarmOverviewContent({
  dashboard,
  currentPlotId,
  onSelectPlot,
  onConfigurePlot,
  disabled,
  farmMap,
  mapError,
  weather,
}: FarmOverviewContentProps) {
  let mapped = farmMap;
  let boundaryError = mapError;
  if (mapped) {
    try {
      mapped = dashboardMap(mapped, dashboard);
    } catch (cause) {
      mapped = undefined;
      boundaryError =
        cause instanceof Error ? cause.message : 'Farm boundaries unavailable.';
    }
  }
  return (
    <div className="grid gap-5">
      <header className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-[#e0e8da] bg-[#edf3e6] px-5 py-5 sm:px-6">
        <div>
          <p className="mb-1.5 text-[11px] font-semibold tracking-[.15em] text-[#7b8b71] uppercase">
            Farm overview
          </p>
          <h2 className="text-2xl font-semibold tracking-tight text-[#233f2d]">
            {dashboard.farm.name}
          </h2>
          <p className="mt-2 text-xs text-[#76846e]">
            {dashboard.summary.totalPlots}{' '}
            {dashboard.summary.totalPlots === 1 ? 'plot' : 'plots'}
          </p>
        </div>
        <a
          href="#/settings"
          className="rounded-lg border border-[#d1ddc9] bg-white/80 px-4 py-2.5 text-xs font-medium text-[#42613d] no-underline hover:bg-white"
        >
          Manage farm
        </a>
      </header>
      {dashboard.plots
        .filter((plot) => plot.setupState === 'settings_required')
        .map((plot) => (
          <section
            key={plot.plotId}
            className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#dce5d7] bg-white px-5 py-4"
          >
            <div>
              <h3 className="text-sm font-semibold">
                Finish setting up {plot.name}
              </h3>
              <p className="mt-1 text-sm text-[#53645a]">
                Configure sensors, soil and irrigation in Plot settings.
              </p>
            </div>
            <button
              type="button"
              disabled={disabled}
              className="rounded-lg bg-[#356b2f] px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50"
              onClick={() => onConfigurePlot?.(plot.plotId)}
            >
              Configure plot
            </button>
          </section>
        ))}
      {dashboard.activeAlerts.length > 0 ? (
        <AlertSummary alerts={dashboard.activeAlerts} plots={dashboard.plots} />
      ) : (
        <p className="-mt-2 text-xs text-[#7d8975]">No active alerts.</p>
      )}
      <DataFreshnessNotice
        count={dashboard.summary.staleOrUnavailablePlots}
        plots={dashboard.plots}
      />
      <div className="grid min-w-0 items-start gap-5 min-[1100px]:grid-cols-[minmax(0,1.7fr)_minmax(260px,1fr)]">
        <section
          aria-label="Farm map"
          className="overflow-hidden rounded-2xl border border-[#dde5d8] bg-white shadow-sm shadow-[#233f2d]/3"
        >
          <div className="flex items-center justify-between gap-3 border-b border-[#e7ebe2] px-5 py-3.5">
            <h3 className="text-sm font-semibold text-[#304b3b]">Your farm</h3>
            <a
              href="#/settings/boundaries"
              className="text-xs font-medium text-[#7a8c70] no-underline hover:text-[#294e38]"
            >
              Manage boundaries ↗
            </a>
          </div>
          {mapped ? (
            <Suspense
              fallback={
                <p role="status" className="p-6 text-sm text-[#7b8873]">
                  Loading farm map…
                </p>
              }
            >
              <FarmMap
                compact
                data={mapped}
                selected={currentPlotId}
                onInspect={(id) => {
                  if (!disabled) onSelectPlot(id);
                }}
              />
            </Suspense>
          ) : (
            <p
              role="status"
              className="flex min-h-72 items-center justify-center p-6 text-sm text-[#7b8873]"
            >
              {boundaryError ?? 'Loading farm boundaries…'}
            </p>
          )}
          <div className="border-t border-[#e7ebe2] px-5 py-3">
            <MapLegend />
          </div>
        </section>
        <IrrigationPlanSummary
          operations={dashboard.todaysPlan}
          plots={dashboard.plots}
          timezone={dashboard.farm.timezone}
        />
      </div>
      {weather}
      <details className="border-t border-[#e2e8dd] pt-4 text-sm text-[#5c7154]">
        <summary className="cursor-pointer font-medium">All plots</summary>
        <div className="mt-4">
          <PlotList
            plots={dashboard.plots}
            currentPlotId={currentPlotId}
            onSelectPlot={onSelectPlot}
            timezone={dashboard.farm.timezone}
            disabled={disabled ?? false}
          />
        </div>
      </details>
    </div>
  );
}

export function FarmOverviewPage() {
  const context = useFarmContext();
  const dashboard = useFarmDashboard();
  const geometry = useFarmMap();
  const navigate = useNavigate();
  if (context.status === 'error')
    return (
      <section className="grid gap-3">
        <h2 className="text-xl font-semibold">
          {context.selectionError
            ? 'Plot selection needs confirmation'
            : 'Farm overview unavailable'}
        </h2>
        <p role="alert">
          {context.error?.message ??
            context.selectionError?.message ??
            'Refresh the farm context before continuing.'}
        </p>
        <button
          type="button"
          className="w-fit rounded-lg border border-[#dce4dc] bg-white px-4 py-2 text-sm text-[#244636]"
          onClick={() => {
            void context.refresh().catch(() => {});
          }}
        >
          Refresh farm context
        </button>
      </section>
    );
  if (context.status === 'switching') return <p role="status">Opening plot…</p>;
  if (context.status !== 'ready') return <EmptyFarmOverview />;
  if (dashboard.isError)
    return (
      <>
        <section className="min-w-0 rounded-xl border border-gray-200 bg-white p-5 shadow-xs sm:p-8 [&>h2]:mt-0">
          <h2 className="mt-4 mb-2.5 text-xl font-semibold">
            Farm overview unavailable
          </h2>
          <p className="my-3 leading-relaxed text-gray-600" role="alert">
            {dashboard.error.message}
          </p>
          <button
            type="button"
            className="min-h-11 cursor-pointer rounded-lg border border-[#aac9a5] bg-[#f5faf4] px-4 py-2.5 font-semibold text-[#2d5a27] hover:bg-[#e8f5e6] disabled:cursor-wait disabled:opacity-65"
            onClick={() => {
              void (
                dashboard.error instanceof ApiError &&
                dashboard.error.kind === 'context'
                  ? context.refresh()
                  : dashboard.refetch()
              ).catch(() => {});
            }}
          >
            Retry
          </button>
        </section>
      </>
    );
  // Clear old totals/cards on refresh or failure rather than relabelling a cached
  // response as current. Query cancellation prevents superseded responses winning.
  if (dashboard.isPending || dashboard.isFetching || !dashboard.data)
    return <EmptyFarmOverview />;
  return (
    <>
      {context.selectionError && (
        <p
          role="alert"
          className="mb-4 rounded-lg bg-red-50 p-3 text-sm text-red-800"
        >
          {context.selectionError.message}
        </p>
      )}
      <FarmOverviewContent
        dashboard={dashboard.data}
        {...(geometry.data ? { farmMap: geometry.data } : {})}
        {...(geometry.isError ? { mapError: geometry.error.message } : {})}
        weather={<FarmWeather />}
        currentPlotId={context.currentPlot?.id ?? ''}
        disabled={context.status !== 'ready'}
        onConfigurePlot={(plotId) => {
          void context
            .selectPlot(plotId)
            .then(() => navigate('/settings'))
            .catch(() => {
              /* Selection errors are displayed by the shell. */
            });
        }}
        onSelectPlot={(plotId) => {
          void context
            .selectPlot(plotId)
            .then(() => navigate('/plot'))
            .catch(() => {
              /* The shell displays the persisted-selection error. */
            });
        }}
      />
    </>
  );
}
