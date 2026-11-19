import { HelpTooltip } from '../../components/HelpTooltip';
import { growthDays } from './growthPresentation';
import { EmptyGrowthOverview } from './components/EmptyPlotOverview';
import { useFarmContext } from '../farm-context/FarmContext';
import { phenologyApi } from '../../api/phenologyApi';
import { usePlotSatellite } from '../earth-observation/usePlotSatellite';
import { usePlotQuery } from './hooks/usePlotQuery';
import { resourceState } from './types';
import { EoPanel } from './components/EoPanel';
import { GrowthDetails } from './components/GrowthDetails';

export function GrowthOverviewPage() {
  const context = useFarmContext();
  const displayed =
    context.snapshot ??
    (context.status === 'refreshing' ? context.lastConfirmedSnapshot : null);
  const plot =
    context.currentPlot ??
    displayed?.plots.find((p) => p.id === displayed.currentPlotId);
  const farm = context.farm ?? displayed?.farm;
  const phenology = usePlotQuery('phenology', (signal, plotId) =>
    phenologyApi.getCurrentSummary({ signal }, plotId),
  );
  const eo = usePlotSatellite(
    plot?.id,
    context.status === 'ready',
    plot?.coordinates,
  );
  if (
    (context.status !== 'ready' && context.status !== 'refreshing') ||
    !plot ||
    !farm
  )
    return <EmptyGrowthOverview />;
  const summary = phenology.data?.available ? phenology.data : null;
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <h2 className="flex items-center gap-2 text-base font-semibold text-[#253247]">
          <span>Growth stage timeline</span>
          <HelpTooltip
            label="Growth stage timeline"
            text="The app estimates crop growth from the planting date and accumulated heat (GDD). Growth stage helps calculate the crop water-use factor (Kc) and irrigation advice."
          />
        </h2>
        <dl className="flex flex-wrap gap-5 text-xs [&_dt]:text-[10px] [&_dt]:uppercase [&_dt]:text-slate-500 [&_dd]:mt-1 [&_dd]:font-semibold">
          <div>
            <dt>Crop</dt>
            <dd>{summary?.cropName ?? 'Not specified'}</dd>
          </div>
          <div>
            <dt>Planting date</dt>
            <dd>{summary?.plantingDate ?? 'Not available'}</dd>
          </div>
          <div>
            <dt>Days since planting</dt>
            <dd>
              {summary
                ? (growthDays(
                    summary.plantingDate,
                    summary.evaluatedAt,
                    farm.timezone,
                  ) ?? '—')
                : '—'}
            </dd>
          </div>
        </dl>
      </div>
      {(phenology.isFetching ||
        (phenology.data &&
          !phenology.data.available &&
          phenology.data.loading)) && (
        <p className="text-xs text-slate-500" role="status">
          Loading growth stage timeline…
        </p>
      )}
      {phenology.data &&
        !phenology.data.available &&
        !phenology.data.loading && (
          <p role="status" className="text-xs text-slate-500">
            {phenology.data.reason}
          </p>
        )}
      {phenology.isError && (
        <p className="text-xs text-slate-500" role="status">
          Some growth data couldn’t be loaded.{' '}
          <button
            type="button"
            className="underline"
            onClick={() => {
              void phenology.refetch();
            }}
          >
            Refresh growth data
          </button>
        </p>
      )}
      <GrowthDetails summary={summary} advice={null}>
        <EoPanel
          state={resourceState(eo)}
          timezone={farm.timezone}
          onRetry={() => void eo.refetch()}
        />
      </GrowthDetails>
    </div>
  );
}
