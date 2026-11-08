import { growthDays } from './growthPresentation';
import { EmptyGrowthOverview } from './components/EmptyPlotOverview';
import { useFarmContext } from '../farm-context/FarmContext';
import { phenologyApi } from '../../api/phenologyApi';
import { usePlotSatellite } from '../earth-observation/usePlotSatellite';
import { usePlotQuery } from './hooks/usePlotQuery';
import { usePlotRecommendation } from './hooks/usePlotRecommendation';
import { resourceState } from './types';
import { EoPanel } from './components/EoPanel';
import { GrowthDetails } from './components/GrowthDetails';

export function GrowthOverviewPage() {
  const context = useFarmContext();
  const phenology = usePlotQuery('phenology', (signal) =>
    phenologyApi.getCurrentSummary({ signal }),
  );
  const eo = usePlotSatellite(
    context.currentPlot?.id,
    context.status === 'ready',
    context.currentPlot?.coordinates,
  );
  const recommendation = usePlotRecommendation();
  if (context.status !== 'ready' || !context.currentPlot || !context.farm)
    return <EmptyGrowthOverview />;
  const summary =
    phenology.isSuccess && !phenology.isFetching && phenology.data.available
      ? phenology.data
      : null;
  const advice =
    recommendation.isSuccess &&
    !recommendation.isFetching &&
    recommendation.data.available
      ? recommendation.data
      : null;
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <h2 className="text-base font-semibold text-[#253247]">
          Growth stage timeline
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
                    context.farm.timezone,
                  ) ?? '—')
                : '—'}
            </dd>
          </div>
        </dl>
      </div>
      {phenology.isFetching && (
        <p className="text-xs text-slate-500" role="status">
          Loading growth stage timeline…
        </p>
      )}
      {(phenology.isError || recommendation.isError) && (
        <p className="text-xs text-slate-500" role="status">
          Some growth data couldn’t be loaded.{' '}
          <button
            type="button"
            className="underline"
            onClick={() => {
              void phenology.refetch();
              void recommendation.refetch();
            }}
          >
            Refresh growth data
          </button>
        </p>
      )}
      <GrowthDetails summary={summary} advice={advice}>
        {advice?.freshness.stale && (
          <p
            role="status"
            className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900"
          >
            Stale source data:{' '}
            {advice.freshness.staleSources.join(', ') || 'Source not specified'}
            .
          </p>
        )}
        <EoPanel
          state={resourceState(eo)}
          timezone={context.farm.timezone}
          onRetry={() => void eo.refetch()}
        />
      </GrowthDetails>
    </div>
  );
}
