import { EmptyPlotOverview } from './components/EmptyPlotOverview';
import { useFarmContext } from '../farm-context/FarmContext';
import { usePlotMetrics } from './hooks/usePlotMetrics';
import { usePlotRecommendation } from './hooks/usePlotRecommendation';
import { usePlotQuery } from './hooks/usePlotQuery';
import { chartApi } from '../../api/chartApi';
import { alertApi } from '../../api/alertApi';
import { recommendationApi } from '../../api/recommendationApi';
import { resourceState } from './types';
import { CurrentReadings } from './components/CurrentReadings';
import { RecommendationCard } from './components/RecommendationCard';
import { ForecastChart } from './components/ForecastChart';
import { urgencyPresentation } from '../../shared/presentation';

export function PlotOverviewPage() {
  const context = useFarmContext();
  const metrics = usePlotMetrics();
  const recommendation = usePlotRecommendation();
  const history = usePlotQuery('history', (signal) =>
    chartApi.getCurrentHistory({ signal }),
  );
  const threshold = usePlotQuery('threshold', (signal) =>
    recommendationApi.getCurrentThreshold({ signal }),
  );
  const prediction = usePlotQuery('prediction', (signal) =>
    chartApi.getCurrentPrediction({ signal }),
  );
  const alerts = usePlotQuery(
    'alerts',
    (signal) => alertApi.getCurrentAlertStatus({ signal }),
    30_000,
  );
  if (context.status !== 'ready' || !context.currentPlot || !context.farm)
    return <EmptyPlotOverview />;
  const recommendationState = resourceState(recommendation);
  const metricsState = resourceState(metrics);
  const historyState = resourceState(history);
  const predictionState = resourceState(prediction);
  const retryAll = () => {
    for (const query of [
      metrics,
      recommendation,
      history,
      prediction,
      threshold,
      alerts,
    ])
      void query.refetch();
  };
  const alertState = resourceState(alerts);
  const hasActiveAlert =
    alertState.status === 'ready' &&
    alertState.data.available &&
    ['watch', 'advise', 'critical'].includes(alertState.data.urgency);
  return (
    <div className="grid gap-5">
      {hasActiveAlert && (
        <div
          className="rounded-xl border border-[#eed9c3] bg-[#fffaf3] px-5 py-4 text-sm text-[#906035]"
          role="status"
        >
          <p className="font-semibold">
            {alertState.status === 'ready' && alertState.data.available
              ? urgencyPresentation[alertState.data.urgency].label
              : 'Plot alert'}
          </p>
          {alertState.status === 'ready' &&
            alertState.data.available &&
            alertState.data.notifyMessage && (
              <p className="mt-1">{alertState.data.notifyMessage}</p>
            )}
        </div>
      )}
      <RecommendationCard
        plotId={context.currentPlot.id}
        state={recommendationState}
        timezone={context.farm.timezone}
        onRetry={() => void recommendation.refetch()}
      />
      <CurrentReadings
        state={metricsState}
        recommendation={recommendationState}
        onRetry={retryAll}
      />
      <ForecastChart
        key={context.currentPlot.id}
        threshold={resourceState(threshold)}
        history={historyState}
        prediction={predictionState}
        timezone={context.farm.timezone}
        onRetry={retryAll}
      />
    </div>
  );
}
