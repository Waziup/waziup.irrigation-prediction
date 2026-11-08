import {
  adviceSources,
  adviceLimitations,
} from '../../../shared/adviceWarnings';
import type { IrrigationRecommendation } from '../../../domain/recommendation';
import type { ResourceState } from '../types';
import { PumpControls } from './PumpControls';
import { ResourcePanel } from './ResourcePanel';
import { StatusBadge } from '../../../components/StatusBadge';
import {
  formatAmount,
  formatDashboardTime,
  urgencyPresentation,
} from '../../../shared/presentation';
export function RecommendationCard({
  state,
  timezone,
  onRetry,
  plotId,
}: {
  state: ResourceState<IrrigationRecommendation>;
  timezone: string;
  onRetry: () => void;
  plotId?: string;
}) {
  if (state.status !== 'ready' || !state.data.available) {
    return (
      <section
        aria-label="Today's plot status"
        aria-busy={state.status === 'loading'}
        className="min-w-0 rounded-2xl border border-[#c0d2ef] bg-[#edf4ff] p-5 sm:p-8"
      >
        <h2 className="text-xs font-bold uppercase tracking-wide text-[#5b7090]">
          Today’s plot status
        </h2>
        <p className="my-5 text-2xl font-semibold text-[#253247]">--</p>
        <dl className="grid grid-cols-1 gap-4 sm:grid-cols-3 [&>div]:min-w-0 [&>div]:rounded-xl [&>div]:border [&>div]:border-[#c0d2ef] [&>div]:bg-white/70 [&>div]:p-4 [&_dt]:text-xs [&_dt]:text-[#5b7090] [&_dd]:mt-3 [&_dd]:text-xl [&_dd]:font-semibold">
          <div>
            <dt>Calculated water to apply</dt>
            <dd>--</dd>
          </div>
          <div>
            <dt>When to irrigate</dt>
            <dd>--</dd>
          </div>
          {plotId && (
            <div>
              <dt className="sr-only">Water pump controls</dt>
              <dd>
                <PumpControls key={plotId} plotId={plotId} embedded />
              </dd>
            </div>
          )}
        </dl>
      </section>
    );
  }
  return (
    <ResourcePanel title="Today’s plot status" state={state} onRetry={onRetry}>
      {(recommendation) => {
        if (!recommendation.available) return null;
        const urgency = urgencyPresentation[recommendation.action.urgency];
        const staleSources = adviceSources(
          recommendation.freshness.staleSources,
        );
        const limitations = adviceLimitations(recommendation.limitations);
        return (
          <>
            <StatusBadge tone={urgency.tone}>{urgency.label}</StatusBadge>
            {recommendation.freshness.stale &&
              (staleSources.length > 0 ||
                recommendation.freshness.staleSources.length === 0) && (
                <p
                  className="m-0 rounded-md border-l-3 border-[#b88000] bg-[#fff8e1] p-4 text-[#775300]"
                  role="status"
                >
                  Stale source data:{' '}
                  {staleSources.join(', ') || 'Source not specified'}.
                </p>
              )}
            {recommendation.action.urgency !== 'none' && (
              <h3 className="text-lg font-semibold">
                {recommendation.action.label}
              </h3>
            )}
            <p className="my-3 text-xs leading-relaxed text-gray-600">
              {recommendation.reason ?? recommendation.condition.message}
            </p>
            <dl className="my-4 grid grid-cols-1 sm:grid-cols-3 gap-5 [&_dt]:text-xs [&_dt]:text-gray-600 [&_dd]:mt-2 [&_dd]:font-semibold [&_dd]:wrap-anywhere">
              <div>
                <dt>Calculated water to apply</dt>
                <dd>
                  {formatAmount(
                    recommendation.water.recommendedVolumeM3,
                    'm³',
                    2,
                  )}
                </dd>
              </div>
              <div>
                <dt>When to irrigate</dt>
                <dd>
                  {formatDashboardTime(
                    recommendation.timing.firstBreachTimestamp ??
                      recommendation.timing.firstBreachHorizon,
                    timezone,
                  )}
                </dd>
              </div>
              {plotId && (
                <div>
                  <dt className="sr-only">Water pump controls</dt>
                  <dd>
                    <PumpControls key={plotId} plotId={plotId} embedded />
                  </dd>
                </div>
              )}
            </dl>
            {limitations.length > 0 && (
              <ul className="list-disc pl-5">
                {limitations.map((limitation, index) => (
                  <li key={`${index}:${limitation}`}>{limitation}</li>
                ))}
              </ul>
            )}
          </>
        );
      }}
    </ResourcePanel>
  );
}
