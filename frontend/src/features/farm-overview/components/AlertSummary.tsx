import type { AlertSummary as Alert } from '../../../domain/alert';
import type { FarmPlotCard } from '../../../domain/farm';
import { StatusBadge } from '../../../components/StatusBadge';
import { urgencyPresentation } from '../types';

export function AlertSummary({
  alerts,
  plots,
}: {
  alerts: Alert[];
  plots: FarmPlotCard[];
}) {
  const names = new Map(plots.map((plot) => [plot.plotId, plot.name]));
  return (
    <section
      className="min-w-0 rounded-xl border border-[#e0e6d9] bg-white p-5"
      aria-labelledby="farm-alerts-title"
    >
      <h2
        className="mb-3 text-sm font-semibold text-[#304b3b]"
        id="farm-alerts-title"
      >
        Active alerts{' '}
        <span className="font-normal text-gray-600">({alerts.length})</span>
      </h2>
      {alerts.length ? (
        <ul className="m-0 grid list-none gap-3 p-0 [&>li]:flex [&>li]:items-baseline [&>li]:justify-between [&>li]:gap-3 [&>li]:border-t [&>li]:border-gray-200 [&>li]:pt-3 [&>li]:wrap-anywhere [&_p]:my-1 [&_p]:text-sm">
          {alerts.map((alert) => (
            <li key={alert.id}>
              <span>{names.get(alert.plotId) ?? 'Unknown plot'}</span>
              <StatusBadge tone={urgencyPresentation[alert.urgency].tone}>
                {urgencyPresentation[alert.urgency].label}
              </StatusBadge>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs leading-relaxed text-[#7c8876]">
          No active alerts.
        </p>
      )}
    </section>
  );
}
