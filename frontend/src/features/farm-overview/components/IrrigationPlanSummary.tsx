import { useEffect, useState } from 'react';
import type { OperationSummary } from '../../../domain/operation';
import type { FarmPlotCard } from '../../../domain/farm';
import { StatusBadge } from '../../../components/StatusBadge';
import { formatAmount, formatDashboardTime } from '../types';

export function IrrigationPlanSummary({
  operations,
  plots,
  timezone,
}: {
  operations: OperationSummary[];
  plots: FarmPlotCard[];
  timezone: string;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(timer);
  }, []);
  const names = new Map(plots.map((plot) => [plot.plotId, plot.name]));
  const byPlot = new Map<string, OperationSummary[]>();
  const seen = new Set<string>();
  for (const operation of operations) {
    const signature = JSON.stringify([
      operation.plotId,
      operation.status,
      operation.amountM3,
      operation.plannedStart,
      operation.plannedEnd,
    ]);
    if (seen.has(operation.id) || seen.has(signature)) continue;
    seen.add(operation.id);
    seen.add(signature);
    const entries = byPlot.get(operation.plotId) ?? [];
    entries.push(operation);
    byPlot.set(operation.plotId, entries);
  }
  const labels = {
    planned: 'Recommended',
    pending_approval: 'Awaiting approval',
    approved: 'Approved',
    active: 'Irrigation in progress',
    completed: 'Completed',
    verified: 'Water delivery confirmed',
    declined: 'Declined',
    failed: 'Failed',
  } as const;
  return (
    <section
      className="min-w-0 rounded-xl border border-[#e0e6d9] bg-white p-5"
      aria-labelledby="farm-plan-title"
    >
      <h2
        className="mb-3 text-sm font-semibold text-[#304b3b]"
        id="farm-plan-title"
      >
        Today’s irrigation
      </h2>
      {operations.length ? (
        <ul className="m-0 grid list-none gap-4 p-0">
          {[...byPlot].map(([plotId, entries]) => (
            <li key={plotId} className="border-t border-gray-200 pt-3">
              <h3 className="text-sm font-semibold">
                {names.get(plotId) ?? 'Unknown plot'}
              </h3>
              <ul className="mt-2 grid list-none gap-3 p-0">
                {entries.map((operation) => (
                  <li
                    key={operation.id}
                    className="flex items-start justify-between gap-3"
                  >
                    <p className="text-xs leading-relaxed text-[#7c8876]">
                      {formatAmount(operation.amountM3, 'm³', 2)} ·{' '}
                      {operation.plannedStart ? (
                        <>
                          {operation.source === 'schedule'
                            ? 'Scheduled start: '
                            : 'Recommended time: '}
                          {operation.source === 'recommendation' &&
                          operation.recommendationUrgency === 'critical' &&
                          ['planned', 'pending_approval', 'approved'].includes(
                            operation.status,
                          ) &&
                          new Date(operation.plannedStart).getTime() <= now
                            ? 'Now'
                            : formatDashboardTime(
                                operation.plannedStart,
                                timezone,
                              )}
                        </>
                      ) : (
                        'Timing unavailable'
                      )}
                    </p>
                    <StatusBadge
                      tone={
                        operation.status === 'failed' ? 'danger' : 'neutral'
                      }
                    >
                      {operation.status === 'planned' &&
                      operation.source === 'schedule'
                        ? 'Scheduled'
                        : labels[operation.status]}
                    </StatusBadge>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs leading-relaxed text-[#7c8876]">
          No irrigation planned for today.
        </p>
      )}
    </section>
  );
}
