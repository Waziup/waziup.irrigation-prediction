import type { FarmPlotCard } from '../../../domain/farm';
import { StatusBadge } from '../../../components/StatusBadge';
import {
  formatAmount,
  formatDashboardTime,
  urgencyPresentation,
} from '../types';

const conditions = {
  unknown: 'Condition unavailable',
  within_threshold: 'Within threshold',
  at_threshold: 'At threshold',
  threshold_exceeded: 'Threshold exceeded',
} as const;
const equipmentStates = {
  active: 'Active operation',
  not_configured: 'Not configured',
  advisory_only: 'Advisory only',
  ready: 'Ready',
} as const;
export function PlotCard({
  plot,
  selected,
  onSelect,
  timezone,
  disabled = false,
}: {
  plot: FarmPlotCard;
  selected: boolean;
  onSelect: (plotId: string) => void;
  timezone: string;
  disabled?: boolean;
}) {
  const urgency = urgencyPresentation[plot.urgency];
  const unavailable = plot.dataState === 'unavailable';
  const margin = unavailable ? null : plot.thresholdMarginCbar;
  return (
    <article
      className={`grid min-w-0 items-center gap-3 rounded-xl border border-[#dce4ed] border-l-4 bg-white px-3.5 py-2.5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] ${selected ? 'border-l-[#2f7133]' : 'border-l-[#94a3b8]'}`}
    >
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-[#1f2937]">
          {plot.name || 'Unnamed plot'}
        </h3>
        <p className="mt-1 text-xs text-[#64748b]">
          {plot.cropType ?? '—'} · {plot.growthStage ?? '—'}
        </p>
        {selected && <span className="sr-only">Selected plot</span>}
        <p className="mt-1 text-xs text-[#64748b]">
          {unavailable
            ? 'Condition unavailable'
            : conditions[plot.conditionStatus]}{' '}
          · {equipmentStates[plot.equipment.state]}
        </p>
      </div>
      <div>
        <p className="text-sm font-semibold text-[#1f2937]">
          {plot.nextAction}
        </p>
        <p className="mt-1 text-xs text-[#64748b]">
          {margin === null
            ? '—'
            : `${formatAmount(Math.abs(margin), 'cbar')} ${margin < 0 ? 'over threshold' : 'remaining'}`}{' '}
          · {formatDashboardTime(plot.nextActionTime, timezone)}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2 xl:justify-end">
        <StatusBadge tone={urgency.tone}>{urgency.label}</StatusBadge>
        {plot.dataState !== 'fresh' && (
          <StatusBadge tone="warning">
            {unavailable ? 'Advice unavailable' : 'Advice needs updating'}
          </StatusBadge>
        )}
        {plot.alert && (
          <StatusBadge tone={urgencyPresentation[plot.alert.urgency].tone}>
            Active alert: {urgencyPresentation[plot.alert.urgency].label}
          </StatusBadge>
        )}
        <button
          type="button"
          className="min-h-9 cursor-pointer rounded-md px-2 text-xs font-semibold text-[#2f7133] hover:bg-[#e7f2e4] disabled:opacity-55"
          disabled={disabled}
          onClick={() => onSelect(plot.plotId)}
          aria-label={`Open plot ${plot.name || 'Unnamed plot'}`}
        >
          View plot →
        </button>
      </div>
      <details className="col-span-full text-xs text-[#64748b]">
        <summary className="w-fit cursor-pointer">Plot details</summary>
        <dl className="mt-3 grid gap-2 sm:grid-cols-2">
          <div>
            <dt>Soil tension</dt>
            <dd>
              {formatAmount(unavailable ? null : plot.tensionCbar, 'cbar')}
            </dd>
          </div>
          <div>
            <dt>Threshold margin</dt>
            <dd>
              {margin === null
                ? 'Unavailable'
                : `${formatAmount(Math.abs(margin), 'cbar')} ${margin < 0 ? 'over threshold' : 'remaining'}`}
            </dd>
          </div>
          <div>
            <dt>Next action time</dt>
            <dd>{formatDashboardTime(plot.nextActionTime, timezone)}</dd>
          </div>
          <div>
            <dt>Equipment</dt>
            <dd>{equipmentStates[plot.equipment.state]}</dd>
          </div>
          {plot.operation && (
            <div>
              <dt>Irrigation record</dt>
              <dd>
                {plot.operation.status.replaceAll('_', ' ')} ·{' '}
                {formatAmount(plot.operation.amountM3, 'm³', 2)}
              </dd>
            </div>
          )}
        </dl>
      </details>
    </article>
  );
}
