import type { FarmDashboard } from '../../../domain/farm';
import type { OperationSummary } from '../../../domain/operation';
import { formatAmount } from '../types';
export function FarmSummary({
  summary,
  plan,
}: {
  summary: FarmDashboard['summary'] | null;
  plan: OperationSummary[] | null;
}) {
  const scheduled = plan?.filter(
    (item) => !['declined', 'failed'].includes(item.status),
  ).length;
  return (
    <section
      aria-label="Farm totals"
      className="grid grid-cols-2 gap-2.5 min-[600px]:grid-cols-3 xl:grid-cols-4"
    >
      {[
        ['Total plots', summary?.totalPlots],
        ['Need attention', summary?.plotsNeedingAttention],
        ['Equipment configured', summary?.equipmentConfigured],
      ].map(([label, value]) => (
        <div
          key={label}
          className="grid min-h-24 content-start gap-4 rounded-[13px] border border-[#e0e6d9] bg-white px-4 py-4"
        >
          <span className="text-[11px] font-bold uppercase tracking-wide text-[#64748b]">
            {label}
          </span>
          <strong
            className={`text-xl font-semibold ${label === 'Need attention' ? 'text-[#b45309]' : 'text-[#304b3b]'}`}
          >
            {value ?? '—'}
          </strong>
        </div>
      ))}
      <div className="col-span-2 grid min-h-24 content-start gap-2 rounded-[13px] border border-[#e0e6d9] bg-white px-4 py-4 min-[600px]:col-span-1">
        <span className="text-[11px] font-bold uppercase tracking-wide text-[#64748b]">
          Scheduled irrigations
        </span>
        <strong className="text-xl font-semibold text-[#304b3b]">
          {scheduled ?? '—'}
        </strong>
        <small className="text-xs font-semibold text-[#64748b]">
          <span className="sr-only">Planned water: </span>
          {summary ? formatAmount(summary.plannedWaterM3, 'm³', 2) : '—'}{' '}
          planned
        </small>
      </div>
    </section>
  );
}
