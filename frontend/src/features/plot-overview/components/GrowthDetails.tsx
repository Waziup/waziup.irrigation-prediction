import type { PhenologySummary } from '../../../domain/phenology';
import type { IrrigationRecommendation } from '../../../domain/recommendation';
import { DataSection } from '../../../components/DataSection';
import { formatAmount } from '../../../shared/presentation';
import type { ReactNode } from 'react';

type Summary = Extract<PhenologySummary, { available: true }>;
type Advice = Extract<IrrigationRecommendation, { available: true }>;
export function GrowthDetails({
  summary,
  advice,
  children,
}: {
  summary: Summary | null;
  advice: Advice | null;
  children?: ReactNode;
}) {
  return (
    <>
      <section
        aria-label="Growth metrics"
        className="grid grid-cols-1 gap-3 sm:grid-cols-2"
      >
        {[
          [
            'Threshold',
            formatAmount(
              summary
                ? summary.thresholdActiveCbar
                : (advice?.condition.thresholdCbar ?? null),
              'cbar',
            ),
            summary
              ? summary.thresholdMode === 'dynamic'
                ? 'Dynamic threshold'
                : 'Static field trigger'
              : '',
          ],
          [
            'Kc today',
            formatAmount(advice?.crop.kc ?? null, '', 3),
            summary?.currentStage ?? '',
          ],
          [
            'ETc today',
            formatAmount(advice?.weather.etcDailyMm ?? null, 'mm'),
            'Crop water need',
          ],
          [
            'GDD today',
            formatAmount(
              summary?.currentGdd ?? advice?.crop.gddCumulative ?? null,
              '',
              1,
            ),
            'Cumulative heat units',
          ],
        ].map(([label, value, detail]) => (
          <div
            key={label}
            className="min-w-0 rounded-2xl border border-[#e0e0e0] bg-white p-5"
          >
            <h3 className="text-xs font-semibold uppercase text-[#64748b]">
              {label}
            </h3>
            <p className="mt-3 text-2xl font-semibold text-[#111827]">
              {value === 'Unavailable' ? '—' : value}
            </p>
            {detail && <p className="mt-2 text-xs text-[#64748b]">{detail}</p>}
          </div>
        ))}
      </section>
      {children}
      <DataSection title="Phenology timeline">
        {summary?.stages.length ? (
          <ol
            aria-label="Crop stages"
            className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5"
          >
            {summary.stages.map((stage) => (
              <li
                key={stage.stage}
                aria-current={
                  stage.stage === summary.currentStage ? 'step' : undefined
                }
                className="rounded-xl border border-[#d8dee8] px-4 py-3 text-sm text-[#52627a] aria-[current=step]:border-[#315f48] aria-[current=step]:bg-[#e7f2e4] aria-[current=step]:font-semibold aria-[current=step]:text-[#174f2a]"
              >
                {stage.stage}
              </li>
            ))}
          </ol>
        ) : (
          <ol
            aria-label="Crop stages"
            className="grid grid-cols-2 gap-2 sm:grid-cols-5"
          >
            {[
              'Pre-emergence',
              'Development',
              'Mid-season',
              'Late-season',
              'Post-maturity',
            ].map((stage) => (
              <li
                key={stage}
                className="rounded-full bg-slate-100 px-3 py-3 text-center text-xs text-slate-600"
              >
                {stage}
              </li>
            ))}
          </ol>
        )}
      </DataSection>
      <DataSection title="Crop development reference">
        <div className="max-w-full overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                {['Stage', 'GDD range', 'Kc range', 'Tension trigger'].map(
                  (label) => (
                    <th
                      key={label}
                      scope="col"
                      className="border-b border-gray-200 p-3 text-left whitespace-nowrap"
                    >
                      {label}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {summary?.stages.length ? (
                summary.stages.map((stage) => (
                  <tr
                    key={stage.stage}
                    aria-current={
                      stage.stage === summary.currentStage ? 'step' : undefined
                    }
                    className="aria-[current=step]:bg-[#e8f5e6] aria-[current=step]:font-semibold"
                  >
                    <th
                      scope="row"
                      className="border-b border-gray-200 p-3 text-left whitespace-nowrap"
                    >
                      {stage.stage}
                    </th>
                    <td className="border-b border-gray-200 p-3 whitespace-nowrap">
                      {stage.gddStart} – {stage.gddEnd ?? 'onward'}
                    </td>
                    <td className="border-b border-gray-200 p-3 whitespace-nowrap">
                      {stage.kcStart} – {stage.kcEnd}
                    </td>
                    <td className="border-b border-gray-200 p-3 whitespace-nowrap">
                      {stage.thresholdSource === 'inactive_season'
                        ? 'Not applicable'
                        : formatAmount(stage.thresholdCbar, 'cbar')}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="p-3 text-gray-500">
                    —
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </DataSection>
    </>
  );
}
