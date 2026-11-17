import { ReadingCards } from './ReadingCards';
import type { SensorReadings } from '../../../domain/sensor';
import type { IrrigationRecommendation } from '../../../domain/recommendation';
import type { PhenologySummary } from '../../../domain/phenology';
import type { ResourceState } from '../types';
import { formatAmount } from '../../../shared/presentation';
export function CurrentReadings({
  state,
  recommendation,
  phenology,
}: {
  state: ResourceState<SensorReadings>;
  recommendation: ResourceState<IrrigationRecommendation>;
  phenology?: ResourceState<PhenologySummary>;
  onRetry: () => void;
}) {
  const threshold =
    phenology?.status === 'ready' && phenology.data.available
      ? phenology.data.thresholdActiveCbar
      : recommendation.status === 'ready'
        ? recommendation.data.available
          ? recommendation.data.condition.thresholdCbar
          : recommendation.data.thresholdCbar
        : null;
  const readings =
    state.status === 'ready' && state.data.available ? state.data : null;
  const value = (amount: number | null | undefined, unit: string) =>
    amount == null ? '—' : formatAmount(amount, unit);
  return (
    <section aria-labelledby="current-readings-title">
      <h2
        id="current-readings-title"
        className="mb-3 text-base font-semibold text-[#253247]"
      >
        Current readings
      </h2>
      <ReadingCards
        values={[
          value(readings?.tensionCbar, 'cbar'),
          value(threshold != null && threshold > 0 ? threshold : null, 'cbar'),
          value(readings?.temperatureC, '°C'),
          value(readings?.vwcPercent, '%'),
        ]}
      />
      {phenology?.status === 'ready' &&
        phenology.data.available &&
        phenology.data.thresholdMode === 'dynamic' &&
        phenology.data.thresholdActiveCbar == null && (
          <p role="status" className="mt-3 text-xs text-slate-500">
            Dynamic threshold unavailable. Check soil calibration and weather
            data.
          </p>
        )}
      {state.status === 'loading' ? (
        <p role="status" className="mt-3 text-xs text-slate-500">
          Loading current readings…
        </p>
      ) : !readings ? (
        <p role="status" className="mt-3 text-xs text-slate-500">
          Sensor readings unavailable.
        </p>
      ) : null}
    </section>
  );
}
