import { ReadingCards } from './ReadingCards';
import type { SensorReadings } from '../../../domain/sensor';
import type { IrrigationRecommendation } from '../../../domain/recommendation';
import type { ResourceState } from '../types';
import { formatAmount } from '../../../shared/presentation';
export function CurrentReadings({
  state,
  recommendation,
}: {
  state: ResourceState<SensorReadings>;
  recommendation: ResourceState<IrrigationRecommendation>;
  onRetry: () => void;
}) {
  const threshold =
    recommendation.status === 'ready'
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
