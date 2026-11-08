import { sensorApi } from '../../../api/sensorApi';
import { usePlotQuery } from './usePlotQuery';
export function usePlotMetrics() {
  return usePlotQuery(
    'metrics',
    (signal) => sensorApi.getCurrentReadings({ signal }),
    30_000,
  );
}
