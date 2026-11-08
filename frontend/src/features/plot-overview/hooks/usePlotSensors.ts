import { settingsApi } from '../../../api/settingsApi';
import { sensorApi } from '../../../api/sensorApi';
import { ApiError } from '../../../api/errors';
import { usePlotQuery } from './usePlotQuery';
export async function loadPlotSettings(signal: AbortSignal, plotId: string) {
  const result = await settingsApi.getPlotSettings(plotId, { signal });
  signal.throwIfAborted();
  if (result.configured && result.settings.plotId !== plotId)
    throw new ApiError(
      'The returned settings belong to another plot. Refresh farm context before continuing.',
      { kind: 'context' },
    );
  return result;
}
export function usePlotSensors() {
  const settings = usePlotQuery('settings', loadPlotSettings);
  const devices = usePlotQuery('devices', (signal) =>
    sensorApi.getDevices({ signal }),
  );
  return { settings, devices };
}
