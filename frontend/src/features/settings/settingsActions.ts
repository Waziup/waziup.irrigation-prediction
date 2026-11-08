import { ApiError } from '../../api/errors';
import { trainingApi } from '../../api/trainingApi';
import { settingsApi } from '../../api/settingsApi';
import type { PlotSettingsInput } from '../../domain/settings';
import { type SettingsSnapshot } from './types';

export async function loadSettings(
  plotId: string,
  signal?: AbortSignal,
): Promise<SettingsSnapshot> {
  return settingsApi.getPlotSettings(plotId, signal ? { signal } : undefined);
}

export class SettingsSaveError extends Error {
  constructor(
    readonly original: unknown,
    readonly requiresReload: boolean,
    mayHaveCommitted = false,
  ) {
    super(
      `${original instanceof Error ? original.message : 'Settings could not be saved.'}${requiresReload ? ` Reload saved settings before another save.${mayHaveCommitted ? ' The request may have committed.' : ''}` : ''}`,
      { cause: original },
    );
    this.name = 'SettingsSaveError';
  }
}
export async function saveSettings(
  plotId: string,
  baseline: SettingsSnapshot,
  input: PlotSettingsInput,
) {
  if (baseline.plotId !== plotId || !baseline.configVersion)
    throw new ApiError('Reload settings for the intended plot before saving.', {
      kind: 'context',
    });
  const training = await trainingApi.getTrainingStatus(plotId);
  if (training.currentlyTraining || training.status === 'running')
    throw new Error('Wait for training to finish before saving settings.');
  let writeAcknowledged = false;
  try {
    const result = await settingsApi.savePlotSettings(
      plotId,
      input,
      baseline.configVersion,
    );
    writeAcknowledged = true;
    const saved = await loadSettings(plotId);
    if (!saved.configured)
      throw new Error('The saved configuration could not be verified.');
    if (result.configVersion !== saved.configVersion)
      throw new Error(
        'Settings changed again after saving. Reload to see the latest values.',
      );
    return { result, saved };
  } catch (error) {
    // Never automatically repeat a state-changing request after a lost response.
    const rejected =
      !writeAcknowledged &&
      error instanceof ApiError &&
      error.kind === 'http' &&
      error.status !== null &&
      error.status >= 400 &&
      error.status < 500;
    const conflict =
      rejected &&
      error instanceof ApiError &&
      error.status === 409 &&
      typeof error.payload === 'object' &&
      error.payload !== null &&
      'conflict' in error.payload &&
      error.payload.conflict === 'config_version';
    throw new SettingsSaveError(error, !rejected || conflict, !rejected);
  }
}
