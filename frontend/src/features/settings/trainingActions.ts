import { trainingApi } from '../../api/trainingApi';
import { loadSettings } from './settingsActions';
import { sameSettings, type SettingsSnapshot } from './types';
export async function startPlotTraining(
  plotId: string,
  baseline: SettingsSnapshot,
) {
  const latest = await loadSettings(plotId);
  if (
    !sameSettings(baseline, latest) ||
    !latest.configured ||
    !latest.settings.trainingReadiness.ready
  )
    throw new Error(
      'Saved settings changed or are not ready. Reload settings before training.',
    );
  const current = await trainingApi.getTrainingStatus(plotId);
  if (current.currentlyTraining || current.status === 'running')
    throw new Error('Training is already running.');
  if (current.status === 'finished' || current.isTrainingFinished)
    throw new Error(
      'This plot is already trained. Predictions update automatically.',
    );
  return trainingApi.startTraining(plotId);
}
