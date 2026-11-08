import { recommendationApi } from '../../../api/recommendationApi';
import { ApiError } from '../../../api/errors';
import { usePlotQuery } from './usePlotQuery';
export async function loadPlotRecommendation(
  signal: AbortSignal,
  plotId: string,
) {
  const result = await recommendationApi.getCurrentRecommendation(
    { signal },
    plotId,
  );
  signal.throwIfAborted();
  if (result.available && result.plotId !== plotId)
    throw new ApiError(
      'The recommendation belongs to another plot. Refresh farm context before continuing.',
      { kind: 'context' },
    );
  return result;
}
export function usePlotRecommendation() {
  return usePlotQuery('recommendation', loadPlotRecommendation);
}
