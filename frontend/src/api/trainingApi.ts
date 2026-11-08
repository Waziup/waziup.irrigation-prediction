import type { RequestOptions } from '../domain/api';
import type { TrainingStart, TrainingState } from '../domain/training';
import { trainingStatuses } from '../domain/training';
import { apiClient } from './client';
import {
  boolean,
  id,
  invalid,
  member,
  nullable,
  object,
  string,
} from './validation';
function expectedId(value: unknown, expected: string): string {
  const result = id(value);
  if (result !== expected) invalid('plot_id', 'the requested training plot');
  return result;
}
export const trainingApi = {
  // Despite GET on the legacy wire, call this through a Query mutation, never a query/loader.
  startTraining(
    plotId: string,
    options?: RequestOptions,
  ): Promise<TrainingStart> {
    return apiClient.json(
      'startTraining',
      (value) => {
        const d = object(value);
        return {
          status: member(d.status, ['started']),
          plotId: expectedId(d.plot_id, plotId),
        };
      },
      { ...options, query: { plot_id: id(plotId) } },
    );
  },
  getTrainingStatus(
    plotId: string,
    options?: RequestOptions,
  ): Promise<TrainingState> {
    return apiClient.json(
      'isTrainingReady',
      (value) => {
        const d = object(value);
        return {
          plotId: expectedId(d.plot_id, plotId),
          status: member(d.status, trainingStatuses),
          isTrainingFinished: boolean(d.isTrainingFinished),
          currentlyTraining: boolean(d.currentlyTraining),
          error: nullable(d.error, string),
          recommendationAvailable: boolean(d.recommendation_available),
          recommendationError: nullable(d.recommendation_error, string),
        };
      },
      { ...options, query: { plot_id: id(plotId) } },
    );
  },
};
