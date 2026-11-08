export const trainingStatuses = [
  'idle',
  'running',
  'finished',
  'error',
] as const;
export type TrainingStatus = (typeof trainingStatuses)[number];
export interface TrainingReadiness {
  ready: boolean;
  missing: string[];
  dataSource: string;
}
export interface TrainingStart {
  status: 'started';
  plotId: string;
}
export interface TrainingState {
  plotId: string;
  status: TrainingStatus;
  isTrainingFinished: boolean;
  currentlyTraining: boolean;
  error: string | null;
  recommendationAvailable: boolean;
  recommendationError: string | null;
}
