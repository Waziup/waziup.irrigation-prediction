import type { Urgency } from './alert';
export const irrigationModes = [
  'automatic',
  'approval_required',
  'manual',
  'advisory_only',
] as const;
export type IrrigationMode = (typeof irrigationModes)[number];
export const operationStatuses = [
  'planned',
  'pending_approval',
  'approved',
  'active',
  'completed',
  'declined',
  'failed',
  'verified',
] as const;
export type OperationStatus = (typeof operationStatuses)[number];
// Read model only. No scheduling or actuation commands are introduced.
export interface OperationSummary {
  id: string;
  source?: string | null;
  recommendationUrgency?: Urgency | null;
  mode?: IrrigationMode | null;
  plotId: string;
  status: OperationStatus;
  amountM3: number | null;
  plannedStart: string | null;
  plannedEnd: string | null;
}
