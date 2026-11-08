import type { JsonObject, Unavailable } from './api';
export const urgencies = [
  'ok',
  'none',
  'watch',
  'advise',
  'critical',
  'error',
  'unknown',
] as const;
export type Urgency = (typeof urgencies)[number];
export interface AlertSummary {
  id: string;
  plotId: string;
  urgency: Urgency;
  payload: JsonObject;
}
export type AlertStatus =
  | Unavailable
  | {
      available: true;
      urgency: Urgency;
      notify: boolean;
      notifyHeadline: string | null;
      notifyMessage: string | null;
      timestampUtc: string | null;
      details: JsonObject;
    };
