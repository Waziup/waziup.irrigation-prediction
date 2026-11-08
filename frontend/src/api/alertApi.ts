import type { RequestOptions } from '../domain/api';
import type { AlertStatus } from '../domain/alert';
import { urgencies } from '../domain/alert';
import { apiClient } from './client';
import {
  boolean,
  invalid,
  jsonObject,
  member,
  nullable,
  object,
  string,
  timestamp,
  unavailable,
} from './validation';
export const alertApi = {
  // This GET advances notification-throttle state; avoid automatic retries/prefetch.
  getCurrentAlertStatus(options?: RequestOptions): Promise<AlertStatus> {
    return apiClient.json(
      'getAlertStatus',
      (value) => {
        const d = object(value);
        const missing = unavailable(d);
        if (missing) return missing;
        if (d.available !== true) invalid('available', 'a boolean');
        return {
          available: true,
          urgency: member(d.urgency, urgencies),
          notify: boolean(d.notify),
          notifyHeadline: nullable(d.notify_headline, string),
          notifyMessage: nullable(d.notify_message, string),
          timestampUtc: nullable(d.timestamp_utc, timestamp),
          details: jsonObject(d),
        };
      },
      options,
    );
  },
};
