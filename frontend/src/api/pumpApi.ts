import type { RequestOptions } from '../domain/api';
import { apiClient } from './client';
import { boolean, id, nullable, object, string } from './validation';

export const pumpApi = {
  status(plotId: string, options?: RequestOptions) {
    return apiClient.json(
      'getPumpControl',
      (value) => {
        const d = object(value);
        return {
          plotId: id(d.plot_id),
          configured: boolean(d.configured),
          mode: string(d.mode),
          busy: boolean(d.busy),
          operationId: nullable(d.operation_id, id),
          operationStatus: nullable(d.operation_status, string),
        };
      },
      { ...options, query: { plot_id: plotId } },
    );
  },
  command(
    plotId: string,
    state: 'on' | 'off',
    amount: string,
    requestId: string,
  ) {
    return apiClient.json(
      'pumpControl',
      (value) => {
        const d = object(value);
        return {
          plotId: id(d.plot_id),
          message: string(d.message),
          operationId: nullable(d.operation_id, id),
        };
      },
      {
        method: 'POST',
        form: new URLSearchParams({
          plot_id: plotId,
          state,
          amount,
          idempotency_key: requestId,
        }),
      },
    );
  },
};
