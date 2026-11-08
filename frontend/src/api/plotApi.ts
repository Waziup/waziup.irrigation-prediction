import type { RequestOptions } from '../domain/api';
import type { AreaUnit } from '../domain/plot';
import { fromSquareMetres } from '../domain/plot';
import { apiClient } from './client';
import { parseContext } from './adapters';
import { id, invalid, member, object } from './validation';
export const plotApi = {
  getPlots: (options?: RequestOptions) =>
    apiClient.json('getPlots', parseContext, options),
  selectPlot(plotId: string, options?: RequestOptions) {
    id(plotId);
    return apiClient.json(
      'setPlot',
      (value) => {
        const data = object(value);
        member(data.status, ['ok']);
        const selected = id(data.plot_id);
        if (selected !== plotId) invalid('plot_id', 'the requested plot ID');
        return { plotId: selected };
      },
      {
        ...options,
        method: 'POST',
        form: new URLSearchParams({ plot_id: plotId }),
      },
    );
  },
  addPlot(
    input: { name: string; areaM2: number; areaUnit: AreaUnit },
    options?: RequestOptions,
  ) {
    return apiClient.json(
      'addPlot',
      (value) => ({ plotId: id(object(value).plot_id) }),
      {
        ...options,
        method: 'POST',
        form: new URLSearchParams({
          name: input.name,
          area: String(fromSquareMetres(input.areaM2, input.areaUnit)),
          area_unit: input.areaUnit,
        }),
      },
    );
  },
  removePlot(plotId: string, options?: RequestOptions) {
    return apiClient.json(
      'removePlot',
      (value) => {
        const removed = id(object(value).plot_id);
        if (removed !== plotId) invalid('plot_id', 'the removed plot ID');
        return { plotId: removed };
      },
      {
        ...options,
        method: 'POST',
        form: new URLSearchParams({ plot_id: id(plotId) }),
      },
    );
  },
};
