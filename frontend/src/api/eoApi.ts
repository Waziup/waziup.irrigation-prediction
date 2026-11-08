import type { RequestOptions } from '../domain/api';
import type { EOObservation, VegetationIndex } from '../domain/eo';
import { apiClient } from './client';
import {
  array,
  boolean,
  integer,
  invalid,
  jsonObject,
  member,
  nullable,
  number,
  object,
  string,
  timestamp,
  unavailable,
} from './validation';
function parseIndex(value: unknown, index: 'ndvi' | 'ndre'): VegetationIndex {
  const d = object(value);
  const available = boolean(d.available);
  if (!available)
    return {
      available: false,
      reason: nullable(d.reason, string),
      latestValue: null,
      trend: 'unknown',
      observations: [],
      greening: jsonObject(d.greening),
    };
  return {
    available: true,
    reason: null,
    latestValue: nullable(d[`latest_${index}`], number),
    trend: member(object(d.trend).direction, [
      'increasing',
      'decreasing',
      'stable',
      'unknown',
    ]),
    greening: jsonObject(d.greening),
    observations: array(d.observations, (value) => {
      const row = object(value);
      return {
        timestamp: timestamp(row.timestamp),
        value: number(row[index]),
        source: nullable(row.source, string),
      };
    }),
  };
}
export function parseEO(value: unknown): EOObservation {
  const d = object(value);
  const missing = unavailable(d);
  if (missing) return missing;
  if (d.available !== true) invalid('available', 'a boolean');
  const indices = object(d.indices);
  return {
    available: true,
    provider: string(d.provider),
    latestNdvi: nullable(d.latest_ndvi, number),
    latestNdre: nullable(d.latest_ndre, number),
    latestObservation: timestamp(d.latest_observation),
    validObservations: integer(d.valid_observations),
    ndvi: parseIndex(indices.ndvi, 'ndvi'),
    ndre: parseIndex(indices.ndre, 'ndre'),
    cropIdentityConfirmed: boolean(d.crop_identity_confirmed),
    phenologyInferred: boolean(d.phenology_inferred),
  };
}
export const eoApi = {
  getCurrentObservation: (options?: RequestOptions, plotId?: string) =>
    apiClient.json(
      'getEOObservation',
      (value) => {
        if (plotId && object(value).plot_id !== plotId)
          invalid('plot_id', 'the requested plot');
        return parseEO(value);
      },
      { ...options, ...(plotId ? { query: { plot_id: plotId } } : {}) },
    ),
};
