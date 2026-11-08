import type { RequestOptions } from '../domain/api';
import type { PhenologySummary } from '../domain/phenology';
import { thresholdModes } from '../domain/plot';
import { apiClient } from './client';
import {
  array,
  invalid,
  jsonObject,
  jsonValue,
  member,
  nullable,
  number,
  object,
  string,
  timestamp,
  unavailable,
} from './validation';
export function parsePhenology(value: unknown): PhenologySummary {
  const d = object(value);
  const missing = unavailable(d);
  if (missing) return missing;
  if (d.available !== true) invalid('available', 'a boolean');
  return {
    available: true,
    cropType: string(d.crop_type),
    cropName: string(d.crop_name),
    plantingDate: nullable(d.planting_date, string),
    referenceDate: nullable(d.phenology_reference_date, string),
    referenceSource: nullable(d.phenology_reference_source, string),
    referenceConfidence: nullable(d.phenology_reference_confidence, string),
    referenceInterval:
      d.phenology_reference_interval == null
        ? null
        : jsonValue(d.phenology_reference_interval),
    currentStage: nullable(d.current_stage, string),
    currentGdd: nullable(d.current_gdd, number),
    initialGdd: number(d.initial_gdd),
    thresholdMode: member(d.threshold_mode, thresholdModes),
    thresholdBaselineCbar: nullable(d.threshold_baseline_cbar, number),
    thresholdActiveCbar: nullable(d.threshold_active_cbar, number),
    thresholdDetails: nullable(d.threshold_details, jsonObject),
    evaluatedAt: nullable(d.evaluated_at, timestamp),
    gddThresholds: jsonObject(d.gdd_thresholds),
    kc: jsonObject(d.kc),
    stages: array(d.stage_rows, (value) => {
      const row = object(value);
      return {
        stage: string(row.stage),
        gddStart: number(row.gdd_start),
        gddEnd: nullable(row.gdd_end, number),
        kcStart: number(row.kc_start),
        kcEnd: number(row.kc_end),
        thresholdCbar: nullable(row.threshold_cbar, number),
        thresholdSource: string(row.threshold_source),
        thresholdBasis: nullable(row.threshold_basis, string),
      };
    }),
  };
}
export const phenologyApi = {
  getCurrentSummary: (options?: RequestOptions) =>
    apiClient.json('getPhenologySummary', parsePhenology, options),
};
