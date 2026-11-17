import type { RequestOptions } from '../domain/api';
import type { PhenologySummary } from '../domain/phenology';
import { thresholdModes } from '../domain/plot';
import { apiClient } from './client';
import { ApiError } from './errors';
import {
  array,
  boolean,
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
    plotId: nullable(d.plot_id, string),
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
    kcToday: nullable(d.kc_today, number),
    etcForecastMm: nullable(d.etc_forecast_mm, number),
    etcTodayMm: nullable(d.etc_today_mm, number),
    growthReason: nullable(d.growth_reason, string),
    waterDemand:
      d.water_demand == null
        ? null
        : (() => {
            const demand = object(d.water_demand);
            return {
              available: boolean(demand.available),
              reason: nullable(demand.reason, string),
              periodStart: timestamp(demand.period_start),
              periodEnd: timestamp(demand.period_end),
              periodHours: number(demand.period_hours),
              et0Mm: nullable(demand.et0_mm, number),
              etcMm: nullable(demand.etc_mm, number),
              rainfallMm: nullable(demand.rainfall_mm, number),
              volumeM3: nullable(demand.volume_m3, number),
              creditedVolumeM3: nullable(demand.credited_volume_m3, number),
              remainingVolumeM3: nullable(demand.remaining_volume_m3, number),
            };
          })(),
    initialGdd: number(d.initial_gdd),
    thresholdMode: member(d.threshold_mode, thresholdModes),
    thresholdBaselineCbar: nullable(d.threshold_baseline_cbar, number),
    thresholdActiveCbar: nullable(d.threshold_active_cbar, number),
    thresholdDetails: nullable(d.threshold_details, jsonObject),
    thresholdReason: nullable(d.threshold_reason, string),
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
        thresholdMinCbar: nullable(row.threshold_min_cbar, number),
        thresholdMaxCbar: nullable(row.threshold_max_cbar, number),
        thresholdSource: string(row.threshold_source),
        thresholdBasis: nullable(row.threshold_basis, string),
      };
    }),
  };
}
export const phenologyApi = {
  async getCurrentSummary(options?: RequestOptions, plotId?: string) {
    const result = await apiClient.json('getPhenologySummary', parsePhenology, {
      ...options,
      ...(plotId ? { query: { plot_id: plotId } } : {}),
    });
    if (plotId && result.available && result.plotId && result.plotId !== plotId)
      throw new ApiError('The growth summary belongs to another plot.', {
        kind: 'context',
      });
    return result;
  },
};
