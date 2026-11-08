import type { RequestOptions } from '../domain/api';
import type {
  IrrigationRecommendation,
  ThresholdStatus,
} from '../domain/recommendation';
import { urgencies } from '../domain/alert';
import { conditionStatuses } from '../domain/recommendation';
import { irrigationModes } from '../domain/operation';
import { apiClient } from './client';
import {
  array,
  boolean,
  id,
  invalid,
  jsonObject,
  member,
  nullable,
  number,
  object,
  string,
  timestamp,
} from './validation';
export function parseRecommendation(value: unknown): IrrigationRecommendation {
  const d = object(value);
  if (d.available === false) {
    const crop = d.crop == null ? null : object(d.crop);
    const condition = d.condition == null ? null : object(d.condition);
    return {
      available: false,
      reason: nullable(d.reason ?? d.availability_reason, string),
      modelForecastAvailable: nullable(d.model_forecast_available, boolean),
      cropType: nullable(crop?.type ?? d.crop_type, string),
      currentStage: nullable(crop?.current_stage ?? d.growth_stage, string),
      thresholdCbar: nullable(
        condition?.threshold_cbar ?? d.threshold_cbar,
        number,
      ),
      details: jsonObject(d),
    };
  }
  if (d.available !== true) invalid('available', 'a boolean');
  const action = object(d.action);
  const condition = object(d.condition);
  const crop = object(d.crop);
  const water = object(d.water);
  const freshness = object(d.freshness);
  const timing = object(d.timing);
  const weather = object(d.weather);
  const satellite = object(d.satellite);
  const outlook = object(water.outlook);
  const requirement =
    outlook.requirement == null ? null : object(outlook.requirement);
  return {
    available: true,
    schemaVersion: member(d.schema_version, ['2.0']),
    generatedAt: timestamp(d.generated_at),
    plotId: id(object(d.plot).plot_id),
    reason: nullable(d.reason, string),
    limitations: array(d.limitations, string),
    action: {
      shouldIrrigate: boolean(action.should_irrigate),
      urgency: member(action.urgency, urgencies),
      label: string(action.label),
      mode: nullable(action.mode, (v) => member(v, irrigationModes)),
      advisoryOnly: boolean(action.advisory_only),
      reason: nullable(action.reason, string),
    },
    condition: {
      tensionCbar: nullable(condition.current_tension_cbar, number),
      thresholdCbar: nullable(condition.threshold_cbar, number),
      marginCbar: nullable(condition.margin_cbar, number),
      status: member(condition.status, conditionStatuses),
      message: string(condition.message),
    },
    crop: {
      type: nullable(crop.type, string),
      currentStage: nullable(crop.current_stage, string),
      gddCumulative: nullable(crop.gdd_cumulative, number),
      kc: nullable(crop.kc, number),
      seasonActive: nullable(crop.season_active, boolean),
      seasonStatus: nullable(crop.season_status, string),
    },
    water: {
      recommendedVolumeM3: nullable(water.recommended_volume_m3, number),
      calculatedRequirementM3: nullable(
        water.calculated_requirement_volume_m3,
        number,
      ),
      appliedSinceCalculationM3: nullable(
        water.applied_since_previous_calculation_m3,
        number,
      ),
      remainingRequirementM3: nullable(
        water.remaining_requirement_volume_m3,
        number,
      ),
      outlook: {
        remainingNetIrrigationMm: nullable(
          requirement?.remaining_net_irrigation_mm,
          number,
        ),
        remainingGrossIrrigationMm: nullable(
          requirement?.remaining_gross_irrigation_mm,
          number,
        ),
        details: jsonObject(outlook),
      },
    },
    freshness: {
      stale: boolean(freshness.stale),
      staleSources: array(freshness.stale_sources, string),
    },
    timing: {
      firstBreachTimestamp: nullable(timing.first_breach_timestamp, timestamp),
      firstBreachHorizon: nullable(timing.first_breach_horizon, string),
    },
    weather: {
      etcDailyMm: nullable(weather.etc_daily_mm, number),
      ageHours: nullable(weather.age_hours, number),
      details: jsonObject(weather),
    },
    satellite: {
      ndvi: nullable(satellite.ndvi, number),
      ndre: nullable(satellite.ndre, number),
      sourceAgeHours: nullable(satellite.source_age_hours, number),
      details: jsonObject(satellite),
    },
    capabilityReadiness: jsonObject(d.capability_readiness),
  };
}
export const recommendationApi = {
  getCurrentRecommendation: (options?: RequestOptions, plotId?: string) =>
    apiClient.json('getIrrigationRecommendation', parseRecommendation, {
      ...options,
      ...(plotId ? { query: { plot_id: plotId } } : {}),
    }),
  getCurrentThreshold(options?: RequestOptions): Promise<ThresholdStatus> {
    return apiClient.json(
      'getThreshold',
      (value) => {
        const d = object(value);
        if (d.timestamp !== undefined)
          return { breached: true, timestamp: timestamp(d.timestamp) };
        if (d.threshold !== false)
          invalid('threshold', 'false or a breach timestamp');
        return {
          breached: false,
          available: true,
          reason: nullable(d.reason, string),
        };
      },
      options,
    );
  },
};
