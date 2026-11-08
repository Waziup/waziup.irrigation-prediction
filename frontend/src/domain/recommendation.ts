import type { JsonObject } from './api';
import type { Urgency } from './alert';
import type { IrrigationMode } from './operation';
export const conditionStatuses = [
  'unknown',
  'threshold_exceeded',
  'at_threshold',
  'within_threshold',
] as const;
export type ConditionStatus = (typeof conditionStatuses)[number];

export interface AvailableRecommendation {
  available: true;
  schemaVersion: string;
  generatedAt: string;
  plotId: string;
  reason: string | null;
  limitations: string[];
  action: {
    shouldIrrigate: boolean;
    urgency: Urgency;
    label: string;
    mode: IrrigationMode | null;
    advisoryOnly: boolean;
    reason: string | null;
  };
  condition: {
    tensionCbar: number | null;
    thresholdCbar: number | null;
    marginCbar: number | null;
    status: ConditionStatus;
    message: string;
  };
  crop: {
    type: string | null;
    currentStage: string | null;
    gddCumulative: number | null;
    kc: number | null;
    seasonActive: boolean | null;
    seasonStatus: string | null;
  };
  water: {
    recommendedVolumeM3: number | null;
    calculatedRequirementM3: number | null;
    appliedSinceCalculationM3: number | null;
    remainingRequirementM3: number | null;
    outlook: {
      remainingNetIrrigationMm: number | null;
      remainingGrossIrrigationMm: number | null;
      details: JsonObject;
    };
  };
  freshness: { stale: boolean; staleSources: string[] };
  timing: {
    firstBreachTimestamp: string | null;
    firstBreachHorizon: string | null;
  };
  // Provider/decision metadata is retained but is not an unchecked domain model.
  weather: {
    etcDailyMm: number | null;
    ageHours: number | null;
    details: JsonObject;
  };
  satellite: {
    ndvi: number | null;
    ndre: number | null;
    sourceAgeHours: number | null;
    details: JsonObject;
  };
  capabilityReadiness: JsonObject;
}
export type ThresholdStatus =
  | { breached: true; timestamp: string }
  | { breached: false; available: boolean; reason: string | null };

export type IrrigationRecommendation =
  | AvailableRecommendation
  | {
      available: false;
      reason: string | null;
      modelForecastAvailable: boolean | null;
      cropType: string | null;
      currentStage: string | null;
      thresholdCbar: number | null;
      details: JsonObject;
    };
