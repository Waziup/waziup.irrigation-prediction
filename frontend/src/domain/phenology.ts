import type { JsonObject, JsonValue, Unavailable } from './api';
import type { ThresholdMode } from './plot';
export interface PhenologyStage {
  stage: string;
  gddStart: number;
  gddEnd: number | null;
  kcStart: number;
  kcEnd: number;
  thresholdCbar: number | null;
  thresholdMinCbar?: number | null;
  thresholdMaxCbar?: number | null;
  thresholdSource: string;
  thresholdBasis: string | null;
}
export interface GrowthWaterDemand {
  available: boolean;
  reason: string | null;
  periodStart: string;
  periodEnd: string;
  periodHours: number;
  et0Mm: number | null;
  etcMm: number | null;
  rainfallMm: number | null;
  volumeM3: number | null;
  creditedVolumeM3: number | null;
  remainingVolumeM3: number | null;
}
export type PhenologySummary =
  | Unavailable
  | {
      available: true;
      plotId?: string | null;
      cropType: string;
      cropName: string;
      plantingDate: string | null;
      referenceDate: string | null;
      referenceSource: string | null;
      referenceConfidence: string | null;
      referenceInterval: JsonValue;
      currentStage: string | null;
      currentGdd: number | null;
      kcToday?: number | null;
      etcForecastMm?: number | null;
      etcTodayMm?: number | null;
      growthReason?: string | null;
      waterDemand?: GrowthWaterDemand | null;
      initialGdd: number;
      thresholdMode: ThresholdMode;
      thresholdBaselineCbar: number | null;
      thresholdActiveCbar: number | null;
      thresholdDetails: JsonObject | null;
      thresholdReason?: string | null;
      evaluatedAt: string | null;
      gddThresholds: JsonObject;
      kc: JsonObject;
      stages: PhenologyStage[];
    };
