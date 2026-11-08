import type { JsonObject, JsonValue, Unavailable } from './api';
import type { ThresholdMode } from './plot';
export interface PhenologyStage {
  stage: string;
  gddStart: number;
  gddEnd: number | null;
  kcStart: number;
  kcEnd: number;
  thresholdCbar: number | null;
  thresholdSource: string;
  thresholdBasis: string | null;
}
export type PhenologySummary =
  | Unavailable
  | {
      available: true;
      cropType: string;
      cropName: string;
      plantingDate: string | null;
      referenceDate: string | null;
      referenceSource: string | null;
      referenceConfidence: string | null;
      referenceInterval: JsonValue;
      currentStage: string | null;
      currentGdd: number | null;
      initialGdd: number;
      thresholdMode: ThresholdMode;
      thresholdBaselineCbar: number | null;
      thresholdActiveCbar: number | null;
      thresholdDetails: JsonObject | null;
      evaluatedAt: string | null;
      gddThresholds: JsonObject;
      kc: JsonObject;
      stages: PhenologyStage[];
    };
