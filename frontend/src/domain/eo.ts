import type { JsonObject, Unavailable } from './api';
export interface VegetationIndex {
  available: boolean;
  reason: string | null;
  latestValue: number | null;
  trend: 'increasing' | 'decreasing' | 'stable' | 'unknown';
  observations: { timestamp: string; value: number; source: string | null }[];
  greening: JsonObject;
}
export type EOObservation =
  | Unavailable
  | {
      available: true;
      provider: string;
      latestNdvi: number | null;
      latestNdre: number | null;
      latestObservation: string;
      validObservations: number;
      ndvi: VegetationIndex;
      ndre: VegetationIndex;
      cropIdentityConfirmed: boolean;
      phenologyInferred: boolean;
    };
