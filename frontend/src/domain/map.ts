import type { FarmDashboard, FarmPlotCard } from './farm';
import type { PolygonGeometry } from './geometry';

/** Saved boundaries only; unsaved editor drafts are separate state. */
export interface SavedMapBoundary {
  geometry: PolygonGeometry;
  areaM2: number;
  geometryVersion: string;
  geometryUpdatedAt: string;
}

export interface FarmMapPlot extends FarmPlotCard, SavedMapBoundary {}

/** Domain projection of GET /api/farmMap; API adapters validate unknown JSON. */
export type FarmMap =
  | {
      schemaVersion: '1.0';
      registryRevision: number;
      setup: { complete: false; reason: 'no_farm' };
      farm: null;
      plots: [];
      summary: null;
      generatedAt: null;
    }
  | {
      schemaVersion: '1.0';
      registryRevision: number;
      setup: { complete: true; reason: null };
      farm: FarmDashboard['farm'] & SavedMapBoundary;
      plots: FarmMapPlot[];
      summary: FarmDashboard['summary'];
      generatedAt: string;
    };
