import type { Farm } from '../../domain/farm';
import type { Plot, AreaUnit } from '../../domain/plot';
import type { GeoJsonPosition, PolygonGeometry } from '../../domain/geometry';

export interface MappedFarm extends Farm {
  geometry: PolygonGeometry;
  geometryVersion: string;
  areaM2: number;
}
export interface MappedPlot extends Plot {
  cropType: string;
  geometry: PolygonGeometry;
  geometryVersion: string;
}
export type Installation =
  | {
      state: 'empty';
      revision: number;
      farm: null;
      plots: [];
      currentPlotId: null;
    }
  | {
      state: 'complete';
      revision: number;
      farm: MappedFarm;
      plots: MappedPlot[];
      currentPlotId: string;
    };
export interface SetupDraft {
  name: string;
  owner: string;
  timezone: string;
  areaUnit: AreaUnit;
  latitude: string;
  longitude: string;
  farmGeometry: PolygonGeometry | null;
  plotGeometry: PolygonGeometry | null;
  plotName: string;
  cropType: string;
}
export interface CreationAcknowledgement {
  revision: number;
  farm: MappedFarm;
  plot: MappedPlot;
  currentPlotId: string;
}
export const setupKey = ['farm-setup'] as const;
export const initialDraft: SetupDraft = {
  name: '',
  owner: '',
  timezone: 'Africa/Nairobi',
  areaUnit: 'm2',
  latitude: '',
  longitude: '',
  farmGeometry: null,
  plotGeometry: null,
  plotName: '',
  cropType: '',
};
export type BoundaryVertices = readonly GeoJsonPosition[];

export const crops = [
  'maize',
  'beans',
  'wheat',
  'tomato',
  'sorghum',
  'rice',
  'olive',
] as const;
