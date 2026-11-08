import type { Farm } from './farm';

export const areaUnits = ['m2', 'ha', 'acre'] as const;
export type AreaUnit = (typeof areaUnits)[number];
export const thresholdModes = ['static', 'dynamic'] as const;
export type ThresholdMode = (typeof thresholdModes)[number];
export interface Coordinates {
  latitude: number;
  longitude: number;
}
export interface Plot {
  id: string;
  name: string;
  cropType?: string | null;
  coordinates?: Coordinates | null;
  areaM2: number;
  areaUnit: AreaUnit;
}
export interface PlotContext {
  farm: Farm;
  plots: Plot[];
  currentPlotId: string;
}
const areaFactors: Record<AreaUnit, number> = {
  m2: 1,
  ha: 10000,
  acre: 4046.8564224,
};
export function isValidArea(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}
export function toSquareMetres(value: number, unit: AreaUnit): number {
  if (!isValidArea(value))
    throw new RangeError('Area must be finite and nonnegative.');
  const result = value * areaFactors[unit];
  if (!Number.isFinite(result))
    throw new RangeError('Converted area is not finite.');
  return result;
}
export function fromSquareMetres(value: number, unit: AreaUnit): number {
  if (!isValidArea(value))
    throw new RangeError('Area must be finite and nonnegative.');
  return value / areaFactors[unit];
}
export function isValidCoordinates(value: Coordinates): boolean {
  return (
    Number.isFinite(value.latitude) &&
    Math.abs(value.latitude) <= 90 &&
    Number.isFinite(value.longitude) &&
    Math.abs(value.longitude) <= 180
  );
}
