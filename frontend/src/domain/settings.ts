import type { FieldErrors, JsonObject } from './api';
import { irrigationModes, type IrrigationMode } from './operation';
import type { AreaUnit, Coordinates, ThresholdMode } from './plot';
import {
  areaUnits,
  thresholdModes,
  isValidArea,
  isValidCoordinates,
} from './plot';
import type { TrainingReadiness } from './training';
export const flowConfirmationModes = [
  'event',
  'metered',
  'cumulative',
] as const;
export type FlowConfirmationMode = (typeof flowConfirmationModes)[number];
export interface RetentionPoint {
  tensionCbar: number;
  vwc: number;
}
export interface PlotSettings {
  plotId: string;
  name: string;
  zoneName: string;
  coordinates: Coordinates;
  areaM2: number;
  areaUnit: AreaUnit;
  sensorKind: string;
  moistureSensorIds: string[];
  temperatureSensorIds: string[];
  actuatorIds: string[];
  flowConfirmationSensorIds: string[];
  flowConfirmationMode: FlowConfirmationMode;
  slope: number;
  thresholdCbar: number;
  thresholdMode: ThresholdMode;
  irrigationType: string;
  irrigationMode: IrrigationMode;
  applicationEfficiency: number;
  effectiveRainfallFraction: number;
  lookaheadHours: number;
  startDate: string;
  period: number;
  soilType: string;
  soilTextureClass: number | null;
  soilCalibration: JsonObject;
  permanentWiltingPoint: number;
  fieldCapacityUpper: number;
  fieldCapacityLower: number;
  saturation: number;
  retentionCurve: RetentionPoint[];
  cropType: string;
  plantingDate: string;
  initialGdd: number;
  trainingReadiness: TrainingReadiness;
}
export type PlotSettingsInput = Omit<
  PlotSettings,
  'plotId' | 'trainingReadiness'
>;
export interface SaveSettingsResult {
  status: 'saved' | 'pending';
  message: string;
}
// Snapshot metadata belongs to the API contract, not editable settings fields.
export type PlotSettingsSnapshot = {
  plotId: string;
  configVersion: string;
} & ({ configured: true; settings: PlotSettings } | { configured: false });
export interface PlotSettingsSaveResult extends SaveSettingsResult {
  plotId: string;
  configVersion: string;
}
export function validatePlotSettings(input: PlotSettingsInput): FieldErrors {
  const errors: FieldErrors = {};
  if (!areaUnits.includes(input.areaUnit))
    errors.areaUnit = 'Unsupported area unit.';
  if (!thresholdModes.includes(input.thresholdMode))
    errors.thresholdMode = 'Unsupported threshold mode.';
  if (!irrigationModes.includes(input.irrigationMode))
    errors.irrigationMode = 'Unsupported irrigation mode.';
  if (!flowConfirmationModes.includes(input.flowConfirmationMode))
    errors.flowConfirmationMode = 'Unsupported confirmation mode.';
  if (input.cropType && !input.plantingDate.trim())
    errors.plantingDate = 'Enter the planting date for the selected crop.';
  if (!input.cropType && input.plantingDate)
    errors.plantingDate =
      'A planting date cannot be set for an unplanted plot.';
  for (const key of ['plantingDate', 'startDate'] as const) {
    if (input[key] && !Number.isFinite(Date.parse(input[key])))
      errors[key] = 'A valid date is required.';
  }

  for (const key of ['name', 'zoneName', 'sensorKind', 'soilType'] as const) {
    if (!input[key].trim()) errors[key] = 'A value is required.';
  }
  if (!isValidCoordinates(input.coordinates))
    errors.coordinates = 'Invalid latitude or longitude.';
  if (!isValidArea(input.areaM2))
    errors.areaM2 = 'Area must be finite and nonnegative.';
  for (const key of [
    'slope',
    'thresholdCbar',
    'lookaheadHours',
    'initialGdd',
    'permanentWiltingPoint',
    'fieldCapacityUpper',
    'fieldCapacityLower',
    'saturation',
  ] as const) {
    if (!Number.isFinite(input[key]))
      errors[key] = 'A finite number is required.';
  }
  for (const key of [
    'applicationEfficiency',
    'effectiveRainfallFraction',
  ] as const) {
    if (!Number.isFinite(input[key]) || input[key] < 0.01 || input[key] > 1)
      errors[key] = 'Must be between 0.01 and 1.';
  }
  for (const key of [
    'permanentWiltingPoint',
    'fieldCapacityUpper',
    'fieldCapacityLower',
    'saturation',
  ] as const) {
    if (input[key] < 0) errors[key] = 'Must be nonnegative.';
  }
  if (input.lookaheadHours < 0) errors.lookaheadHours = 'Must be nonnegative.';
  if (!Number.isInteger(input.period) || input.period < 1)
    errors.period = 'A positive integer is required.';
  if (
    !input.retentionCurve.length ||
    input.retentionCurve.some(
      (p) => !Number.isFinite(p.tensionCbar) || !Number.isFinite(p.vwc),
    )
  )
    errors.retentionCurve = 'Finite retention points are required.';
  if (input.thresholdMode === 'dynamic' && input.retentionCurve.length < 3)
    errors.retentionCurve = 'Dynamic mode requires at least three points.';
  if (['automatic', 'approval_required'].includes(input.irrigationMode)) {
    if (!input.actuatorIds.length)
      errors.actuatorIds =
        'This mode requires an actuator. Select a pump or valve under “Pump or valve actuator”, then save.';
    if (!(input.areaM2 > 0))
      errors.areaM2 = 'This mode requires a positive plot area.';
  }
  return errors;
}
