import type { FieldErrors } from '../../domain/api';
import type {
  PlotSettingsInput,
  PlotSettingsSnapshot,
} from '../../domain/settings';
import type { Plot } from '../../domain/plot';
import type { Farm } from '../../domain/farm';
import { fromSquareMetres } from '../../domain/plot';
import { soilPreset } from '../../domain/soilPresets';

export const numericFields = [
  'areaM2',
  'slope',
  'thresholdCbar',
  'applicationEfficiency',
  'effectiveRainfallFraction',
  'lookaheadHours',
  'period',
  'permanentWiltingPoint',
  'fieldCapacityUpper',
  'fieldCapacityLower',
  'saturation',
  'initialGdd',
] as const;
export type NumericField = (typeof numericFields)[number];
export type SettingsDraft = Omit<
  PlotSettingsInput,
  NumericField | 'coordinates' | 'retentionCurve'
> &
  Record<NumericField, string> & {
    latitude: string;
    longitude: string;
    retentionText: string;
  };
export type SettingsSnapshot = PlotSettingsSnapshot;
export interface SectionProps {
  draft: SettingsDraft;
  update: <K extends keyof SettingsDraft>(
    key: K,
    value: SettingsDraft[K],
  ) => void;
  errors: FieldErrors;
}
export function settingsInput(settings: PlotSettingsInput): PlotSettingsInput {
  // Explicitly exclude response metadata from mutation inputs and comparisons.
  const {
    name,
    zoneName,
    coordinates,
    areaM2,
    areaUnit,
    sensorKind,
    moistureSensorIds,
    temperatureSensorIds,
    actuatorIds,
    flowConfirmationSensorIds,
    flowConfirmationMode,
    slope,
    thresholdCbar,
    thresholdMode,
    irrigationType,
    irrigationMode,
    applicationEfficiency,
    effectiveRainfallFraction,
    lookaheadHours,
    startDate,
    period,
    soilType,
    soilTextureClass,
    soilCalibration,
    permanentWiltingPoint,
    fieldCapacityUpper,
    fieldCapacityLower,
    saturation,
    retentionCurve,
    cropType,
    plantingDate,
    initialGdd,
  } = settings;
  return {
    name,
    zoneName,
    coordinates,
    areaM2,
    areaUnit,
    sensorKind,
    moistureSensorIds,
    temperatureSensorIds,
    actuatorIds,
    flowConfirmationSensorIds,
    flowConfirmationMode,
    slope,
    thresholdCbar,
    thresholdMode,
    irrigationType,
    irrigationMode,
    applicationEfficiency,
    effectiveRainfallFraction,
    lookaheadHours,
    startDate,
    period,
    soilType,
    soilTextureClass,
    soilCalibration,
    permanentWiltingPoint,
    fieldCapacityUpper,
    fieldCapacityLower,
    saturation,
    retentionCurve,
    cropType,
    plantingDate,
    initialGdd,
  };
}
export function toDraft(input: PlotSettingsInput): SettingsDraft {
  const { coordinates, retentionCurve, ...rest } = settingsInput(input);
  const numbers = Object.fromEntries(
    numericFields.map((key) => [key, String(input[key])]),
  ) as Record<NumericField, string>;
  return {
    ...rest,
    ...numbers,
    areaM2: String(fromSquareMetres(input.areaM2, input.areaUnit)),
    plantingDate: input.cropType ? input.plantingDate.slice(0, 10) : '',
    latitude: String(coordinates.latitude),
    longitude: String(coordinates.longitude),
    retentionText: retentionCurve
      .map((p) => `${p.tensionCbar}, ${p.vwc}`)
      .join('\n'),
  };
}
export function initialSettings(
  plot: Plot,
  farm: Farm | null,
): PlotSettingsInput {
  const preset = soilPreset('Sandy_Soil')!;
  return {
    name: plot.name,
    zoneName: plot.name,
    coordinates: plot.coordinates ??
      farm?.coordinates ?? { latitude: 0, longitude: 0 },
    areaM2: plot.areaM2,
    areaUnit: plot.areaUnit,
    sensorKind: 'tension',
    moistureSensorIds: [],
    temperatureSensorIds: [],
    actuatorIds: [],
    flowConfirmationSensorIds: [],
    flowConfirmationMode: 'event',
    slope: 0,
    thresholdCbar: 30,
    thresholdMode: 'static',
    irrigationType: 'manual',
    irrigationMode: 'manual',
    applicationEfficiency: 0.85,
    effectiveRainfallFraction: 0.8,
    lookaheadHours: 24,
    startDate: '',
    period: 30,
    ...preset,
    cropType: plot.cropType ?? '',
    plantingDate: '',
    initialGdd: 0,
  };
}
export function sameSettings(
  a: SettingsSnapshot,
  b: SettingsSnapshot,
): boolean {
  if (a.plotId !== b.plotId || a.configVersion !== b.configVersion)
    return false;
  return a.configured && b.configured
    ? JSON.stringify(settingsInput(a.settings)) ===
        JSON.stringify(settingsInput(b.settings))
    : a.configured === b.configured;
}
