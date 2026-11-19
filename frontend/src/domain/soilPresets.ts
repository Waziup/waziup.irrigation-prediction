import type { PlotSettingsInput } from './settings';
import legacyPresets from '../../../soil_presets.json';

// ROSETTA compatibility for existing saved soil names.
// New selections use the original shared table below. Never apply on hydration.
const parameters: Record<
  string,
  { thetaR: number; thetaS: number; logAlpha: number; logN: number }
> = {
  Sandy_Soil: { thetaR: 0.053, thetaS: 0.375, logAlpha: -1.453, logN: 0.502 },
  Loamy_Sand: { thetaR: 0.049, thetaS: 0.39, logAlpha: -1.459, logN: 0.242 },
  Sandy_Loam: { thetaR: 0.039, thetaS: 0.387, logAlpha: -1.574, logN: 0.161 },
  Loam: { thetaR: 0.061, thetaS: 0.399, logAlpha: -1.954, logN: 0.168 },
  Silt_Loam: { thetaR: 0.065, thetaS: 0.439, logAlpha: -2.296, logN: 0.221 },
  Silt: { thetaR: 0.05, thetaS: 0.489, logAlpha: -2.182, logN: 0.225 },
  Sandy_Clay_Loam: {
    thetaR: 0.063,
    thetaS: 0.384,
    logAlpha: -1.676,
    logN: 0.124,
  },
  Clay_Loam: { thetaR: 0.079, thetaS: 0.442, logAlpha: -1.801, logN: 0.151 },
  Sandy_Clay: { thetaR: 0.117, thetaS: 0.385, logAlpha: -1.476, logN: 0.082 },
  Silty_Clay_Loam: {
    thetaR: 0.09,
    thetaS: 0.482,
    logAlpha: -2.076,
    logN: 0.182,
  },
  Silty_Clay: { thetaR: 0.111, thetaS: 0.481, logAlpha: -1.79, logN: 0.121 },
  Clay: { thetaR: 0.098, thetaS: 0.459, logAlpha: -1.825, logN: 0.098 },
};
const modernTextureClasses: Record<string, number> = {
  Sandy_Soil: 0,
  Loamy_Sand: 1,
  Sandy_Loam: 2,
  Loam: 3,
  Silt_Loam: 4,
  Sandy_Clay: 5,
  Sandy_Clay_Loam: 6,
  Clay_Loam: 7,
  Silty_Clay_Loam: 8,
  Silt: 9,
  Silty_Clay: 10,
  Clay: 11,
};
export const soilTextureClasses: Record<string, number> = Object.fromEntries(
  Object.entries(legacyPresets).map(([name, preset]) => [
    name,
    preset.soilTextureClass,
  ]),
);
export const legacyPresetReference = 'WaziFarm original soil presets';

export function isLegacySoilPreset(
  input: Pick<
    PlotSettingsInput,
    | 'retentionCurve'
    | 'fieldCapacityLower'
    | 'permanentWiltingPoint'
    | 'saturation'
  > & { soilCalibration?: PlotSettingsInput['soilCalibration'] },
) {
  const metadata = input.soilCalibration;
  if (
    metadata?.source !== 'estimate' ||
    metadata.reference !== legacyPresetReference
  )
    return false;
  const name = String(metadata.preset ?? '');
  const preset = Object.hasOwn(legacyPresets, name)
    ? legacyPresets[name as keyof typeof legacyPresets]
    : undefined;
  if (
    !preset ||
    input.fieldCapacityLower !== preset.FieldCapacity[0] ||
    input.permanentWiltingPoint !== preset.PermanentWiltingPoint ||
    input.saturation !== preset.Saturation
  )
    return false;
  const points = [...input.retentionCurve].sort(
    (a, b) => a.tensionCbar - b.tensionCbar,
  );
  return (
    points.length === preset.soilTension.length &&
    points.every(
      (point, index) =>
        point.tensionCbar === preset.soilTension[index] &&
        point.vwc === preset.waterRetention[index],
    )
  );
}

export function soilPreset(
  name: string,
): Pick<
  PlotSettingsInput,
  | 'soilType'
  | 'soilTextureClass'
  | 'retentionCurve'
  | 'permanentWiltingPoint'
  | 'fieldCapacityUpper'
  | 'fieldCapacityLower'
  | 'saturation'
  | 'soilCalibration'
> | null {
  const legacy = Object.hasOwn(legacyPresets, name)
    ? legacyPresets[name as keyof typeof legacyPresets]
    : undefined;
  if (legacy)
    return {
      soilType: name,
      soilTextureClass: legacy.soilTextureClass,
      permanentWiltingPoint: legacy.PermanentWiltingPoint,
      fieldCapacityLower: legacy.FieldCapacity[0]!,
      fieldCapacityUpper: legacy.FieldCapacity[1]!,
      saturation: legacy.Saturation,
      retentionCurve: legacy.soilTension.map((tensionCbar, index) => ({
        tensionCbar,
        vwc: legacy.waterRetention[index]!,
      })),
      soilCalibration: {
        source: 'estimate',
        reference: legacyPresetReference,
        preset: name,
        date: '',
        sample_depth_cm: null,
      },
    };
  // Keep reset support for previously saved ROSETTA soil names.
  const p = parameters[name];
  if (!p) return null;
  const alpha = Math.pow(10, p.logAlpha);
  const n = Math.pow(10, p.logN);
  const m = 1 - 1 / n;
  const fc = ['Sandy_Soil', 'Loamy_Sand'].includes(name) ? 10 : 33;
  return {
    soilType: name,
    soilTextureClass: modernTextureClasses[name] ?? null,
    retentionCurve: [0, 10, 33, 100, 300, 500, 1000, 1500].map(
      (tensionCbar) => ({
        tensionCbar,
        vwc: Number(
          (
            p.thetaR +
            (p.thetaS - p.thetaR) /
              Math.pow(1 + Math.pow(alpha * tensionCbar * 10.19716213, n), m)
          ).toFixed(6),
        ),
      }),
    ),
    permanentWiltingPoint: 1500,
    fieldCapacityUpper: fc,
    fieldCapacityLower: fc,
    saturation: 0,
    soilCalibration: {
      source: 'estimate',
      reference: 'USDA ROSETTA',
      date: '',
      sample_depth_cm: null,
    },
  };
}
