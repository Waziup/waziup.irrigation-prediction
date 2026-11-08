import type { PlotSettingsInput } from './settings';
// Preserved from ui/settings.html: USDA-ARS ROSETTA texture-only estimates.
// These are not measured curves. Never apply a preset during hydration.
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
export const soilTextureClasses: Record<string, number> = {
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
  const p = parameters[name];
  if (!p) return null;
  const alpha = Math.pow(10, p.logAlpha);
  const n = Math.pow(10, p.logN);
  const m = 1 - 1 / n;
  const fc = ['Sandy_Soil', 'Loamy_Sand'].includes(name) ? 10 : 33;
  return {
    soilType: name,
    soilTextureClass: soilTextureClasses[name] ?? null,
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
