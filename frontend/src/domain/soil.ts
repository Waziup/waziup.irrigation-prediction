import type { PlotSettingsInput } from './settings';
import type { FieldErrors } from './api';

// Existing soil_calibration.py rules. The server remains authoritative.
export function validateSoilSettings(input: PlotSettingsInput): FieldErrors {
  const errors: FieldErrors = {};
  const points = [...input.retentionCurve].sort(
    (a, b) => a.tensionCbar - b.tensionCbar,
  );
  if (
    points.length < 3 ||
    points.some(
      (p, i) =>
        !Number.isFinite(p.tensionCbar) ||
        !Number.isFinite(p.vwc) ||
        p.tensionCbar < 0 ||
        p.vwc <= 0 ||
        p.vwc >= 1 ||
        (i > 0 &&
          (p.tensionCbar <= points[i - 1]!.tensionCbar ||
            p.vwc >= points[i - 1]!.vwc)),
    )
  )
    errors.retentionCurve =
      'Use at least three points with increasing cbar and decreasing VWC fractions between 0 and 1.';
  else if (
    points[0]!.tensionCbar > input.saturation ||
    points.at(-1)!.tensionCbar < input.permanentWiltingPoint
  )
    errors.retentionCurve =
      'The curve must cover saturation through wilting tension.';
  if (!(
    0 <= input.saturation &&
    input.saturation < input.fieldCapacityLower &&
    input.fieldCapacityLower < input.permanentWiltingPoint
  ))
    errors.soilCalibration =
      'Require saturation < field capacity < wilting tension.';
  const c = input.soilCalibration;
  if (
    !['estimate', 'manual', 'laboratory', 'field'].includes(
      String(c.source ?? 'estimate'),
    )
  )
    errors.soilCalibration = 'Unsupported calibration source.';
  if (c.date && !isCalendarDate(String(c.date)))
    errors.soilCalibration = 'Enter a valid measurement date.';
  if (
    c.sample_depth_cm !== null &&
    c.sample_depth_cm !== undefined &&
    c.sample_depth_cm !== '' &&
    (!Number.isFinite(Number(c.sample_depth_cm)) ||
      Number(c.sample_depth_cm) <= 0)
  )
    errors.soilCalibration = 'Sample depth must be positive.';
  if (
    input.permanentWiltingPoint !== 1500 &&
    !['laboratory', 'field'].includes(String(c.source))
  )
    errors.soilCalibration =
      'Nonstandard wilting tension requires a field or laboratory source; estimates use 1500 cbar.';
  return errors;
}
export function isCalendarDate(value: string): boolean {
  const date = new Date(`${value}T00:00:00Z`);
  return (
    /^\d{4}-\d{2}-\d{2}$/.test(value) &&
    Number(value.slice(0, 4)) > 0 &&
    Number.isFinite(date.getTime()) &&
    date.toISOString().slice(0, 10) === value
  );
}
// Same log1p interpolation used by the legacy settings water-content summary.
export function soilWaterContents(
  input: Pick<
    PlotSettingsInput,
    | 'retentionCurve'
    | 'saturation'
    | 'fieldCapacityLower'
    | 'permanentWiltingPoint'
  >,
) {
  const points = [...input.retentionCurve].sort(
    (a, b) => a.tensionCbar - b.tensionCbar,
  );
  if (
    points.length < 3 ||
    points.some(
      (p, i) =>
        !Number.isFinite(p.tensionCbar) ||
        !Number.isFinite(p.vwc) ||
        p.tensionCbar < 0 ||
        p.vwc <= 0 ||
        p.vwc >= 1 ||
        (i > 0 &&
          (p.tensionCbar <= points[i - 1]!.tensionCbar ||
            p.vwc >= points[i - 1]!.vwc)),
    )
  )
    return null;
  const at = (t: number) => {
    if (
      !Number.isFinite(t) ||
      t < points[0]!.tensionCbar ||
      t > points.at(-1)!.tensionCbar
    )
      return null;
    const index = points.findIndex((p) => p.tensionCbar >= t);
    if (index === 0) return points[0]!.vwc;
    const lo = points[index - 1]!,
      hi = points[index]!;
    return (
      lo.vwc +
      ((hi.vwc - lo.vwc) * (Math.log1p(t) - Math.log1p(lo.tensionCbar))) /
        (Math.log1p(hi.tensionCbar) - Math.log1p(lo.tensionCbar))
    );
  };
  const saturation = at(input.saturation),
    fieldCapacity = at(input.fieldCapacityLower),
    wilting = at(input.permanentWiltingPoint);
  return saturation === null || fieldCapacity === null || wilting === null
    ? null
    : {
        saturation,
        fieldCapacity,
        wilting,
        availableWater: fieldCapacity - wilting,
      };
}
