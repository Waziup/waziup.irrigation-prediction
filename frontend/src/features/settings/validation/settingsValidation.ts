import { isCalendarDate, validateSoilSettings } from '../../../domain/soil';
import type { FieldErrors } from '../../../domain/api';
import type {
  PlotSettingsInput,
  RetentionPoint,
} from '../../../domain/settings';
import { validatePlotSettings } from '../../../domain/settings';
import { toSquareMetres } from '../../../domain/plot';
import { numericFields, type NumericField, type SettingsDraft } from '../types';

export function parseRetention(text: string): RetentionPoint[] | null {
  const lines = text.trim().split(/\r?\n/);
  const points: RetentionPoint[] = [];
  for (const line of lines) {
    const values = line.split(',').map((v) => v.trim());
    if (
      values.length !== 2 ||
      values.some((v) => !v || !Number.isFinite(Number(v)))
    )
      return null;
    points.push({ tensionCbar: Number(values[0]), vwc: Number(values[1]) });
  }
  return points.length ? points : null;
}
export function validateSettingsDraft(draft: SettingsDraft): {
  input: PlotSettingsInput | null;
  errors: FieldErrors;
} {
  const errors: FieldErrors = {};
  const numbers = {} as Record<NumericField, number>;
  for (const key of numericFields) {
    numbers[key] = Number(draft[key]);
    if (!draft[key].trim() || !Number.isFinite(numbers[key]))
      errors[key] = 'Enter a finite number.';
  }
  for (const key of ['latitude', 'longitude'] as const) {
    if (!draft[key].trim() || !Number.isFinite(Number(draft[key])))
      errors[key] = 'Enter a coordinate.';
  }
  const retentionCurve = parseRetention(draft.retentionText);
  if (!retentionCurve)
    errors.retentionCurve = 'Enter tension, VWC pairs, one per line.';
  const { latitude, longitude, retentionText: _text, ...rest } = draft;
  void _text;
  let areaM2 = numbers.areaM2;
  try {
    areaM2 = toSquareMetres(areaM2, draft.areaUnit);
  } catch {
    errors.areaM2 = 'Area must be finite and nonnegative.';
  }
  const input: PlotSettingsInput = {
    ...rest,
    ...numbers,
    areaM2,
    coordinates: { latitude: Number(latitude), longitude: Number(longitude) },
    retentionCurve: retentionCurve ?? [],
  };
  Object.assign(errors, validatePlotSettings(input));
  Object.assign(errors, validateSoilSettings(input));
  if (input.plantingDate && !isCalendarDate(input.plantingDate))
    errors.plantingDate = 'Enter a valid planting date.';
  return { input: Object.keys(errors).length ? null : input, errors };
}
