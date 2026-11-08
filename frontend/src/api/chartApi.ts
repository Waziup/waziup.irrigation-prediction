import type { RequestOptions } from '../domain/api';
import type { ChartData } from '../domain/chart';
import { thresholdModes } from '../domain/plot';
import { readWithUnavailable } from './client';
import {
  array,
  invalid,
  member,
  nullable,
  number,
  object,
  string,
  timestamp,
  unavailable,
} from './validation';
function series(value: unknown): (number | null)[] {
  return array(value, (v) => nullable(v, number));
}
export function parseChart(value: unknown): ChartData {
  const d = object(value);
  const missing = unavailable(d);
  if (missing) return missing;
  const timestamps = array(d.timestamps, timestamp);
  const moistureSeries = series(d.moistureSeries);
  if (
    !timestamps.length ||
    timestamps.length !== moistureSeries.length ||
    !moistureSeries.some((value) => value !== null)
  )
    invalid('chart', 'aligned timestamps with at least one finite measurement');
  function optionalSeries(
    value: unknown,
    allowEmpty = false,
  ): (number | null)[] | null {
    if (value === undefined || value === null) return null;
    const result = series(value);
    if (
      result.length !== timestamps.length &&
      !(allowEmpty && result.length === 0)
    )
      invalid('chart series', 'one value per timestamp');
    return result;
  }
  return {
    available: true,
    timestamps,
    moistureSeries,
    temperatureSeries: optionalSeries(d.temperatureSeries),
    vwcSeries: optionalSeries(d.moistureSeriesVol ?? d.vwcSeries),
    unit: string(d.unit),
    kind: string(d.kind),
    thresholdMode: member(d.threshold_mode, thresholdModes),
    thresholdCbar: nullable(d.threshold_cbar, number),
    staticThresholdCbar: nullable(d.threshold_cbar_static, number),
    thresholdSeries: optionalSeries(d.threshold_series, true),
    staticThresholdSeries: optionalSeries(d.threshold_series_static, true),
    saturation: nullable(d.saturation, number),
    fieldCapacityLower: nullable(d.fieldCapacityLower, number),
    fieldCapacityUpper: nullable(d.fieldCapacityUpper, number),
    permanentWiltingPoint: nullable(d.permanentWiltingPoint, number),
    dataSource: nullable(d.data_source, string),
    thresholdReason: nullable(d.threshold_reason, string),
    forecastHorizonsHours: nullable(d.forecast_horizons_available, (value) =>
      array(value, number),
    ),
    forecastHorizonMaxHours: nullable(d.forecast_horizon_max_hours, number),
  };
}
export const chartApi = {
  getCurrentHistory: (options?: RequestOptions) =>
    readWithUnavailable('getHistoricalChartData', parseChart, options),
  getCurrentPrediction: (options?: RequestOptions) =>
    readWithUnavailable('getPredictionChartData', parseChart, options),
};
