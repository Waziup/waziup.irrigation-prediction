import type { RequestOptions } from '../domain/api';
import type {
  WeatherForecast,
  WeatherView,
  WeatherRange,
} from '../domain/weather';
import { apiClient } from './client';
import {
  array,
  invalid,
  nullable,
  member,
  number,
  object,
  string,
  timestamp,
  unavailable,
} from './validation';
export function parseWeather(value: unknown): WeatherForecast {
  const d = object(value);
  const ranges = d.range_days == null ? null : object(d.range_days);
  const rangeDays = ranges
    ? {
        model: nullable(ranges.model, number),
        recent: nullable(ranges.recent, number),
      }
    : null;
  const missing = unavailable(d);
  if (missing) return rangeDays ? { ...missing, rangeDays } : missing;
  if (d.available !== true) invalid('available', 'a boolean');
  return {
    available: true,
    rangeDays,
    source: string(d.source),
    sourceDetail: nullable(d.source_detail, string),
    generatedAt: timestamp(d.generated_at),
    cacheState:
      d.cache_state == null
        ? 'fresh'
        : member(d.cache_state, ['fresh', 'stale']),
    farmName: nullable(d.farm_name, string),
    rangeStart: nullable(d.range_start, string),
    rangeEnd: nullable(d.range_end, string),
    days: array(d.days, (value) => {
      const day = object(value);
      return {
        date: timestamp(day.date),
        label: string(day.label),
        rainMm: nullable(day.rain_mm, number),
        temperatureC: nullable(day.temperature_c, number),
        temperatureMinC: nullable(day.temperature_min_c, number),
        temperatureMaxC: nullable(day.temperature_max_c, number),
        humidityPercent: nullable(day.humidity_percent, number),
        windSpeedKmh: nullable(day.wind_speed, number),
        windGustKmh: nullable(day.wind_gust, number),
        windDirectionDegrees: nullable(day.wind_direction_degrees, number),
        icon: string(day.icon),
      };
    }),
  };
}
export const weatherApi = {
  getFarmForecast: (
    options?: RequestOptions,
    selection?: { view: WeatherView; range: WeatherRange; plotId: string },
  ) =>
    apiClient.json('getWeatherForecast', parseWeather, {
      ...options,
      ...(selection
        ? {
            query: {
              view: selection.view,
              range: selection.range,
              plot_id: selection.plotId,
            },
          }
        : {}),
    }),
};
