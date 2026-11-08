export type WeatherView = 'forecast' | 'historical';
export type WeatherRange = 'model' | 'today' | 'recent';
import type { Unavailable } from './api';
export interface WeatherDay {
  date: string;
  label: string;
  rainMm: number | null;
  temperatureC: number | null;
  temperatureMinC: number | null;
  temperatureMaxC: number | null;
  humidityPercent: number | null;
  windSpeedKmh: number | null;
  windGustKmh: number | null;
  windDirectionDegrees: number | null;
  icon: string;
}
export interface WeatherRangeDays {
  model: number | null;
  recent: number | null;
}
export type WeatherForecast = (
  | Unavailable
  | {
      available: true;
      days: WeatherDay[];
      source: string;
      sourceDetail: string | null;
      generatedAt: string;
      cacheState?: 'fresh' | 'stale';
      farmName: string | null;
      rangeStart?: string | null;
      rangeEnd?: string | null;
    }
) & { rangeDays?: WeatherRangeDays | null };
