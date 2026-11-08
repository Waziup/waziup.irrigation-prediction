import type { Unavailable } from './api';
import type { ThresholdMode } from './plot';
export type ChartData =
  | Unavailable
  | {
      available: true;
      timestamps: string[];
      moistureSeries: (number | null)[];
      temperatureSeries: (number | null)[] | null;
      vwcSeries: (number | null)[] | null;
      unit: string;
      kind: string;
      thresholdMode: ThresholdMode;
      thresholdCbar: number | null;
      staticThresholdCbar: number | null;
      thresholdSeries: (number | null)[] | null;
      staticThresholdSeries: (number | null)[] | null;
      saturation: number | null;
      fieldCapacityLower: number | null;
      fieldCapacityUpper: number | null;
      permanentWiltingPoint: number | null;
      dataSource: string | null;
      thresholdReason: string | null;
      forecastHorizonsHours: number[] | null;
      forecastHorizonMaxHours: number | null;
    };
