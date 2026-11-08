import type { JsonObject, JsonValue, Unavailable } from './api';
export interface DeviceChannel {
  id: string;
  name: string | null;
  value: JsonValue;
  dataSource: string | null;
  metadata: JsonObject;
}
export interface Device {
  id: string;
  name: string | null;
  sensors: DeviceChannel[];
  actuators: DeviceChannel[];
}
export type SensorReadings =
  | Unavailable
  | {
      available: true;
      tensionCbar: number | null;
      temperatureC: number | null;
      vwcPercent: number | null;
      sensorKind: string;
      dataSource: string | null;
      recordedOrLive: string | null;
    };
// This endpoint reports configured IDs, not verified physical pump state.
export interface LegacyIrrigationConfiguration {
  hasConfiguredFlowActuator: boolean;
}
