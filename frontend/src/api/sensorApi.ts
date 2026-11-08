import type { RequestOptions } from '../domain/api';
import type {
  Device,
  DeviceChannel,
  SensorReadings,
  LegacyIrrigationConfiguration,
} from '../domain/sensor';
import { apiClient, readWithUnavailable } from './client';
import { ApiError, isAbortError } from './errors';
import {
  array,
  boolean,
  id,
  jsonObject,
  jsonValue,
  measurement,
  nullable,
  object,
  string,
  unavailable,
} from './validation';
function parseChannel(value: unknown): DeviceChannel {
  const d = object(value);
  return {
    id: id(d.id),
    name: nullable(d.name, string),
    value: d.value === undefined ? null : jsonValue(d.value),
    dataSource: nullable(d.data_source, string),
    metadata: d.meta === undefined ? {} : jsonObject(d.meta),
  };
}
export function parseDevices(value: unknown): Device[] {
  return array(
    Array.isArray(value) ? value : object(value).devices,
    (entry) => {
      const d = object(entry);
      return {
        id: id(d.id),
        name: nullable(d.name, string),
        sensors: d.sensors === undefined ? [] : array(d.sensors, parseChannel),
        actuators:
          d.actuators === undefined ? [] : array(d.actuators, parseChannel),
      };
    },
  );
}
export function parseReadings(value: unknown): SensorReadings {
  const d = object(value);
  const missing = unavailable(d);
  if (missing) return missing;
  return {
    available: true,
    tensionCbar: measurement(d.moisture_average),
    temperatureC: measurement(d.temp_average),
    vwcPercent: measurement(d.vwc_average),
    sensorKind: string(d.sensor_kind),
    dataSource: nullable(d.data_source, string),
    recordedOrLive: nullable(d.recorded_or_live, string),
  };
}
export const sensorApi = {
  async getDevices(
    options?: RequestOptions,
    plotId?: string,
  ): Promise<Device[]> {
    const scopedOptions = {
      ...options,
      ...(plotId ? { query: { plot_id: plotId } } : {}),
    };
    try {
      return await apiClient.json('devices', parseDevices, scopedOptions);
    } catch (error) {
      if (
        isAbortError(error) ||
        !(error instanceof ApiError) ||
        error.kind !== 'http'
      )
        throw error;
      return apiClient.json('getAvailableDevices', parseDevices, scopedOptions);
    }
  },
  getCurrentReadings: (options?: RequestOptions): Promise<SensorReadings> =>
    readWithUnavailable('getValuesForDashboard', parseReadings, options),
  getLegacyIrrigationConfiguration(
    options?: RequestOptions,
  ): Promise<LegacyIrrigationConfiguration> {
    return apiClient.json(
      'checkActiveIrrigation',
      (value) => ({
        hasConfiguredFlowActuator: boolean(
          object(object(value).data).activeIrrigation,
        ),
      }),
      options,
    );
  },
};
