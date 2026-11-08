import type { IrrigationMode } from '../../domain/operation';
import type { Device } from '../../domain/sensor';
import type { PlotSettings } from '../../domain/settings';

export type ResourceState<T> =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; data: T };
export function resourceState<T>(query: {
  data: T | undefined;
  isPending: boolean;
  isFetching: boolean;
  isError: boolean;
  error: Error | null;
}): ResourceState<T> {
  if (query.isFetching) return { status: 'loading' };
  if (query.isError)
    return {
      status: 'error',
      error: query.error ?? new Error('Data unavailable.'),
    };
  if (query.isPending || query.data === undefined) return { status: 'loading' };
  return { status: 'ready', data: query.data };
}
export const irrigationModeLabels: Record<IrrigationMode, string> = {
  automatic: 'Automatic',
  approval_required: 'Approval required',
  manual: 'Manual',
  advisory_only: 'Advisory only',
};
export interface AssignedSensor {
  id: string;
  label: string;
  role: 'Moisture' | 'Temperature' | 'Actuator';
  discovered: boolean;
}
export function assignedSensors(
  settings: PlotSettings,
  devices: Device[] | null,
): AssignedSensor[] {
  const catalog =
    devices?.flatMap((device) => [
      ...device.sensors.map((sensor) => ({
        id: `${device.id}/${sensor.id}`,
        source: sensor.dataSource,
        label: `${device.name ?? device.id} / ${sensor.name ?? sensor.id}`,
        actuator: false,
      })),
      ...device.actuators.map((sensor) => ({
        id: `${device.id}/${sensor.id}`,
        source: sensor.dataSource,
        label: `${device.name ?? device.id} / ${sensor.name ?? sensor.id}`,
        actuator: true,
      })),
    ]) ?? [];
  return (
    [
      ['Moisture', settings.moistureSensorIds],
      ['Temperature', settings.temperatureSensorIds],
      ['Actuator', settings.actuatorIds],
    ] as const
  ).flatMap(([role, ids]) =>
    ids.map((id) => {
      const channel = catalog.find(
        (channel) =>
          (channel.id === id || channel.source === id) &&
          channel.actuator === (role === 'Actuator'),
      );
      return {
        id,
        role,
        label: channel?.label ?? id,
        discovered: channel !== undefined,
      };
    }),
  );
}

/** An explicit unavailable result is different from loading or an empty list. */
export function unavailableResource(state: ResourceState<unknown>): boolean {
  if (state.status === 'error') return true;
  return (
    state.status === 'ready' &&
    typeof state.data === 'object' &&
    state.data !== null &&
    'available' in state.data &&
    state.data.available === false
  );
}
