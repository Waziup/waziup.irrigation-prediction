import type { Device } from '../../domain/sensor';
interface SensorSelections {
  moistureSensorIds: string[];
  temperatureSensorIds: string[];
  flowConfirmationSensorIds: string[];
  actuatorIds: string[];
}
export function sensorOptions(devices: Device[], selected?: SensorSelections) {
  const sensors = devices.flatMap((device) =>
    device.sensors.map((sensor) => ({
      value: sensor.dataSource || `${device.id}/${sensor.id}`,
      label: `${device.name ?? device.id} / ${sensor.name ?? sensor.id}${sensor.dataSource ? ` · recorded channel ${sensor.id}` : ''}`,
    })),
  );
  const unique = (values: readonly { value: string; label: string }[]) => [
    ...new Map(
      values.map((v) => [v.value, [v.value, v.label] as const]),
    ).values(),
  ];
  const moisture = selected?.moistureSensorIds ?? [];
  const temperature = selected?.temperatureSensorIds ?? [];
  const water = [
    ...(selected?.flowConfirmationSensorIds ?? []),
    ...(selected?.actuatorIds ?? []),
  ];
  const available = (own: string[], elsewhere: string[]) => {
    const retained = new Set(own),
      excluded = new Set(elsewhere);
    return unique(
      sensors.filter(
        ({ value }) => retained.has(value) || !excluded.has(value),
      ),
    );
  };
  return {
    moisture: available(moisture, [...temperature, ...water]),
    temperature: available(temperature, [...moisture, ...water]),
    confirmation: available(water, [...moisture, ...temperature]),
    actuators: unique(
      devices.flatMap((device) =>
        device.actuators.map((actuator) => ({
          value: `${device.id}/${actuator.id}`,
          label: `${device.name ?? device.id} / ${actuator.name ?? actuator.id}`,
        })),
      ),
    ),
  };
}
