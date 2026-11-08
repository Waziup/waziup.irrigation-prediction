import type { Device } from '../../domain/sensor';
export function sensorOptions(devices: Device[]) {
  const sensors = devices.flatMap((device) =>
    device.sensors.map((sensor) => ({
      value: sensor.dataSource || `${device.id}/${sensor.id}`,
      label: `${device.name ?? device.id} / ${sensor.name ?? sensor.id}${sensor.dataSource ? ` · recorded channel ${sensor.id}` : ''}`,
      identity: `${sensor.id} ${sensor.name ?? ''}`.toLowerCase(),
    })),
  );
  const unique = (values: readonly { value: string; label: string }[]) => [
    ...new Map(
      values.map((v) => [v.value, [v.value, v.label] as const]),
    ).values(),
  ];
  return {
    moisture: unique(
      sensors.filter((s) =>
        /(moist|humidity|tension|vwc|water content)/.test(s.identity),
      ),
    ),
    temperature: unique(
      sensors.filter((s) => /(temp|temperature)/.test(s.identity)),
    ),
    confirmation: unique(sensors),
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
