import { sensorOptions } from '../sensorOptions';
import type { Device } from '../../../domain/sensor';
import type { SectionProps } from '../types';
import { Field, SelectField } from './Fields';
export function SensorSettingsForm({
  draft: d,
  update,
  devices,
  errors,
}: {
  devices: Device[];
  discoveryError: string | null;
  onRetry: () => void;
} & SectionProps) {
  const options = sensorOptions(devices);
  const controllerIds = new Set([
    ...options.actuators.map(([id]) => id),
    ...d.actuatorIds,
  ]);
  const waterOptions = [
    ...options.actuators,
    ...options.confirmation.filter(([id]) => !controllerIds.has(id)),
  ];
  return (
    <div className="col-span-full grid gap-5">
      <p className="text-xs text-[#7c8876]">
        Sensors assigned to other plots are hidden. Unassign and save them there
        first to move them to this plot.
      </p>
      <SelectField
        label="Soil moisture / tension sensors"
        help="Select one or more soil-moisture or soil-tension sensors available to this plot."
        multiple
        value={d.moistureSensorIds}
        options={options.moisture}
        onChange={(v) => update('moistureSensorIds', v)}
      />
      <SelectField
        label="Soil temperature sensors"
        help="Select one or more soil-temperature sensors available to this plot."
        multiple
        value={d.temperatureSensorIds}
        options={options.temperature}
        onChange={(v) => update('temperatureSensorIds', v)}
      />
      <SelectField
        label="Water flow sensor"
        help="Select the water flow device and channels available to this plot."
        multiple
        value={[...new Set([...d.actuatorIds, ...d.flowConfirmationSensorIds])]}
        options={waterOptions}
        onChange={(values) => {
          update(
            'actuatorIds',
            values.filter((id) => controllerIds.has(id)),
          );
          update(
            'flowConfirmationSensorIds',
            values.filter((id) => !controllerIds.has(id)),
          );
        }}
      />
      <Field
        label="Irrigation slope detection"
        type="number"
        value={d.slope}
        error={errors.slope}
        help="Specify the slope value to help detect artificial irrigation. Only required if no water flow sensor is added."
        onChange={(v) => update('slope', v)}
      />
    </div>
  );
}
