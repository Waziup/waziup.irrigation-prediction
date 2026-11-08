import { useContext, useState } from 'react';
import { SettingsDataContext } from '../settingsDataContext';
import { Field, SettingsSection } from './Fields';
import type { ReactNode } from 'react';
import { PlotIdentityForm } from './PlotIdentityForm';
import { SensorSettingsForm } from './SensorSettingsForm';
import { IrrigationSettingsForm } from './IrrigationSettingsForm';
import { SoilSettingsForm } from './SoilSettingsForm';
import { CropSettingsForm } from './CropSettingsForm';
import { ThresholdSettingsForm } from './ThresholdSettingsForm';
import { PhenologySettingsForm } from './PhenologySettingsForm';
import type { Device } from '../../../domain/sensor';
import type { SectionProps, SettingsDraft } from '../types';
import { LoadingMessage } from '../../app-shell/LoadingBoundary';
export function PlotSettingsSections({
  props,
  changed,
  devices,
  discoveryError,
  discovering,
  onRetry,
  disabled = false,
  actions,
}: {
  props: SectionProps;
  actions?: ReactNode;
  disabled?: boolean;
  changed: (keys: readonly (keyof SettingsDraft)[]) => boolean;
  devices: Device[];
  discoveryError: string | null;
  discovering: boolean;
  onRetry: () => void;
}) {
  const available = useContext(SettingsDataContext);
  const [selectedSection, setActive] = useState('plot');
  const active =
    props.errors.actuatorIds || props.errors.actuator
      ? 'device'
      : selectedSection;
  return (
    <div className="grid gap-3">
      <nav
        aria-label="Selected plot settings sections"
        className="grid grid-cols-3 overflow-hidden rounded-xl border border-[#c8d4c6] bg-white"
      >
        {[
          ['plot', 'Plot & crop'],
          ['soil', 'Soil & threshold'],
          ['device', 'Sensors & irrigation'],
        ].map(([id, label]) => (
          <button
            key={id}
            type="button"
            aria-pressed={active === id}
            onClick={() => setActive(id!)}
            className={`min-h-12 cursor-pointer border-r border-[#e0e7df] px-2 py-3 text-xs font-semibold last:border-r-0 sm:text-sm ${active === id ? 'bg-[#315f48] text-white' : 'text-[#52627a] hover:bg-[#f3f7f2]'}`}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="grid min-w-0 gap-3">
        {active === 'plot' && (
          <>
            <SettingsSection
              disabled={!available || disabled}
              title="1. Plot and crop"
              footer={
                <div className="col-span-full">
                  <button
                    type="button"
                    onClick={() => setActive('soil')}
                    className="min-h-10 rounded-lg bg-[#356b30] px-4 py-2 text-sm font-semibold text-white"
                  >
                    Next: Soil &amp; threshold →
                  </button>
                </div>
              }
              dirty={changed([
                'name',
                'zoneName',
                'latitude',
                'longitude',
                'areaM2',
                'areaUnit',
                'cropType',
                'plantingDate',
              ])}
            >
              <p className="col-span-full -mt-3 mb-1 text-xs text-[#64748b]">
                Name, location, size, crop, and planting information.
              </p>
              <PlotIdentityForm {...props} />
              <div className="col-span-full grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
                <CropSettingsForm {...props} />
                <ThresholdSettingsForm {...props} section="base" />
              </div>
            </SettingsSection>
            {(props.errors.zoneName ||
              props.errors.initialGdd ||
              props.errors.startDate ||
              props.errors.period) && (
              <SettingsSection
                disabled={!available || disabled}
                title="Review saved values"
                dirty={false}
              >
                <Field
                  label="Zone name"
                  value={props.draft.zoneName}
                  onChange={(value) => props.update('zoneName', value)}
                  error={props.errors.zoneName}
                />
                <PhenologySettingsForm {...props} />
              </SettingsSection>
            )}
          </>
        )}
        {active === 'soil' && (
          <>
            <SettingsSection
              disabled={!available || disabled}
              title="2. Soil and threshold"
              footer={
                <div className="col-span-full">
                  <button
                    type="button"
                    onClick={() => setActive('device')}
                    className="min-h-10 rounded-lg bg-[#356b30] px-4 py-2 text-sm font-semibold text-white"
                  >
                    Next: Sensors &amp; irrigation →
                  </button>
                </div>
              }
              dirty={changed([
                'soilType',
                'soilTextureClass',
                'soilCalibration',
                'permanentWiltingPoint',
                'fieldCapacityLower',
                'fieldCapacityUpper',
                'saturation',
                'retentionText',
              ])}
            >
              <p className="col-span-full -mt-3 text-xs text-[#64748b]">
                Choose this plot's soil preset and how its irrigation threshold
                is determined.
              </p>
              <SoilSettingsForm
                {...props}
                threshold={<ThresholdSettingsForm {...props} section="mode" />}
              />
            </SettingsSection>
          </>
        )}
        {active === 'device' && (
          <>
            <SettingsSection
              disabled={!available || disabled}
              title="3. Sensors and irrigation mode"
              dirty={changed([
                'sensorKind',
                'moistureSensorIds',
                'temperatureSensorIds',
                'actuatorIds',
                'flowConfirmationSensorIds',
                'flowConfirmationMode',
                'slope',
                'irrigationMode',
              ])}
            >
              {discovering && <LoadingMessage message="Discovering devices…" />}
              <SensorSettingsForm
                {...props}
                devices={devices}
                discoveryError={discoveryError}
                onRetry={onRetry}
              />
              <IrrigationSettingsForm {...props} />
              <div className="col-span-full">{actions}</div>
            </SettingsSection>
          </>
        )}
      </div>
    </div>
  );
}
