import { useState, useCallback, useEffect } from 'react';
import { useBeforeUnload, useBlocker } from 'react-router';
import { useFarmContext } from '../../farm-context/FarmContext';
import { PlotRemoval } from './PlotRemoval';
import { FarmManagement } from './FarmManagement';
import { SettingsWorkspace } from './SettingsWorkspace';
import { SettingsDataContext } from '../settingsDataContext';
import { TrainingControls } from './TrainingControls';
import type { SettingsDraft } from '../types';
import { PlotSettingsSections } from './PlotSettingsSections';
// Empty form state only: no resource identity, configuration version or server values.
const emptyDraft: SettingsDraft = {
  name: '',
  zoneName: '',
  latitude: '',
  longitude: '',
  areaM2: '',
  areaUnit: 'm2',
  sensorKind: '',
  moistureSensorIds: [],
  temperatureSensorIds: [],
  actuatorIds: [],
  flowConfirmationSensorIds: [],
  flowConfirmationMode: 'event',
  slope: '',
  thresholdCbar: '',
  thresholdMode: 'static',
  irrigationType: '',
  irrigationMode: 'manual',
  applicationEfficiency: '',
  effectiveRainfallFraction: '',
  lookaheadHours: '',
  startDate: '',
  period: '',
  soilType: '',
  soilTextureClass: null,
  soilCalibration: {},
  permanentWiltingPoint: '',
  fieldCapacityUpper: '',
  fieldCapacityLower: '',
  saturation: '',
  retentionText: '',
  cropType: '',
  plantingDate: '',
  initialGdd: '',
};
export function EmptyPlotSettings({
  editable = true,
  onEditingChange,
}: {
  editable?: boolean;
  onEditingChange?: (editing: boolean) => void;
}) {
  const context = useFarmContext();
  const [draft, setDraft] = useState(emptyDraft);
  const dirty = JSON.stringify(draft) !== JSON.stringify(emptyDraft);
  const blocker = useBlocker(dirty);
  useEffect(() => {
    if (blocker.state !== 'blocked') return;
    if (window.confirm('Discard unsaved changes?')) blocker.proceed();
    else blocker.reset();
  }, [blocker]);
  useBeforeUnload(
    useCallback(
      (event: BeforeUnloadEvent) => {
        if (dirty) {
          event.preventDefault();
          event.returnValue = '';
        }
      },
      [dirty],
    ),
  );
  function discard() {
    if (dirty && !window.confirm('Discard unsaved changes?')) return;
    setDraft(emptyDraft);
    onEditingChange?.(false);
    void context.refresh().catch(() => undefined);
  }
  return (
    <SettingsDataContext.Provider value={true}>
      <div className="grid gap-5">
        <FarmManagement disabled preview={editable} />
        <SettingsWorkspace>
          <div className="grid min-w-0 gap-4">
            <PlotSettingsSections
              disabled={!editable}
              actions={
                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    disabled
                    title="Load this plot’s saved settings from the updated backend before saving."
                    className="min-h-11 rounded-lg bg-[#356b30] px-4 py-2 text-sm font-semibold text-white disabled:opacity-55"
                  >
                    Save all plot settings
                  </button>
                  <button
                    type="button"
                    onClick={discard}
                    className="min-h-11 rounded-lg border border-[#d8dee8] px-4 py-2 text-sm"
                  >
                    Cancel
                  </button>
                </div>
              }
              props={{
                draft,
                errors: {},
                update: (key, value) => {
                  setDraft((previous) => ({ ...previous, [key]: value }));
                  onEditingChange?.(true);
                },
              }}
              changed={(keys) =>
                keys.some(
                  (key) =>
                    JSON.stringify(draft[key]) !==
                    JSON.stringify(emptyDraft[key]),
                )
              }
              devices={[]}
              discoveryError={null}
              discovering={false}
              onRetry={() => undefined}
            />
          </div>
        </SettingsWorkspace>
        <TrainingControls
          plotName="this plot"
          readiness={null}
          dirty={false}
          saving={false}
          loading={false}
          error={null}
          state={undefined}
          onRetry={() => undefined}
          onStart={() => undefined}
          startPending={false}
          startError={null}
        />
        <PlotRemoval disabled />
      </div>
    </SettingsDataContext.Provider>
  );
}
