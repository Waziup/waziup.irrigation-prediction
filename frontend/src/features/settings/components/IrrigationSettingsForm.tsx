import { useContext, useId } from 'react';
import { SettingsDataContext } from '../settingsDataContext';
import type { IrrigationMode } from '../../../domain/operation';
import type { SectionProps } from '../types';
export function IrrigationSettingsForm({
  draft: d,
  update,
  errors: e,
}: SectionProps) {
  const available = useContext(SettingsDataContext);
  const id = useId();
  const requiresActuator = ['automatic', 'approval_required'].includes(
    d.irrigationMode,
  );
  const actuatorMessage =
    'This mode requires an actuator. Select a pump or valve under “Water flow sensor”, then save.';
  const actuatorError =
    requiresActuator && !d.actuatorIds.length
      ? actuatorMessage
      : (e.actuatorIds ?? e.actuator);
  const modes: [IrrigationMode, string, string][] = [
    [
      'automatic',
      'Full-auto',
      'The app starts irrigation and stops at the calculated amount.',
    ],
    [
      'manual',
      'Semi-auto',
      'You start and stop irrigation from the app using a selected pump or valve.',
    ],
    [
      'advisory_only',
      'Off',
      'The app gives advice but does not operate irrigation hardware.',
    ],
  ];
  if (d.irrigationMode === 'approval_required')
    modes.push([
      'approval_required',
      'Approval required',
      'Irrigation requires your approval.',
    ]);
  return (
    <>
      <fieldset className="col-span-full min-w-0 border-0 p-0">
        <legend className="mb-3 text-xs font-bold uppercase text-[#374151]">
          Irrigation mode
        </legend>
        <div className="grid gap-3 sm:grid-cols-3">
          {modes.map(([mode, label, hint]) => (
            <label
              key={mode}
              className={`flex cursor-pointer items-start gap-3 rounded-xl border p-4 ${available && d.irrigationMode === mode ? 'border-[#315f48] bg-[#edf3eb]' : 'border-[#d8dee8] bg-white'}`}
            >
              <input
                type="radio"
                name={`${id}-mode`}
                value={mode}
                checked={available && d.irrigationMode === mode}
                aria-describedby={
                  actuatorError ? `${id}-actuator-error` : undefined
                }
                onChange={() => update('irrigationMode', mode)}
                className="mt-1 size-4 shrink-0 accent-[#315f48]"
              />
              <span>
                <strong className="text-sm">{label}</strong>
                <small className="mt-1 block text-xs leading-relaxed text-[#64748b]">
                  {hint}
                </small>
              </span>
            </label>
          ))}
        </div>
        {actuatorError && (
          <p
            id={`${id}-actuator-error`}
            role="alert"
            className="mt-3 text-sm text-red-700"
          >
            {actuatorError}
          </p>
        )}
        {d.irrigationMode === 'manual' && !d.actuatorIds.length && (
          <p className="mt-3 text-sm text-[#53645a]">
            Select a controller under “Water flow sensor” to use Semi-auto
            controls.
          </p>
        )}
        {e.irrigationMode && (
          <p role="alert" className="mt-2 text-xs text-red-700">
            {e.irrigationMode}
          </p>
        )}
      </fieldset>
    </>
  );
}
