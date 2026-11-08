import { useContext, useId } from 'react';
import { SettingsDataContext } from '../settingsDataContext';
import type { SectionProps } from '../types';
import { Field } from './Fields';
export function ThresholdSettingsForm({
  draft: d,
  update,
  errors: e,
  section,
}: SectionProps & { section?: 'base' | 'mode' }) {
  const id = useId();
  const available = useContext(SettingsDataContext);
  return (
    <>
      {section !== 'mode' && (
        <Field
          label="Base irrigation threshold (cbar)"
          type="number"
          min={1}
          step="0.1"
          value={d.thresholdCbar}
          onChange={(v) => update('thresholdCbar', v)}
          error={e.thresholdCbar}
        />
      )}
      {section !== 'base' && (
        <div className="col-span-full grid gap-2">
          <span className="text-xs font-bold uppercase text-[#374151]">
            Threshold mode
          </span>
          <label
            htmlFor={id}
            className="flex items-center gap-3 text-sm font-semibold"
          >
            <input
              id={id}
              type="checkbox"
              checked={available && d.thresholdMode === 'dynamic'}
              onChange={(event) =>
                update(
                  'thresholdMode',
                  event.target.checked ? 'dynamic' : 'static',
                )
              }
              className="size-4 accent-[#315f48]"
            />
            Use dynamic tension thresholds
          </label>
          <p className="text-xs text-[#64748b]">
            On: adjusts the trigger using crop demand and soil data. Off: uses
            your fixed tension threshold.
          </p>
          {e.thresholdMode && (
            <p role="alert" className="text-xs text-red-700">
              {e.thresholdMode}
            </p>
          )}
        </div>
      )}
    </>
  );
}
