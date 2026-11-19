import { HelpTooltip } from '../../../components/HelpTooltip';
import { useContext, useId } from 'react';
import { SettingsDataContext } from '../settingsDataContext';
import type { SectionProps } from '../types';
import { Field } from './Fields';

export function ThresholdSettingsForm({
  draft: d,
  update,
  errors: e,
}: SectionProps) {
  const id = useId();
  const available = useContext(SettingsDataContext);
  return (
    <fieldset
      className="col-span-full grid min-w-0 gap-3"
      disabled={!available}
    >
      <legend className="mb-3 text-xs font-bold uppercase text-[#374151]">
        <span className="inline-flex items-center gap-2">
          Irrigation threshold
          <HelpTooltip
            label="Irrigation threshold"
            text="Choose Static to keep a field threshold you enter. Choose Dynamic to let the app calculate the threshold from the crop, soil and available weather."
          />
        </span>
      </legend>
      <div className="flex flex-wrap gap-x-8 gap-y-2">
        {(['static', 'dynamic'] as const).map((mode) => (
          <label
            key={mode}
            htmlFor={`${id}-${mode}`}
            className="flex min-h-11 cursor-pointer items-center gap-3 text-sm font-semibold"
          >
            <input
              id={`${id}-${mode}`}
              name={`${id}-threshold-mode`}
              type="radio"
              value={mode}
              checked={available && d.thresholdMode === mode}
              onChange={() => update('thresholdMode', mode)}
              className="size-4 accent-[#315f48]"
            />
            {mode === 'static' ? 'Static' : 'Dynamic'}
          </label>
        ))}
      </div>
      {e.thresholdMode && (
        <p role="alert" className="text-xs text-red-700">
          {e.thresholdMode}
        </p>
      )}
      {d.thresholdMode === 'static' && (
        <Field
          label="Static irrigation threshold (cbar)"
          help="The app uses this value unchanged. Higher cbar means drier soil. Use a threshold suitable for your field and crop."
          type="number"
          min={1}
          step="0.1"
          value={d.thresholdCbar}
          onChange={(value) => update('thresholdCbar', value)}
          error={e.thresholdCbar}
        />
      )}
    </fieldset>
  );
}
