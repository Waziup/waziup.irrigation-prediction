import { useContext, useId } from 'react';
import { SettingsDataContext } from '../settingsDataContext';
import type { SectionProps } from '../types';
import { Field, SelectField } from './Fields';

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
        Irrigation threshold
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
      {d.thresholdMode === 'static' ? (
        <Field
          label="Static irrigation threshold (cbar)"
          type="number"
          min={1}
          step="0.1"
          value={d.thresholdCbar}
          onChange={(value) => update('thresholdCbar', value)}
          error={e.thresholdCbar}
        />
      ) : (
        <SelectField
          label="Irrigation method"
          value={d.irrigationType}
          options={[
            [
              [
                'drip',
                'micro_sprinkler',
                'sprinkler',
                'furrow',
                'flood',
              ].includes(d.irrigationType)
                ? ''
                : d.irrigationType,
              'Not specified',
            ],
            ['drip', 'Drip'],
            ['micro_sprinkler', 'Micro-sprinkler'],
            ['sprinkler', 'Sprinkler'],
            ['furrow', 'Furrow'],
            ['flood', 'Flood'],
          ]}
          onChange={(values) => update('irrigationType', values[0]!)}
        />
      )}
    </fieldset>
  );
}
