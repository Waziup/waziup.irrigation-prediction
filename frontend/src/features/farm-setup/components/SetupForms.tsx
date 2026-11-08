import { BoundaryPreview } from './BoundaryPreview';
import { useId } from 'react';
import type { SetupDraft } from '../types';
import { areaLabel, previewArea } from '../validation/farmSetupValidation';

const input =
  'min-h-11 w-full min-w-0 rounded-lg border border-[#cdd8ca] bg-white px-3 py-2 text-[#243b29] focus:outline-2 focus:outline-[#548f3c]';
export function SetupField({
  label,
  value,
  onChange,
  type = 'text',
  required = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  required?: boolean;
}) {
  const id = useId();
  return (
    <label htmlFor={id} className="grid min-w-0 gap-2 text-sm font-semibold">
      {label}
      <input
        id={id}
        className={input}
        type={type}
        step={type === 'number' ? 'any' : undefined}
        value={value}
        required={required}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}
export function FarmDetailsForm({
  draft,
  update,
}: {
  draft: SetupDraft;
  update: (fields: Partial<SetupDraft>) => void;
}) {
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      <SetupField
        label="Farm name"
        value={draft.name}
        onChange={(name) => update({ name })}
        required
      />
    </div>
  );
}
export function PlotNameForm({
  draft,
  update,
}: {
  draft: SetupDraft;
  update: (fields: Partial<SetupDraft>) => void;
}) {
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      <SetupField
        label="Plot name"
        value={draft.plotName}
        onChange={(plotName) => update({ plotName })}
        required
      />
    </div>
  );
}
export function SetupReview({
  draft,
  mode,
}: {
  draft: SetupDraft;
  mode: 'farm' | 'plot';
}) {
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      {draft.farmGeometry && draft.plotGeometry && (
        <BoundaryPreview farm={draft.farmGeometry} plot={draft.plotGeometry} />
      )}
      {mode === 'farm' && (
        <section className="rounded-xl bg-[#edf4e8] p-5">
          <h3 className="font-semibold">{draft.name}</h3>
          <p className="mt-2 text-sm">
            Farm boundary:{' '}
            {draft.farmGeometry
              ? `approximately ${areaLabel(previewArea(draft.farmGeometry))}`
              : 'Not drawn'}
          </p>
        </section>
      )}
      <section className="rounded-xl border border-[#dce5d7] p-5">
        <h3 className="font-semibold">{draft.plotName}</h3>
        <p className="mt-2 text-sm">
          Plot boundary:{' '}
          {draft.plotGeometry
            ? `approximately ${areaLabel(previewArea(draft.plotGeometry))}`
            : 'Not drawn'}
        </p>
      </section>
      <p className="text-sm text-[#53645a] sm:col-span-2">
        Check the names and boundaries before saving. Final areas will be
        calculated from the saved boundaries.
      </p>
    </div>
  );
}
