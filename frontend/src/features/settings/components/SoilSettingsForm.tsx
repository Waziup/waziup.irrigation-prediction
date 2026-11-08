import { soilWaterContents } from '../../../domain/soil';
import { useId } from 'react';
import type { ReactNode } from 'react';
import { soilPreset, soilTextureClasses } from '../../../domain/soilPresets';
import type { SectionProps } from '../types';
import { parseRetention } from '../validation/settingsValidation';
import { Field, SelectField } from './Fields';
export function SoilSettingsForm({
  draft: d,
  update,
  errors: e,
  threshold,
}: SectionProps & { threshold?: ReactNode }) {
  const id = useId();
  function apply(name: string) {
    const p = soilPreset(name);
    if (!p) return;
    update('soilType', p.soilType);
    update('soilTextureClass', p.soilTextureClass);
    update('soilCalibration', p.soilCalibration);
    update('permanentWiltingPoint', String(p.permanentWiltingPoint));
    update('fieldCapacityUpper', String(p.fieldCapacityUpper));
    update('fieldCapacityLower', String(p.fieldCapacityLower));
    update('saturation', String(p.saturation));
    update(
      'retentionText',
      p.retentionCurve.map((p) => `${p.tensionCbar}, ${p.vwc}`).join('\n'),
    );
  }
  function reset() {
    if (
      window.confirm(
        'Replace soil values and calibration with estimates for the selected soil type?',
      )
    )
      apply(d.soilType);
  }
  const source = String(d.soilCalibration.source ?? 'estimate');
  const points = parseRetention(d.retentionText);
  const contents =
    points &&
    [d.saturation, d.fieldCapacityLower, d.permanentWiltingPoint].every((v) =>
      v.trim(),
    )
      ? soilWaterContents({
          retentionCurve: points,
          saturation: Number(d.saturation),
          fieldCapacityLower: Number(d.fieldCapacityLower),
          permanentWiltingPoint: Number(d.permanentWiltingPoint),
        })
      : null;
  return (
    <>
      <SelectField
        label="Soil type"
        value={d.soilType}
        options={Object.keys(soilTextureClasses).map((v) => [
          v,
          v === 'Sandy_Soil' ? 'Sand' : v.replaceAll('_', ' '),
        ])}
        onChange={(v) => apply(v[0]!)}
        error={e.soilType}
      />
      {threshold}
      <details
        className="col-span-full"
        open={
          !!(
            e.retentionCurve ||
            e.soilCalibration ||
            e.permanentWiltingPoint ||
            e.fieldCapacityLower ||
            e.fieldCapacityUpper ||
            e.saturation
          )
        }
      >
        <summary className="cursor-pointer text-sm font-semibold text-[#374151]">
          Soil values &amp; source
        </summary>
        <div className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
          <SelectField
            label="Values obtained from"
            value={source}
            options={['estimate', 'laboratory', 'field'].map((v) => [
              v,
              v === 'estimate'
                ? 'Soil-type estimates'
                : v === 'laboratory'
                  ? 'Laboratory analysis'
                  : 'Field measurement',
            ])}
            onChange={(v) => {
              const next = v[0]!;
              if (next === 'estimate') {
                reset();
              } else
                update('soilCalibration', {
                  ...d.soilCalibration,
                  source: next,
                });
            }}
          />
          <Field
            label="Permanent wilting point (cbar)"
            type="number"
            min={0}
            value={d.permanentWiltingPoint}
            onChange={(v) => update('permanentWiltingPoint', v)}
            error={e.permanentWiltingPoint}
          />
          <Field
            label="Field capacity (cbar)"
            type="number"
            min={0}
            value={d.fieldCapacityLower}
            onChange={(v) => {
              update('fieldCapacityLower', v);
              update('fieldCapacityUpper', v);
            }}
            error={e.fieldCapacityLower ?? e.fieldCapacityUpper}
          />
          <Field
            label="Saturation (cbar)"
            type="number"
            min={0}
            value={d.saturation}
            onChange={(v) => update('saturation', v)}
            error={e.saturation}
          />
          <div className="flex min-w-0 flex-col gap-1.5 [&>label]:font-semibold col-span-full">
            <label htmlFor={id}>Soil water retention curve</label>
            <textarea
              className="min-h-11 w-full min-w-0 rounded-md border border-[#afbeb4] bg-white px-3 py-2.5 text-[#173c29] aria-invalid:border-[#a92323] disabled:bg-[#f0f2ef] font-mono"
              id={id}
              rows={6}
              value={d.retentionText}
              onChange={(v) => update('retentionText', v.target.value)}
              aria-invalid={!!e.retentionCurve}
              aria-describedby={`${id}-hint`}
            />
            <p
              id={`${id}-hint`}
              className={
                e.retentionCurve
                  ? 'text-[#a92323]'
                  : 'm-0 text-sm text-[#53645a]'
              }
            >
              {e.retentionCurve ??
                'cbar, VWC fraction. Example: 33, 0.30 = 30% VWC.'}
            </p>
            {contents && (
              <p className="my-3 leading-relaxed text-gray-600">
                Water content: saturation{' '}
                {(contents.saturation * 100).toFixed(1)}%; field capacity{' '}
                {(contents.fieldCapacity * 100).toFixed(1)}%; wilting point{' '}
                {(contents.wilting * 100).toFixed(1)}%. Available water:{' '}
                {(contents.availableWater * 100).toFixed(1)} percentage points.
              </p>
            )}
            {e.soilCalibration && (
              <p className="text-[#a92323]">{e.soilCalibration}</p>
            )}
            {Number(d.permanentWiltingPoint) !== 1500 &&
              source === 'estimate' && (
                <p className="my-3 leading-relaxed text-gray-600">
                  Select a field or laboratory source for a nonstandard wilting
                  point, or restore soil-type estimates.
                </p>
              )}
          </div>
        </div>
      </details>
    </>
  );
}
