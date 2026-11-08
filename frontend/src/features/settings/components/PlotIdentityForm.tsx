import {
  areaUnits,
  fromSquareMetres,
  toSquareMetres,
} from '../../../domain/plot';
import type { AreaUnit } from '../../../domain/plot';
import type { SectionProps } from '../types';
import { Field, SelectField } from './Fields';
export function PlotIdentityForm({
  draft: d,
  update,
  errors: e,
}: SectionProps) {
  function unit(value: AreaUnit) {
    if (
      d.areaM2.trim() &&
      Number.isFinite(Number(d.areaM2)) &&
      Number(d.areaM2) >= 0
    )
      update(
        'areaM2',
        String(
          fromSquareMetres(toSquareMetres(Number(d.areaM2), d.areaUnit), value),
        ),
      );
    update('areaUnit', value);
  }
  return (
    <>
      <Field
        className="col-span-full w-full sm:max-w-72"
        label="Plot name"
        value={d.name}
        onChange={(v) => update('name', v)}
        error={e.name}
      />
      <Field
        label="Latitude"
        type="number"
        min={-90}
        max={90}
        value={d.latitude}
        onChange={(v) => update('latitude', v)}
        error={e.latitude ?? e.coordinates}
      />
      <Field
        label="Longitude"
        type="number"
        min={-180}
        max={180}
        value={d.longitude}
        onChange={(v) => update('longitude', v)}
        error={e.longitude ?? e.coordinates}
      />
      <Field
        label="Plot area"
        decimalPlaces={2}
        type="number"
        min={0}
        value={d.areaM2}
        onChange={(v) => update('areaM2', v)}
        error={e.areaM2}
      />
      <div className="col-span-full grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
        <SelectField
          label="Area unit"
          value={d.areaUnit}
          options={areaUnits.map((v) => [
            v,
            v === 'm2' ? 'm²' : v === 'ha' ? 'Hectares' : 'Acres',
          ])}
          onChange={(v) => unit(v[0] as AreaUnit)}
        />
      </div>
    </>
  );
}
