import type { SectionProps } from '../types';
import { Field, SelectField } from './Fields';
export function CropSettingsForm({
  draft: d,
  update,
  errors: e,
}: SectionProps) {
  return (
    <>
      <SelectField
        label="Crop"
        value={d.cropType}
        options={[
          '',
          'maize',
          'beans',
          'wheat',
          'tomato',
          'sorghum',
          'rice',
          'olive',
        ].map((v) => [v, v || 'Not planted / no crop'])}
        onChange={(v) => {
          update('cropType', v[0]!);
          if (!v[0]) update('plantingDate', '');
        }}
        error={e.cropType}
      />
      <Field
        label="Planting date"
        type="date"
        value={d.cropType ? d.plantingDate : ''}
        required={!!d.cropType}
        disabled={!d.cropType}
        onChange={(v) => update('plantingDate', v)}
        error={e.plantingDate}
      />
    </>
  );
}
