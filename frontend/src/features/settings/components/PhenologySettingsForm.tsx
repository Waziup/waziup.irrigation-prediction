import { Link } from 'react-router';
import type { SectionProps } from '../types';
import { Field } from './Fields';
export function PhenologySettingsForm({
  draft: d,
  update,
  errors: e,
}: SectionProps) {
  return (
    <>
      <Field
        label="Initial growing degree days (GDD)"
        type="number"
        value={d.initialGdd}
        onChange={(v) => update('initialGdd', v)}
        error={e.initialGdd}
      />
      <Field
        label="Training window start (ISO date/time, optional)"
        value={d.startDate}
        onChange={(v) => update('startDate', v)}
        error={e.startDate}
        hint="Preserved independently of planting date. Empty uses the backend default for a new plot."
      />
      <Field
        label="Training period (days)"
        type="number"
        min={1}
        step="1"
        value={d.period}
        onChange={(v) => update('period', v)}
        error={e.period}
      />
      <p className="col-span-full">
        Crop stages and coefficients remain in the backend crop library.{' '}
        <Link
          className="text-[#2d5a27] underline-offset-4 hover:underline"
          to="/phenology"
        >
          View growth and water demand
        </Link>
        .
      </p>
    </>
  );
}
