import { HelpTooltip } from '../../../components/HelpTooltip';
import { useContext, useId, useState } from 'react';
import { SettingsDataContext } from '../settingsDataContext';
import type { ReactNode } from 'react';
export function Field({
  label,
  value,
  onChange,
  error,
  type = 'text',
  hint,
  help,
  step,
  min,
  max,
  className,
  disabled = false,
  required = false,
  decimalPlaces,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  error?: string | undefined;
  type?: string;
  hint?: string;
  help?: string;
  step?: string;
  min?: number;
  max?: number;
  className?: string;
  disabled?: boolean;
  required?: boolean;
  decimalPlaces?: number;
}) {
  const id = useId();
  const [editing, setEditing] = useState(false);
  const displayedValue =
    !editing &&
    decimalPlaces !== undefined &&
    value.trim() &&
    Number.isFinite(Number(value))
      ? Number(value).toFixed(decimalPlaces)
      : value;
  return (
    <div
      className={`flex min-w-0 flex-col gap-1.5 [&>label]:text-xs [&>label]:font-bold [&>label]:uppercase [&>label]:tracking-wide [&>label]:text-[#374151] ${className ?? ''}`}
    >
      <div className="flex items-start justify-between gap-2">
        <label
          htmlFor={id}
          className="text-xs font-bold uppercase tracking-wide text-[#374151]"
        >
          {label}
        </label>
        {help && <HelpTooltip label={label} text={help} id={`${id}-tooltip`} />}
      </div>
      <input
        className="min-h-11 w-full min-w-0 rounded-lg border border-[#d8dee8] bg-white px-3.5 py-2.5 text-sm text-[#1f2937] aria-invalid:border-[#a92323] disabled:bg-[#f9fafb]"
        id={id}
        type={type}
        disabled={disabled}
        required={required}
        value={displayedValue}
        onFocus={() => setEditing(true)}
        onBlur={() => setEditing(false)}
        onChange={(e) => onChange(e.target.value)}
        step={step ?? (type === 'number' ? 'any' : undefined)}
        min={min}
        max={max}
        aria-invalid={!!error}
        aria-describedby={
          [error || hint ? `${id}-help` : null, help ? `${id}-tooltip` : null]
            .filter(Boolean)
            .join(' ') || undefined
        }
      />
      {(error || hint) && (
        <p
          id={`${id}-help`}
          className={error ? 'text-[#a92323]' : 'm-0 text-sm text-[#53645a]'}
        >
          {error ?? hint}
        </p>
      )}
    </div>
  );
}
export function SelectField({
  label,
  value,
  options,
  onChange,
  error,
  multiple = false,
  help,
}: {
  label: string;
  value: string | string[];
  options: readonly (readonly [string, string])[];
  onChange: (values: string[]) => void;
  error?: string | undefined;
  multiple?: boolean;
  help?: string;
}) {
  const id = useId();
  const available = useContext(SettingsDataContext);
  const displayValue = available ? value : multiple ? [] : '';
  const displayOptions = available
    ? options
    : options.filter(([v]) => v !== '');
  const selected =
    typeof displayValue === 'string' ? [displayValue] : displayValue;
  const missing = selected.filter(
    (v) => !displayOptions.some((o) => o[0] === v),
  );
  return (
    <div className="flex min-w-0 flex-col gap-1.5 [&>label]:text-xs [&>label]:font-bold [&>label]:uppercase [&>label]:tracking-wide [&>label]:text-[#374151]">
      <div className="flex items-start justify-between gap-2">
        <label
          htmlFor={id}
          className="text-xs font-bold uppercase tracking-wide text-[#374151]"
        >
          {label}
        </label>
        {help && <HelpTooltip label={label} text={help} id={`${id}-help`} />}
      </div>
      <select
        className="min-h-11 w-full min-w-0 rounded-lg border border-[#d8dee8] bg-white px-3.5 py-2.5 text-sm text-[#1f2937] aria-invalid:border-[#a92323] disabled:bg-[#f9fafb] **:text-gray-800 [&[multiple]]:min-h-36"
        id={id}
        value={displayValue}
        multiple={multiple}
        onChange={(e) =>
          onChange(Array.from(e.target.selectedOptions, (o) => o.value))
        }
        aria-invalid={!!error}
        aria-describedby={
          [error ? `${id}-error` : null, help ? `${id}-help` : null]
            .filter(Boolean)
            .join(' ') || undefined
        }
      >
        {missing.map((v) => (
          <option key={v} value={v}>
            {v ? `${v} (saved value)` : '—'}
          </option>
        ))}
        {displayOptions.map(([v, name]) => (
          <option key={v} value={v}>
            {name}
          </option>
        ))}
      </select>
      {error && (
        <p id={`${id}-error`} className="text-[#a92323]">
          {error}
        </p>
      )}
    </div>
  );
}
export function SettingsSection({
  title,
  dirty,
  children,
  disabled = false,
  footer,
}: {
  title: string;
  dirty: boolean;
  children: ReactNode;
  disabled?: boolean;
  footer?: ReactNode;
}) {
  const id = useId();
  return (
    <section
      className="min-w-0 rounded-2xl border border-[#e0e0e0] bg-white p-5 shadow-[0_2px_8px_#00000014] sm:p-6 [&>h2]:mt-0 [&_p]:text-xs [&_p]:leading-relaxed"
      aria-labelledby={id}
    >
      <div className="mb-5 flex flex-wrap items-baseline justify-between gap-3 [&>span]:text-sm [&>span]:text-[#795018]">
        <h2 className="text-base font-bold text-[#333333]" id={id}>
          {title}
        </h2>
        {dirty && <span>Unsaved changes</span>}
      </div>
      <fieldset
        disabled={disabled}
        className="m-0 grid min-w-0 grid-cols-1 gap-x-5 gap-y-5 border-0 p-0 sm:grid-cols-2 xl:grid-cols-3"
      >
        {children}
      </fieldset>
      {footer && <div className="mt-5">{footer}</div>}
    </section>
  );
}
