import type { BadgeTone } from '../components/StatusBadge';
import type { Urgency } from '../domain/alert';
export const urgencyPresentation: Record<
  Urgency,
  { label: string; tone: BadgeTone }
> = {
  ok: { label: 'OK', tone: 'success' },
  none: { label: 'No action needed', tone: 'success' },
  watch: { label: 'Watch', tone: 'warning' },
  advise: { label: 'Advisory', tone: 'warning' },
  critical: { label: 'Critical', tone: 'danger' },
  error: { label: 'Error', tone: 'danger' },
  unknown: { label: 'Unknown', tone: 'neutral' },
};
export function formatAmount(
  value: number | null,
  unit: string,
  digits = 1,
): string {
  return value === null
    ? 'Unavailable'
    : `${value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })} ${unit}`;
}
export function formatDashboardTime(
  value: string | null,
  timezone: string,
): string {
  if (!value || value === 'none') return 'Not forecast';
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return value;
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
      timeZone: timezone,
    }).format(date);
  } catch {
    return date.toISOString();
  }
}
