// Use the response's evaluation time, rather than advancing potentially stale data.
export function growthDays(
  reference: string | null,
  evaluatedAt: string | null,
  timezone: string,
): number | null {
  if (!reference || !evaluatedAt) return null;
  const start = Date.parse(`${reference.slice(0, 10)}T00:00:00Z`);
  const evaluated = new Date(evaluatedAt);
  if (!Number.isFinite(start) || !Number.isFinite(evaluated.getTime()))
    return null;
  if (new Date(start).toISOString().slice(0, 10) !== reference.slice(0, 10))
    return null;
  try {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).formatToParts(evaluated);
    const part = (type: Intl.DateTimeFormatPartTypes) =>
      Number(parts.find((item) => item.type === type)?.value);
    const end = Date.UTC(part('year'), part('month') - 1, part('day'));
    const days = Math.round((end - start) / 86_400_000);
    return days >= 0 ? days : null;
  } catch {
    return null;
  }
}
