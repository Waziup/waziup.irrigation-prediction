export interface TimeRange {
  start: number;
  end: number;
}
export function clampRange(range: TimeRange, domain: TimeRange): TimeRange {
  const minimum = Math.min(3600000, domain.end - domain.start);
  const width = Math.min(
    domain.end - domain.start,
    Math.max(minimum, range.end - range.start),
  );
  const start = Math.max(
    domain.start,
    Math.min(domain.end - width, range.start),
  );
  return { start, end: start + width };
}
export function panRange(
  range: TimeRange,
  domain: TimeRange,
  fraction: number,
): TimeRange {
  const delta = (range.end - range.start) * fraction;
  return clampRange(
    { start: range.start + delta, end: range.end + delta },
    domain,
  );
}
