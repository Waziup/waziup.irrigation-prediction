import type { ChartData } from '../../../domain/chart';
export type ChartView = 'recent' | 'planting' | 'all';
type AvailableChart = Extract<ChartData, { available: true }>;

export function selectChartRows(
  data: AvailableChart | null,
  include: (time: number) => boolean,
): AvailableChart | null {
  if (!data) return null;
  const indices = data.timestamps
    .map((value, index) => ({ time: Date.parse(value), index }))
    .filter(({ time }) => Number.isFinite(time) && include(time))
    .sort((a, b) => a.time - b.time)
    .filter(
      (row, index, rows) => index === 0 || row.time !== rows[index - 1]!.time,
    )
    .map(({ index }) => index);
  if (!indices.length) return null;
  const series = (values: (number | null)[] | null) =>
    values === null ? null : indices.map((index) => values[index] ?? null);
  return {
    ...data,
    timestamps: indices.map((index) => data.timestamps[index]!),
    moistureSeries: series(data.moistureSeries)!,
    temperatureSeries: series(data.temperatureSeries),
    vwcSeries: series(data.vwcSeries),
    thresholdSeries: series(data.thresholdSeries),
    staticThresholdSeries: series(data.staticThresholdSeries),
    forecastHorizonsHours:
      data.forecastHorizonsHours === null
        ? null
        : indices.map((index) => data.forecastHorizonsHours![index]!),
  };
}

export function chartWindow(
  history: AvailableChart | null,
  prediction: AvailableChart | null,
  view: ChartView,
  now: number,
  plantingDate: string | null,
  timezone: string,
) {
  const dateFormatter = new Intl.DateTimeFormat('en-CA', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  const localDate = (time: number) => {
    const parts = dateFormatter.formatToParts(time);
    return ['year', 'month', 'day']
      .map((key) => parts.find((part) => part.type === key)!.value)
      .join('-');
  };
  const validPlanting =
    !!plantingDate &&
    /^\d{4}-\d{2}-\d{2}$/.test(plantingDate) &&
    Number.isFinite(Date.parse(plantingDate));
  const old = selectChartRows(
    history,
    (time) =>
      time <= now &&
      (view === 'recent'
        ? time >= now - 7 * 86400000
        : view === 'planting'
          ? validPlanting && localDate(time) >= plantingDate!
          : true),
  );
  // Use every actual future prediction; the model owns the forecast horizon.
  const future =
    view === 'recent'
      ? selectChartRows(prediction, (time) => time > now)
      : null;
  return { history: old, prediction: future, validPlanting };
}
