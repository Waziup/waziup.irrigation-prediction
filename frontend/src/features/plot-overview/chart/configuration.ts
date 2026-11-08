import type { ChartData } from '../../../domain/chart';
import type { ForecastConfiguration, LineDataset } from './chartTypes';
import { soilDecorations } from './soilDecorations';
import { formatDashboardTime } from '../../../shared/presentation';
type AvailableChart = Extract<ChartData, { available: true }>;
export function buildForecastConfiguration(
  history: AvailableChart | null,
  prediction: AvailableChart | null,
  timezone: string,
  display: { showVwc?: boolean; now?: number | undefined } = {},
): ForecastConfiguration {
  const historicalCount = history?.timestamps.length ?? 0;
  const predictedCount = prediction?.timestamps.length ?? 0;
  const metadata = prediction ?? history;
  if (!metadata) throw new Error('No chart data is available.');
  const total = historicalCount + predictedCount;
  const observed = history?.moistureSeries ?? [];
  const forecast = prediction?.moistureSeries ?? [];
  const line = (
    label: string,
    data: (number | null)[],
    color: string,
    predicted = false,
    axis = 'y',
  ): LineDataset => ({
    label,
    data,
    borderColor: color,
    borderWidth: 3,
    borderDash: predicted ? [7, 5] : [],
    pointRadius: 0,
    pointHoverRadius: 4,
    tension: 0.2,
    yAxisID: axis,
    spanGaps: false,
  });
  const joined = (old: (number | null)[], future: (number | null)[]) =>
    Array.from({ length: historicalCount }, (_, index) =>
      index === historicalCount - 1 ? (old[index] ?? null) : null,
    ).concat(future);
  const datasets: LineDataset[] = [];
  const tension = metadata.kind === 'tension' || metadata.kind === 'both';
  const activeThreshold =
    metadata.thresholdCbar ??
    (metadata.thresholdMode === 'static' ? metadata.staticThresholdCbar : null);
  if (
    tension &&
    activeThreshold !== null &&
    (metadata.thresholdMode !== 'dynamic' || predictedCount > 0)
  ) {
    const threshold = activeThreshold;
    if (metadata.thresholdMode === 'dynamic') {
      const projected = prediction?.thresholdSeries;
      const future =
        projected?.length === predictedCount
          ? projected
          : Array<number | null>(predictedCount).fill(threshold);
      datasets.push(
        line(
          'Dynamic forecast trigger',
          predictedCount
            ? Array<number | null>(historicalCount).fill(null).concat(future)
            : Array<number | null>(total).fill(threshold),
          '#ff4560',
          true,
        ),
      );
    } else
      datasets.push(
        line(
          'Static field trigger',
          Array<number | null>(total).fill(threshold),
          '#ff4560',
        ),
      );
  }
  if (history)
    datasets.push(
      line(
        'Observed',
        observed.concat(Array<null>(predictedCount).fill(null)),
        '#3b82b6',
      ),
    );
  if (prediction)
    datasets.push(
      line('Forecast', joined(observed, forecast), '#3b82b6', true),
    );
  const oldVwc = history?.vwcSeries ?? Array<null>(historicalCount).fill(null);
  const futureVwc =
    prediction?.vwcSeries ?? Array<null>(predictedCount).fill(null);
  const hasVwc =
    display.showVwc !== false &&
    oldVwc.concat(futureVwc).some((value) => value !== null);
  if (hasVwc) {
    if (history)
      datasets.push(
        line(
          'Historical VWC',
          oldVwc.concat(Array<null>(predictedCount).fill(null)),
          '#34d399',
          false,
          'vwc',
        ),
      );
    if (prediction)
      datasets.push(
        line('Forecast VWC', joined(oldVwc, futureVwc), '#2dd4bf', true, 'vwc'),
      );
  }
  return {
    type: 'line',
    data: {
      labels: [
        ...(history?.timestamps ?? []),
        ...(prediction?.timestamps ?? []),
      ].map((time) => Date.parse(time)),
      datasets,
    },
    plugins: [soilDecorations],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      layout: { padding: { bottom: 45 } },
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          callbacks: {
            title: (items) =>
              items[0]
                ? formatDashboardTime(
                    new Date(items[0].parsed.x).toISOString(),
                    timezone,
                  )
                : '',
          },
        },
        soilChartDecorations: {
          enabled: tension,
          nowTimestamp: display.now,
          historyCount: historicalCount && predictedCount ? historicalCount : 0,
          references: {
            saturation: metadata.saturation,
            lower: metadata.fieldCapacityLower,
            upper: metadata.fieldCapacityUpper,
            wilting: metadata.permanentWiltingPoint,
            threshold: activeThreshold,
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          min: Math.min(
            ...[
              ...(history?.timestamps ?? []),
              ...(prediction?.timestamps ?? []),
            ].map(Date.parse),
          ),
          max: Math.max(
            ...[
              ...(history?.timestamps ?? []),
              ...(prediction?.timestamps ?? []),
            ].map(Date.parse),
          ),
          ticks: {
            maxTicksLimit: 6,
            maxRotation: 0,
            callback: (value) =>
              new Intl.DateTimeFormat('en', {
                timeZone: timezone,
                day: 'numeric',
                month: 'short',
              }).format(Number(value)),
          },
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: tension
              ? `Soil tension (${chartUnit(metadata.unit)})`
              : `Soil moisture (${metadata.unit})`,
          },
        },
        vwc: {
          display: hasVwc,
          position: 'right',
          beginAtZero: true,
          grid: { drawOnChartArea: false },
          title: { display: hasVwc, text: 'Volumetric water content' },
        },
      },
    },
  };
}

export function chartUnit(unit: string) {
  return /cbar|centibar|kpa|soil tension/i.test(unit)
    ? 'cbar'
    : unit.trim().toLowerCase();
}
