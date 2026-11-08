import { chartWindow, selectChartRows } from '../chart/views';
import { clampRange, panRange, type TimeRange } from '../chart/navigation';
import { ChartTimeline } from './ChartTimeline';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChartData } from '../../../domain/chart';
import type { ThresholdStatus } from '../../../domain/recommendation';
import { formatDashboardTime } from '../../../shared/presentation';
import type { ResourceState } from '../types';
import type { ChartInstance, ForecastConfiguration } from '../chart/chartTypes';
import { buildForecastConfiguration, chartUnit } from '../chart/configuration';
import { loadChart } from '../chart/loadChart';
function CanvasChart({
  configuration,
  onPan,
  range,
  domain,
}: {
  configuration: ForecastConfiguration;
  onPan: (range: TimeRange) => void;
  range: TimeRange;
  domain: TimeRange;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const instance = useRef<ChartInstance | null>(null);
  const drag = useRef<{
    x: number;
    y: number;
    width: number;
    range: TimeRange;
  } | null>(null);
  useEffect(
    () => () => {
      instance.current?.destroy();
      instance.current = null;
    },
    [],
  );
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    if (instance.current) {
      instance.current.data = configuration.data;
      instance.current.options = configuration.options;
      instance.current.update('none');
      return;
    }
    void loadChart()
      .then((Chart) => {
        if (!cancelled && canvas.current) {
          if (!canvas.current.getContext('2d'))
            throw new Error('This browser could not create a canvas context.');
          instance.current = new Chart(canvas.current, configuration);
          setError(null);
          setLoaded(true);
        }
      })
      .catch((cause) => {
        if (!cancelled)
          setError(
            cause instanceof Error ? cause.message : 'Chart rendering failed.',
          );
      });
    return () => {
      cancelled = true;
    };
  }, [configuration, attempt]);
  return (
    <>
      {!loaded && !error && (
        <p className="my-3 leading-relaxed text-gray-600" role="status">
          Loading chart…
        </p>
      )}
      {error && (
        <div>
          <p className="my-3 leading-relaxed text-gray-600" role="alert">
            The chart couldn’t be displayed. Please try again.
          </p>
          <button
            type="button"
            className="min-h-11 cursor-pointer rounded-lg border border-[#aac9a5] bg-[#f5faf4] px-4 py-2.5 font-semibold text-[#2d5a27] hover:bg-[#e8f5e6] disabled:cursor-wait disabled:opacity-65"
            onClick={() => {
              setError(null);
              setLoaded(false);
              setAttempt((value) => value + 1);
            }}
          >
            Retry chart
          </button>
        </div>
      )}
      {!error && (
        <div
          className="relative h-85 w-full min-w-0 cursor-grab touch-pan-y active:cursor-grabbing min-[651px]:h-107.5"
          onPointerDown={(event) => {
            if (!event.isPrimary || event.button !== 0) return;
            drag.current = {
              x: event.clientX,
              y: event.clientY,
              width: event.currentTarget.getBoundingClientRect().width,
              range,
            };
          }}
          onPointerMove={(event) => {
            const initial = drag.current;
            if (!initial) return;
            const dx = event.clientX - initial.x;
            if (
              Math.abs(dx) < 8 ||
              Math.abs(dx) < Math.abs(event.clientY - initial.y)
            )
              return;
            event.currentTarget.setPointerCapture(event.pointerId);
            onPan(panRange(initial.range, domain, -dx / initial.width));
          }}
          onPointerUp={() => {
            drag.current = null;
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
          onPointerLeave={(event) => {
            if (!event.currentTarget.hasPointerCapture(event.pointerId))
              drag.current = null;
          }}
        >
          <canvas
            ref={canvas}
            role="img"
            aria-label="Observed and forecast soil conditions over time."
          />
        </div>
      )}
    </>
  );
}
export function ForecastChart({
  history,
  prediction,
  threshold,
  timezone,
}: {
  plantingDate?: string | null;
  history: ResourceState<ChartData>;
  threshold?: ResourceState<ThresholdStatus>;
  prediction: ResourceState<ChartData>;
  timezone: string;
  onRetry: () => void;
}) {
  const [customRange, setCustomRange] = useState<TimeRange | null>(null);
  const old =
    history.status === 'ready' && history.data.available ? history.data : null;
  const future =
    prediction.status === 'ready' && prediction.data.available
      ? prediction.data
      : null;
  const incompatible = !!(
    old &&
    future &&
    (chartUnit(old.unit) !== chartUnit(future.unit) || old.kind !== future.kind)
  );
  const forecast = incompatible ? null : future;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const refresh = () => setNow(Date.now());
    const timer = globalThis.setInterval(refresh, 60_000);
    globalThis.addEventListener('focus', refresh);
    return () => {
      globalThis.clearInterval(timer);
      globalThis.removeEventListener('focus', refresh);
    };
  }, []);
  const recent = useMemo(
    () => chartWindow(old, forecast, 'recent', now, null, timezone),
    [old, forecast, now, timezone],
  );
  const fullHistory = useMemo(
    () => selectChartRows(old, (time) => time <= now),
    [old, now],
  );
  const fullForecast = recent.prediction;
  const allTimes = [
    ...(fullHistory?.timestamps ?? []),
    ...(fullForecast?.timestamps ?? []),
  ].map(Date.parse);
  const domain = allTimes.length
    ? { start: Math.min(...allTimes), end: Math.max(...allTimes) }
    : null;
  const recentTimes = [
    ...(recent.history?.timestamps ?? []),
    ...(recent.prediction?.timestamps ?? []),
  ].map(Date.parse);
  const defaultRange =
    domain && recentTimes.length
      ? { start: Math.min(...recentTimes), end: Math.max(...recentTimes) }
      : domain;
  const range =
    domain && defaultRange
      ? clampRange(customRange ?? defaultRange, domain)
      : null;
  const selectedWindow =
    customRange && range
      ? {
          history: selectChartRows(
            fullHistory,
            (time) => time >= range.start && time <= range.end,
          ),
          prediction: selectChartRows(
            fullForecast,
            (time) => time >= range.start && time <= range.end,
          ),
        }
      : recent;
  const configuration =
    selectedWindow.history || selectedWindow.prediction
      ? buildForecastConfiguration(
          selectedWindow.history,
          selectedWindow.prediction,
          timezone,
          { showVwc: false, now },
        )
      : null;
  if (configuration && range) {
    configuration.options.scales.x!.min = range.start;
    configuration.options.scales.x!.max = range.end;
  }
  return (
    <section
      className="min-w-0 rounded-2xl border border-[#e0e0e0] bg-white p-4 shadow-[0_1px_2px_#00000014] sm:p-5 [&>h2]:mt-0 [&_details]:my-5 [&_summary]:cursor-pointer [&_summary]:py-2 [&_summary]:text-[#2d5a27]"
      aria-labelledby="plot-chart-title"
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2
          className="text-base font-bold text-[#111827]"
          id="plot-chart-title"
        >
          Prediction Chart
        </h2>
        {customRange && (
          <button
            type="button"
            className="min-h-11 rounded-md px-3 py-2 text-xs font-semibold text-[#2d5a27] hover:bg-[#f2f7f0]"
            onClick={() => setCustomRange(null)}
          >
            Back to current
          </button>
        )}
      </div>
      {threshold?.status === 'ready' && threshold.data.breached && (
        <p className="my-3 text-sm text-gray-600" role="status">
          Forecast first threshold crossing:{' '}
          {formatDashboardTime(threshold.data.timestamp, timezone)}
        </p>
      )}
      {history.status === 'loading' || prediction.status === 'loading' ? (
        <p className="my-3 leading-relaxed text-gray-600" role="status">
          Loading chart data…
        </p>
      ) : null}
      {incompatible && (
        <p className="my-3 leading-relaxed text-gray-600" role="status">
          Observation and forecast units or sensor kinds differ. Showing
          observations only.
        </p>
      )}
      {history.status !== 'loading' && !old && forecast && (
        <p className="my-3 leading-relaxed text-gray-600" role="status">
          Observations unavailable. Showing forecast only.
        </p>
      )}
      {prediction.status !== 'loading' && old && !forecast && !incompatible && (
        <p className="my-3 leading-relaxed text-gray-600" role="status">
          Forecast unavailable. Showing observations only.
        </p>
      )}
      {!configuration ? (
        <div className="flex min-h-80 flex-col justify-between py-2 sm:min-h-110">
          {history.status !== 'loading' && prediction.status !== 'loading' && (
            <p className="text-sm text-slate-600">
              {old || forecast
                ? 'No readings in this period.'
                : 'Chart data unavailable.'}
            </p>
          )}
        </div>
      ) : (
        <>
          {range && domain && (
            <CanvasChart
              configuration={configuration}
              range={range}
              domain={domain}
              onPan={setCustomRange}
            />
          )}
        </>
      )}
      {domain && range && (
        <ChartTimeline
          domain={domain}
          range={range}
          onChange={setCustomRange}
        />
      )}
    </section>
  );
}
