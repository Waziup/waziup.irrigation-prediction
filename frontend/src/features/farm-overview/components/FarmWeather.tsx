import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { weatherApi } from '../../../api/weatherApi';
import type {
  WeatherForecast,
  WeatherView,
  WeatherRange,
} from '../../../domain/weather';
import { useFarmContext } from '../../farm-context/FarmContext';
import { LoadingMessage } from '../../app-shell/LoadingBoundary';
import { formatAmount } from '../types';
import { Icon } from '../../../components/Icon';

function weatherMessage(reason: string | null) {
  if (!reason || /HTTP|traceback|gateway request failed/i.test(reason))
    return 'The weather service could not be reached.';
  return (
    (
      {
        missing_farm_gps: 'Farm coordinates are unavailable.',
        fetch_failed: 'The weather service could not be reached.',
        no_data: 'No weather data was returned.',
        configuration_required: 'Complete farm setup to view weather.',
      } as Record<string, string>
    )[reason] ?? reason
  );
}
function WeatherSymbol({ icon }: { icon: string }) {
  if (icon.includes('rain'))
    return <Icon name="rain" className="size-8 text-sky-500" />;
  const cloudy = icon.includes('cloud');
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 40 40"
      className="size-9"
      fill="none"
      strokeWidth="2"
      strokeLinecap="round"
    >
      <g stroke="#d49920">
        <circle
          cx={cloudy ? 16 : 20}
          cy={cloudy ? 16 : 20}
          r="6"
          fill="#f9d469"
        />
        {cloudy ? (
          <path d="M16 3v3M3 16h3M6 6l3 3M26 6l-3 3" />
        ) : (
          <path d="M20 3v5M20 32v5M3 20h5M32 20h5M8 8l4 4M28 28l4 4M8 32l4-4M28 12l4-4" />
        )}
      </g>
      {cloudy && (
        <path
          d="M12 30a6 6 0 0 1 0-12 8 8 0 0 1 15-2 7 7 0 1 1 3 14Z"
          fill="#dbe8ec"
          stroke="#a6c1ca"
        />
      )}
    </svg>
  );
}
export function WeatherContent({
  forecast,
  history = false,
}: {
  forecast: WeatherForecast;
  history?: boolean;
}) {
  const timeline = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (history && timeline.current) {
      timeline.current.scrollLeft = timeline.current.scrollWidth;
    }
  }, [history, forecast.available]);
  if (!forecast.available)
    return (
      <p className="my-3 text-sm text-[#68766c]">
        Weather forecast unavailable. {weatherMessage(forecast.reason)}
      </p>
    );
  if (!forecast.days.length)
    return (
      <p className="my-3 text-sm text-[#68766c]">
        No weather days are available.
      </p>
    );
  return (
    <div
      ref={timeline}
      role="region"
      aria-label="Daily weather"
      tabIndex={0}
      className="flex min-w-0 snap-x snap-proximity gap-3 overflow-x-auto overscroll-x-contain pb-2 [scrollbar-width:thin] focus-visible:rounded-xl focus-visible:outline-2 focus-visible:outline-[#74936c]"
    >
      {forecast.days.map((day, index) => {
        const label =
          day.label === 'Today'
            ? 'Today'
            : index === 1 && forecast.days[0]?.label === 'Today'
              ? 'Tomorrow'
              : new Date(day.date).toLocaleDateString('en-GB', {
                  weekday: 'short',
                  day: 'numeric',
                  month: 'short',
                  timeZone: 'UTC',
                });
        const temperature =
          day.temperatureMinC !== null && day.temperatureMaxC !== null
            ? `${Math.round(day.temperatureMinC)}–${Math.round(day.temperatureMaxC)} °C`
            : formatAmount(
                day.temperatureC ?? day.temperatureMaxC ?? day.temperatureMinC,
                '°C',
              );
        return (
          <article
            key={day.date}
            aria-label={label}
            className="flex min-w-40 flex-1 shrink-0 snap-start flex-col items-center rounded-xl border border-[#dfe6ed] bg-[#f7f9fb] px-3 py-4 text-center"
          >
            <WeatherSymbol icon={day.icon} />
            <h3 className="mt-2 text-xs font-normal text-slate-500">
              <time dateTime={day.date} title={day.date}>
                {label}
              </time>
            </h3>
            <p
              className="mt-1 mb-2 text-base font-semibold tabular-nums text-[#253247]"
              aria-label={`Temperature ${temperature}`}
            >
              {temperature}
            </p>
            <dl className="grid gap-1 text-xs text-[#657d9b]">
              <div className="flex justify-center gap-1">
                <dt>Rain total</dt>
                <dd>{formatAmount(day.rainMm, 'mm')}</dd>
              </div>
              <div className="flex justify-center gap-1">
                <dt>Avg humidity</dt>
                <dd>{formatAmount(day.humidityPercent, '%')}</dd>
              </div>
              <div className="flex justify-center gap-1">
                <dt>Avg wind</dt>
                <dd>{formatAmount(day.windSpeedKmh, 'km/h')}</dd>
              </div>
            </dl>
          </article>
        );
      })}
    </div>
  );
}
export function FarmWeather() {
  const context = useFarmContext();
  const [view, setView] = useState<WeatherView>('forecast');
  const range: WeatherRange = view === 'forecast' ? 'model' : 'recent';
  const weather = useQuery({
    queryKey: [
      'farm-weather',
      view === 'forecast' ? context.currentPlot?.id : 'history',
      context.farm?.coordinates,
      view,
      range,
    ],
    queryFn: ({ signal }) =>
      weatherApi.getFarmForecast(
        { signal },
        { view, range, plotId: context.currentPlot!.id },
      ),
    enabled: context.status === 'ready' && !!context.currentPlot,
    retry: false,
    staleTime: 30 * 60_000,
    gcTime: 24 * 60 * 60_000,
    refetchOnMount: true,
    refetchOnWindowFocus: true,
    refetchInterval: 2 * 60 * 60_000,
    refetchIntervalInBackground: false,
  });
  const forecast = weather.data;
  const rangeDays = forecast?.rangeDays;
  const daysLabel = (days: number | null | undefined, fallback: string) =>
    days == null ? fallback : `${days} ${days === 1 ? 'day' : 'days'}`;
  const values = forecast?.available ? forecast.days : [];
  const rains = values
    .map((day) => day.rainMm)
    .filter((v): v is number => v !== null);
  const selectClass =
    'min-h-10 rounded-lg border border-[#d8e1d2] bg-white px-3 py-2 text-sm text-[#304b3b]';
  return (
    <section
      className="min-w-0 rounded-xl border border-[#e0e6d9] bg-white p-4 sm:p-5"
      aria-label="Farm weather"
    >
      <h2 className="mb-4 text-base font-bold text-[#253247]">
        {view === 'forecast' ? 'Farm weather forecast' : 'Farm weather history'}
      </h2>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <label className="grid gap-2 text-base font-semibold text-[#304537]">
          Weather view
          <select
            aria-label="Weather view"
            className={selectClass}
            value={view}
            onChange={(event) => {
              const next = event.target.value as WeatherView;
              setView(next);
            }}
          >
            <option value="forecast">Forecast</option>
            <option value="historical">History</option>
          </select>
        </label>
        <span className="pb-2 text-sm text-[#53645a]">
          {view === 'historical'
            ? 'Last 30 days'
            : daysLabel(rangeDays?.model, '5 days')}
        </span>
      </div>
      {forecast?.available &&
        (forecast.cacheState === 'stale' || weather.isError) && (
          <p role="status" className="mb-3 text-xs text-[#69756d]">
            Showing older saved weather data.
          </p>
        )}
      {weather.isError && !forecast?.available ? (
        <p role="alert" className="text-sm text-[#53645a]">
          Weather data could not be loaded.
        </p>
      ) : weather.isPending || !forecast ? (
        <LoadingMessage message="Loading weather…" />
      ) : !forecast.available ? (
        <p className="text-sm text-[#53645a]">
          {weatherMessage(forecast.reason)}
        </p>
      ) : (
        <>
          {view === 'historical' && (
            <div className="mb-4 rounded-lg bg-[#f3f6ef] p-3 text-sm text-[#304b3b]">
              Total rainfall:{' '}
              <strong>
                {rains.length === values.length && rains.length
                  ? formatAmount(
                      rains.reduce((total, value) => total + value, 0),
                      'mm',
                    )
                  : 'Unavailable'}
              </strong>
              <span className="ml-3">{values.length} days</span>
            </div>
          )}
          <WeatherContent
            key={`${view}:${range}`}
            forecast={forecast}
            history={view === 'historical'}
          />
        </>
      )}
    </section>
  );
}
