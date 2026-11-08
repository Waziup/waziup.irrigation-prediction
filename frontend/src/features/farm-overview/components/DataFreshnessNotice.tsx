import { adviceSources } from '../../../shared/adviceWarnings';
import type { FarmPlotCard } from '../../../domain/farm';
const sourceNames: Record<string, string> = {
  weather: 'weather data',
  sensors: 'sensor readings',
  sensor: 'sensor readings',
  soil: 'soil readings',
  satellite: 'satellite observations',
  eo: 'satellite observations',
  model: 'prediction data',
  prediction: 'prediction data',
};
export function DataFreshnessNotice({
  count,
  plots = [],
}: {
  count: number;
  plots?: FarmPlotCard[];
}) {
  const advicePlots = plots.map((plot) => ({
    ...plot,
    dataSources: adviceSources(plot.dataSources ?? []),
    satelliteOnly:
      (plot.dataSources?.length ?? 0) > 0 &&
      adviceSources(plot.dataSources ?? []).length === 0,
  }));
  const affected = advicePlots.filter(
    (plot) =>
      !plot.satelliteOnly &&
      plot.dataState !== 'fresh' &&
      (!plot.setupState || plot.setupState === 'complete'),
  );
  count = plots.length ? affected.length : count;
  if (count === 0) return null;
  return (
    <section
      className="text-xs leading-relaxed text-[#806536]"
      role="status"
      aria-label="Irrigation data availability"
    >
      <p>
        {count} {count === 1 ? 'plot has' : 'plots have'} old or missing data
        for irrigation advice.
      </p>
      <ul className="mt-1 grid gap-0.5">
        {affected.map((plot) => (
          <li key={plot.plotId}>
            <span className="font-semibold">{plot.name}: </span>
            {plot.dataSources?.length
              ? `${plot.dataSources.map((source) => sourceNames[source] ?? source.replaceAll('_', ' ')).join(', ')} ${plot.dataState === 'stale' ? 'is out of date' : 'is unavailable'}.`
              : plot.dataReason ||
                (plot.dataState === 'stale'
                  ? 'Irrigation source data is out of date; the service has not identified the source.'
                  : 'Irrigation advice is unavailable; the service has not identified the missing source.')}
          </li>
        ))}
      </ul>
    </section>
  );
}
