import { apiClient } from '../../../api/client';
import { parseFarmDashboard } from '../../../api/farmApi';
import { checkedCount } from '../../../api/adapters';
import {
  array,
  boolean,
  id,
  invalid,
  member,
  nonnegative,
  object,
  timestamp,
} from '../../../api/validation';
import type { FarmMap, SavedMapBoundary } from '../../../domain/map';
import type { RequestOptions } from '../../../domain/api';
import {
  parseGeometry,
  validatePlotBoundary,
} from '../../farm-setup/validation/farmSetupValidation';

export function parseSavedBoundary(value: unknown): SavedMapBoundary {
  const d = object(value);
  const areaM2 = nonnegative(d.area_m2);
  if (areaM2 === 0) invalid('area_m2', 'a positive area');
  return {
    geometry: parseGeometry(d.geometry),
    areaM2,
    geometryVersion: id(d.geometry_version),
    geometryUpdatedAt: timestamp(d.geometry_updated_at),
  };
}
export function parseFarmMap(value: unknown): FarmMap {
  const d = object(value);
  const schemaVersion = member(d.schema_version, ['1.0']);
  const registryRevision = checkedCount(d.registry_revision);
  const setup = object(d.setup);
  const complete = boolean(setup.complete);
  if (!complete) {
    if (
      setup.reason !== 'no_farm' ||
      d.farm !== null ||
      !Array.isArray(d.plots) ||
      d.plots.length ||
      d.summary !== null ||
      d.generated_at !== null
    )
      invalid('setup', 'an empty installation');
    return {
      schemaVersion,
      registryRevision,
      setup: { complete: false, reason: 'no_farm' },
      farm: null,
      plots: [],
      summary: null,
      generatedAt: null,
    };
  }
  if (setup.reason !== null) invalid('setup.reason', 'null');
  // Reuse the existing card and summary adapter; no second status contract.
  const dashboard = parseFarmDashboard({
    ...d,
    active_alerts: [],
    todays_plan: [],
  });
  const farm = { ...dashboard.farm, ...parseSavedBoundary(d.farm) };
  const boundaries = array(d.plots, parseSavedBoundary);
  const plots = dashboard.plots.map((card, index) => ({
    ...card,
    ...boundaries[index]!,
  }));
  if (
    !plots.length ||
    new Set(plots.map((p) => p.plotId)).size !== plots.length ||
    dashboard.summary.totalPlots !== plots.length
  )
    invalid('plots', 'unique plots matching the total');
  plots.forEach((plot, index) =>
    validatePlotBoundary(
      plot.geometry,
      farm.geometry,
      plots.slice(0, index).map((p) => p.geometry),
    ),
  );
  return {
    schemaVersion,
    registryRevision,
    setup: { complete: true, reason: null },
    farm,
    plots,
    summary: dashboard.summary,
    generatedAt: dashboard.generatedAt,
  };
}
export const farmMapApi = {
  load: (options?: RequestOptions) =>
    apiClient.json('farmMap', parseFarmMap, options),
};
