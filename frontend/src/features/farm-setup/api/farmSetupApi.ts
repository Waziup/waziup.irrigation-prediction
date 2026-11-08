import { apiClient } from '../../../api/client';
import { ApiError } from '../../../api/errors';
import { parseFarm, parsePlot } from '../../../api/adapters';
import {
  array,
  id,
  integer,
  invalid,
  number,
  object,
  string,
} from '../../../api/validation';
import { geometryJson, parseGeometry } from '../validation/farmSetupValidation';
import type {
  CreationAcknowledgement,
  Installation,
  MappedFarm,
  MappedPlot,
  SetupDraft,
} from '../types';
import type { RequestOptions } from '../../../domain/api';

function version(value: unknown): string {
  const token = string(value);
  if (!/^[a-f0-9]{64}$/.test(token))
    invalid('geometry_version', 'a content version');
  return token;
}
export function parseMappedFarm(value: unknown): MappedFarm {
  const d = object(value);
  const base = parseFarm(value);
  const areaM2 = number(d.area_m2);
  if (
    areaM2 <= 0 ||
    Math.abs(base.sizeM2 - areaM2) > Math.max(0.01, areaM2 * 1e-9)
  )
    invalid('farm.area_m2', 'consistent positive area');
  return {
    ...base,
    areaM2,
    geometry: parseGeometry(d.geometry),
    geometryVersion: version(d.geometry_version),
  };
}
export function parseMappedPlot(value: unknown): MappedPlot {
  const d = object(value),
    base = parsePlot(value);
  if (base.areaM2 <= 0 || base.areaM2 !== number(d.area_m2))
    invalid('plot.area_m2', 'consistent positive area');
  return {
    ...base,
    cropType: string(d.crop_type ?? ''),
    geometry: parseGeometry(d.geometry),
    geometryVersion: version(d.geometry_version),
  };
}
export function parseInstallation(value: unknown): Installation {
  try {
    const d = object(value);
    if (d.version !== 3) invalid('version', 'spatial registry version 3');
    const revision = integer(d.registry_revision),
      plots = array(d.plots, parseMappedPlot);
    if (d.farm === null) {
      if (plots.length || d.current_plot_id !== null || revision !== 0)
        invalid('setup', 'an empty installation');
      return {
        state: 'empty',
        revision,
        farm: null,
        plots: [],
        currentPlotId: null,
      };
    }
    const farm = parseMappedFarm(d.farm),
      currentPlotId = id(d.current_plot_id);
    if (
      !plots.length ||
      new Set(plots.map((p) => p.id)).size !== plots.length ||
      !plots.some((p) => p.id === currentPlotId)
    )
      invalid('plots', 'unique plots containing the selection');
    return { state: 'complete', farm, plots, currentPlotId, revision };
  } catch (cause) {
    if (cause instanceof ApiError) throw cause;
    throw new ApiError('The spatial registry could not be verified.', {
      kind: 'contract',
      cause,
    });
  }
}
function parseAcknowledgement(value: unknown): CreationAcknowledgement {
  const d = object(value);
  if (d.status !== 'created') invalid('status', 'created');
  const plot = parseMappedPlot(d.plot),
    farm = parseMappedFarm(d.farm);
  const currentPlotId = id(d.current_plot_id),
    revision = integer(d.registry_revision);
  if (currentPlotId !== plot.id || revision < 1)
    invalid('creation', 'confirmed plot identity and revision');
  return { farm, plot, currentPlotId, revision };
}
function creationBody(
  draft: SetupDraft,
  revision: number,
  mode: 'farm' | 'plot',
) {
  if (!draft.plotGeometry || (mode === 'farm' && !draft.farmGeometry))
    throw new ApiError('Complete the boundaries before saving.', {
      kind: 'validation',
    });
  const plot = {
    name: draft.plotName.trim(),
    crop_type: draft.cropType,
    geometry: geometryJson(draft.plotGeometry),
    area_unit: draft.areaUnit,
  };
  return mode === 'farm'
    ? {
        name: draft.name.trim(),
        owner: draft.owner.trim(),
        timezone: draft.timezone,
        area_unit: draft.areaUnit,
        geometry: geometryJson(draft.farmGeometry!),
        first_plot: plot,
        registry_revision: revision,
      }
    : { ...plot, registry_revision: revision };
}
export const farmSetupApi = {
  load: (options?: RequestOptions) =>
    apiClient.json('getFarmRegistry', parseInstallation, options),
  validate(draft: SetupDraft, revision: number, mode: 'farm' | 'plot') {
    return apiClient.json(
      'validateSetupBoundary',
      (value) => {
        if (object(value).valid !== true) invalid('valid', 'a valid boundary');
      },
      { method: 'POST', jsonBody: creationBody(draft, revision, mode) },
    );
  },
  create(draft: SetupDraft, revision: number, mode: 'farm' | 'plot') {
    return apiClient.json(
      mode === 'farm' ? 'farms' : 'addPlot',
      parseAcknowledgement,
      {
        method: 'POST',
        jsonBody: creationBody(draft, revision, mode),
      },
    );
  },
};
