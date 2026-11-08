import { apiClient } from '../../api/client';
import { ApiError } from '../../api/errors';
import { object, member, nullable, id, integer } from '../../api/validation';
import type { PolygonGeometry } from '../../domain/geometry';
import type { FarmMap, SavedMapBoundary } from '../../domain/map';
import { geometryJson } from '../farm-setup/validation/farmSetupValidation';
import { farmMapApi, parseSavedBoundary } from './api/farmMapApi';
export type BoundaryTarget =
  { entity: 'farm' } | { entity: 'plot'; plotId: string };
export function savedBoundary(
  data: FarmMap,
  target: BoundaryTarget,
): SavedMapBoundary {
  const value =
    target.entity === 'farm'
      ? data.farm
      : data.plots.find((plot) => plot.plotId === target.plotId);
  if (!value)
    throw new Error('Boundary no longer exists. Reload the farm map.');
  return value;
}
export class BoundarySaveError extends Error {
  constructor(
    message: string,
    readonly requiresReload: boolean,
  ) {
    super(message);
  }
}
export async function saveBoundaryChange(
  data: FarmMap,
  target: BoundaryTarget,
  geometry: PolygonGeometry,
): Promise<FarmMap> {
  const before = savedBoundary(data, target);
  try {
    const result = await apiClient.json(
      'saveBoundary',
      (value) => {
        const d = object(value);
        return {
          status: member(d.status, ['saved']),
          entity: member(d.entity, ['farm', 'plot']),
          plotId: nullable(d.plot_id, id),
          revision: integer(d.registry_revision),
          boundary: parseSavedBoundary(d.boundary),
        };
      },
      {
        method: 'POST',
        jsonBody: {
          entity: target.entity,
          ...(target.entity === 'plot' ? { plot_id: target.plotId } : {}),
          registry_revision: data.registryRevision,
          geometry_version: before.geometryVersion,
          geometry: geometryJson(geometry),
        },
      },
    );
    if (
      result.entity !== target.entity ||
      result.plotId !== (target.entity === 'plot' ? target.plotId : null) ||
      result.revision !== data.registryRevision + 1 ||
      JSON.stringify(result.boundary.geometry) !== JSON.stringify(geometry)
    )
      throw new Error('The server did not confirm the requested boundary.');
    const reloaded = await farmMapApi.load();
    const confirmed = savedBoundary(reloaded, target);
    if (
      reloaded.registryRevision !== result.revision ||
      confirmed.geometryVersion !== result.boundary.geometryVersion ||
      JSON.stringify(confirmed.geometry) !== JSON.stringify(geometry)
    )
      throw new Error('The boundary changed during confirmation.');
    return reloaded;
  } catch (cause) {
    const rejected =
      cause instanceof ApiError &&
      (cause.status === 400 || cause.status === 422);
    throw new BoundarySaveError(
      cause instanceof Error ? cause.message : 'Boundary save failed.',
      !rejected,
    );
  }
}
