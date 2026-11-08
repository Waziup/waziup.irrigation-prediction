import { ApiError } from '../../api/errors';
import { farmSetupApi } from './api/farmSetupApi';
import type {
  CreationAcknowledgement,
  Installation,
  SetupDraft,
} from './types';

export class SetupSaveError extends Error {
  constructor(
    readonly original: unknown,
    readonly requiresReload: boolean,
  ) {
    super(
      original instanceof Error
        ? original.message
        : 'Setup could not be confirmed.',
    );
    this.name = 'SetupSaveError';
  }
}
export function verifyCreation(
  ack: CreationAcknowledgement,
  saved: Installation,
) {
  const plot = saved.plots.find((p) => p.id === ack.plot.id);
  if (
    saved.state !== 'complete' ||
    saved.revision !== ack.revision ||
    saved.farm.geometryVersion !== ack.farm.geometryVersion ||
    !plot ||
    plot.geometryVersion !== ack.plot.geometryVersion ||
    plot.cropType !== ack.plot.cropType ||
    plot.name !== ack.plot.name ||
    saved.currentPlotId !== ack.currentPlotId
  )
    throw new Error(
      'The saved setup changed or could not be verified. Reload the saved farm before continuing.',
    );
}
export async function saveSetup(
  draft: SetupDraft,
  baseline: Installation,
  mode: 'farm' | 'plot',
  api: Pick<typeof farmSetupApi, 'create' | 'load'> = farmSetupApi,
) {
  let acknowledged = false;
  try {
    const ack = await api.create(draft, baseline.revision, mode);
    acknowledged = true;
    const same = (a: unknown, b: unknown) =>
      JSON.stringify(a) === JSON.stringify(b);
    if (
      ack.revision <= baseline.revision ||
      ack.plot.name !== draft.plotName.trim() ||
      ack.plot.cropType !== draft.cropType ||
      ack.plot.areaUnit !== draft.areaUnit ||
      !same(ack.plot.geometry, draft.plotGeometry) ||
      (mode === 'farm' &&
        (ack.farm.name !== draft.name.trim() ||
          ack.farm.timezone !== draft.timezone ||
          ack.farm.owner !== draft.owner.trim() ||
          ack.farm.areaUnit !== draft.areaUnit ||
          !same(ack.farm.geometry, draft.farmGeometry))) ||
      (mode === 'plot' &&
        (baseline.state !== 'complete' ||
          ack.farm.geometryVersion !== baseline.farm.geometryVersion))
    )
      throw new Error(
        'The creation response did not match your draft. Reload the saved farm before continuing.',
      );
    const saved = await api.load();
    verifyCreation(ack, saved);
    return saved;
  } catch (cause) {
    const rejected =
      !acknowledged &&
      cause instanceof ApiError &&
      cause.kind === 'http' &&
      cause.status !== null &&
      cause.status >= 400 &&
      cause.status < 500;
    throw new SetupSaveError(
      cause,
      !rejected ||
        (cause instanceof ApiError && [409, 428].includes(cause.status ?? 0)),
    );
  }
}
export function matchesSavedDraft(
  draft: SetupDraft,
  saved: Installation,
  mode: 'farm' | 'plot',
) {
  if (saved.state !== 'complete') return false;
  const equal = (a: unknown, b: unknown) =>
    JSON.stringify(a) === JSON.stringify(b);
  return (
    (mode === 'plot' ||
      (saved.farm.name === draft.name.trim() &&
        equal(saved.farm.geometry, draft.farmGeometry))) &&
    saved.plots.some(
      (p) =>
        p.name === draft.plotName.trim() &&
        p.cropType === draft.cropType &&
        equal(p.geometry, draft.plotGeometry),
    )
  );
}
