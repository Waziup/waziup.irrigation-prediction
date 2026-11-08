import { initialDraft } from './types';
import type { SetupDraft } from './types';
import { parseGeometry } from './validation/farmSetupValidation';

export function draftKey(mode: string) {
  return `wazifarm.setup.v1:${typeof location === 'undefined' ? '' : location.pathname}:${mode}`;
}
export function readDraft(mode: string): SetupDraft | null {
  try {
    const raw: unknown = JSON.parse(
      localStorage.getItem(draftKey(mode)) ?? 'null',
    );
    if (!raw || typeof raw !== 'object') return null;
    const d = raw as Record<string, unknown>;
    for (const key of [
      'name',
      'owner',
      'timezone',
      'latitude',
      'longitude',
      'plotName',
      'cropType',
    ])
      if (typeof d[key] !== 'string') return null;
    if (!['m2', 'ha', 'acre'].includes(String(d.areaUnit))) return null;
    return {
      ...initialDraft,
      name: String(d.name),
      owner: String(d.owner),
      timezone: String(d.timezone),
      latitude: String(d.latitude),
      longitude: String(d.longitude),
      plotName: String(d.plotName),
      cropType: String(d.cropType),
      areaUnit: d.areaUnit as SetupDraft['areaUnit'],
      farmGeometry:
        d.farmGeometry === null ? null : parseGeometry(d.farmGeometry),
      plotGeometry:
        d.plotGeometry === null ? null : parseGeometry(d.plotGeometry),
    };
  } catch {
    return null;
  }
}
export function writeDraft(mode: string, draft: SetupDraft): boolean {
  try {
    localStorage.setItem(draftKey(mode), JSON.stringify(draft));
    return true;
  } catch {
    return false;
  }
}
export function removeDraft(mode: string) {
  try {
    localStorage.removeItem(draftKey(mode));
  } catch {
    /* Browser storage may be unavailable. */
  }
}
