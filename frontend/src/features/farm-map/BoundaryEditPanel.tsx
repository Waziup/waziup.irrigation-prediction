import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { useBlocker } from 'react-router';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { FarmMap } from '../../domain/map';
import { useFarmContext } from '../farm-context/FarmContext';
import {
  areaLabel,
  previewArea,
  validatePlotBoundary,
} from '../farm-setup/validation/farmSetupValidation';
import {
  BoundarySaveError,
  saveBoundaryChange,
  savedBoundary,
  type BoundaryTarget,
} from './boundaryActions';
import { farmMapApi } from './api/farmMapApi';
import { farmMapKey } from './hooks/useFarmMap';
import { farmContextKey } from '../farm-context/selection';
import { BoundaryPreview } from '../farm-setup/components/BoundaryPreview';
import { mapButton } from './MapControls';
const BoundaryEditor = lazy(
  () => import('../farm-setup/components/BoundaryEditor'),
);

export function BoundaryEditPanel({
  data,
  target,
  onClose,
}: {
  data: FarmMap;
  target: BoundaryTarget;
  onClose: () => void;
}) {
  const [baseline, setBaseline] = useState(data);
  const original = savedBoundary(baseline, target);
  const [draft, setDraft] = useState(original.geometry);
  const [drawing, setDrawing] = useState(false),
    [review, setReview] = useState(false);
  const [confirmed, setConfirmed] = useState(false),
    [mustReload, setMustReload] = useState(false);
  const [message, setMessage] = useState(''),
    [discard, setDiscard] = useState(false);
  const [done, setDone] = useState(false),
    [busy, setBusy] = useState(false);
  const lock = useRef(false);
  const context = useFarmContext();
  const client = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => saveBoundaryChange(baseline, target, draft),
    retry: false,
  });
  const dirty =
    drawing || JSON.stringify(draft) !== JSON.stringify(original.geometry);
  const blocker = useBlocker(!done && (dirty || mustReload || busy));
  const hold = context.holdContext;
  useEffect(() => hold?.(), [hold]);
  useEffect(() => {
    if (!dirty && !mustReload && !busy) return;
    const prevent = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', prevent);
    return () => window.removeEventListener('beforeunload', prevent);
  }, [dirty, mustReload, busy]);
  const delta =
    Math.abs(previewArea(draft) - previewArea(original.geometry)) /
    previewArea(original.geometry);
  async function refreshAfterSave() {
    await client.invalidateQueries({ queryKey: farmMapKey });
    await client.invalidateQueries({
      queryKey: farmContextKey,
      refetchType: 'none',
    });
    setDone(true);
  }
  async function save() {
    if (
      lock.current ||
      mustReload ||
      drawing ||
      !review ||
      (delta >= 0.1 && !confirmed)
    )
      return;
    lock.current = true;
    setBusy(true);
    setMessage('');
    try {
      await mutation.mutateAsync();
      await refreshAfterSave();
      setMessage('Saved boundary verified on the server.');
    } catch (cause) {
      // A rejected acknowledgement may follow a committed write. Refresh both
      // geometry and derived context areas when the editor releases its hold.
      await client.invalidateQueries({
        queryKey: farmMapKey,
        refetchType: 'none',
      });
      await client.invalidateQueries({
        queryKey: farmContextKey,
        refetchType: 'none',
      });
      setMustReload(
        cause instanceof BoundarySaveError ? cause.requiresReload : true,
      );
      setMessage(
        cause instanceof Error
          ? cause.message
          : 'Reload the saved boundary before continuing.',
      );
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }
  async function reload() {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    try {
      const server = await farmMapApi.load();
      const saved = savedBoundary(server, target);
      if (JSON.stringify(saved.geometry) === JSON.stringify(draft)) {
        await refreshAfterSave();
        setMessage(
          'The server contains this boundary. Saved geometry verified.',
        );
      } else {
        setBaseline(server);
        setDraft(saved.geometry);
        setReview(false);
        setConfirmed(false);
        setMustReload(false);
        mutation.reset();
        setMessage(
          'Loaded the current saved boundary. The unsaved draft was discarded; review before editing again.',
        );
      }
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : 'Reload failed.');
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }
  if (!baseline.farm) return null;
  const others = baseline.plots
    .filter((plot) => target.entity === 'farm' || plot.plotId !== target.plotId)
    .map((plot) => plot.geometry);
  return (
    <section className="grid gap-4 rounded-xl border border-[#dce4ed] bg-white p-4 sm:p-6">
      <h2 className="text-xl font-bold text-[#174e25]">
        Edit{' '}
        {target.entity === 'farm'
          ? 'farm'
          : baseline.plots.find((p) => p.plotId === target.plotId)?.name}{' '}
        boundary
      </h2>
      <p className="text-sm text-[#52606d]">
        Move the saved corner points to adjust this boundary. Review the new
        outline before saving. Your farm and plots remain the same.
      </p>
      {message && (
        <p
          role={mustReload ? 'alert' : 'status'}
          className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900"
        >
          {message}
        </p>
      )}
      {!done && !mustReload && !review && (
        <Suspense fallback={<p role="status">Loading boundary editor…</p>}>
          <BoundaryEditor
            startEditing
            label="Edit boundary"
            value={draft}
            center={baseline.farm.geometry.coordinates[0][0]}
            farm={target.entity === 'plot' ? baseline.farm.geometry : null}
            others={others}
            onChange={(geometry) => {
              setDraft(geometry);
              setConfirmed(false);
            }}
            onDrawingChange={setDrawing}
          />
        </Suspense>
      )}
      <p className="text-sm">
        Saved area: {areaLabel(original.areaM2)} · Draft preview:{' '}
        {areaLabel(previewArea(draft))}
      </p>
      {review && !done && (
        <>
          <BoundaryPreview
            farm={target.entity === 'farm' ? draft : baseline.farm.geometry}
            plot={
              target.entity === 'farm' ? baseline.plots[0]!.geometry : draft
            }
          />
          <p className="font-semibold">
            Review the new boundary and area before saving.
          </p>
          {delta >= 0.1 && (
            <label className="flex gap-3 text-sm">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(e) => setConfirmed(e.target.checked)}
              />
              I understand that the plot or farm area changes by approximately{' '}
              {Math.round(delta * 100)}%.
            </label>
          )}
        </>
      )}
      <div className="flex flex-wrap gap-2">
        {done ? (
          <button type="button" className={mapButton} onClick={onClose}>
            Return to farm map
          </button>
        ) : (
          <>
            {mustReload ? (
              <button
                type="button"
                className={mapButton}
                disabled={busy}
                onClick={() => {
                  void reload();
                }}
              >
                Reload saved boundary
              </button>
            ) : review ? (
              <>
                <button
                  type="button"
                  className={mapButton}
                  disabled={busy || (delta >= 0.1 && !confirmed)}
                  onClick={() => {
                    void save();
                  }}
                >
                  {busy ? 'Saving boundary…' : 'Confirm and save boundary'}
                </button>
                <button
                  type="button"
                  className={mapButton}
                  disabled={busy}
                  onClick={() => setReview(false)}
                >
                  Back to editing
                </button>
              </>
            ) : (
              <button
                type="button"
                className={mapButton}
                disabled={!dirty || drawing || busy}
                onClick={() => {
                  try {
                    if (target.entity === 'farm')
                      baseline.plots.forEach((plot) =>
                        validatePlotBoundary(plot.geometry, draft),
                      );
                    else
                      validatePlotBoundary(
                        draft,
                        baseline.farm!.geometry,
                        others,
                      );
                    setReview(true);
                    setMessage('');
                  } catch (cause) {
                    setMessage(
                      cause instanceof Error
                        ? cause.message
                        : 'Check the boundary.',
                    );
                  }
                }}
              >
                Review boundary
              </button>
            )}
            <button
              type="button"
              className={mapButton}
              disabled={busy}
              onClick={() => {
                if (dirty || mustReload) setDiscard(true);
                else onClose();
              }}
            >
              Cancel editing
            </button>
          </>
        )}
      </div>
      {discard && (
        <div
          role="alertdialog"
          aria-label="Discard boundary draft"
          className="grid gap-3"
        >
          <p>
            Discard this draft? Saved server geometry is unchanged by cancel. An
            uncertain save must be checked on the server.
          </p>
          <button type="button" className={mapButton} onClick={onClose}>
            Discard draft
          </button>
          <button
            type="button"
            className={mapButton}
            onClick={() => setDiscard(false)}
          >
            Keep editing
          </button>
        </div>
      )}
      {blocker.state === 'blocked' && (
        <div role="alertdialog" aria-label="Leave boundary editor">
          <p>
            {busy
              ? 'A save is in progress. Wait for server confirmation.'
              : 'Leave and discard the local draft? A submitted save may already be committed.'}
          </p>
          <button
            type="button"
            className={mapButton}
            onClick={() => blocker.reset()}
          >
            Stay here
          </button>
          <button
            type="button"
            className={mapButton}
            disabled={busy}
            onClick={() => blocker.proceed()}
          >
            Leave editor
          </button>
        </div>
      )}
    </section>
  );
}
