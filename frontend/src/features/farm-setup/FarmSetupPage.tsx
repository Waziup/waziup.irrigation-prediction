import { Brand } from '../../components/Brand';
import { Icon } from '../../components/Icon';
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useBeforeUnload, useBlocker, useNavigate } from 'react-router';
import { ApiError } from '../../api/errors';
import { farmSetupApi } from './api/farmSetupApi';
import { readDraft, removeDraft, writeDraft } from './draftStorage';
import { useFarmSetup } from './hooks/useFarmSetup';
import { initialDraft, setupKey } from './types';
import type { Installation, SetupDraft } from './types';
import {
  areaLabel,
  centerPosition,
  previewArea,
  validateDetails,
  validatePlotBoundary,
} from './validation/farmSetupValidation';
import {
  FarmDetailsForm,
  PlotNameForm,
  SetupField,
  SetupReview,
} from './components/SetupForms';
import { SetupSaveError } from './setupActions';
import type { GeoJsonPosition } from '../../domain/geometry';

const BoundaryEditor = lazy(() => import('./components/BoundaryEditor'));
const primary =
  'min-h-11 rounded-lg bg-[#356b30] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#275323] disabled:opacity-45';
const secondary =
  'min-h-11 rounded-lg border border-[#cad6c6] bg-white px-5 py-2.5 text-sm font-semibold text-[#2d5a27] hover:bg-[#edf5e9] disabled:opacity-45';
const allSteps = ['Farm boundary', 'Plot boundary', 'Review'] as const;

export function FarmSetupPage({ mode = 'farm' }: { mode?: 'farm' | 'plot' }) {
  const registry = useQuery({
    queryKey: setupKey,
    queryFn: ({ signal }) => farmSetupApi.load({ signal }),
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 30_000,
  });
  return (
    <div className="min-h-dvh bg-[#f7f8f3] px-4 py-5 text-[#243b29] sm:px-8 sm:py-7">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8 flex flex-wrap items-center justify-between gap-3 border-b border-[#e1e7dc] pb-5">
          <Link
            to="/dashboard"
            className="text-xl font-semibold tracking-tight text-[#294b32] no-underline"
          >
            <Brand />
          </Link>
          <div className="flex items-center gap-4">
            <span className="text-sm text-[#53645a]">
              {mode === 'farm' ? 'Create your farm' : 'Add a plot'}
            </span>
            {mode === 'plot' && (
              <Link
                to="/settings"
                className="inline-flex min-h-10 items-center rounded-lg border border-[#d5dfcc] bg-white px-4 text-sm font-medium text-[#345c3d] no-underline hover:bg-[#f2f6ed]"
              >
                Exit
              </Link>
            )}
          </div>
        </header>
        {registry.isPending ? (
          <p role="status">Loading farm setup…</p>
        ) : registry.isError ? (
          <section
            className="mx-auto mt-12 max-w-xl rounded-2xl border border-[#dfe6d8] bg-white p-6 shadow-sm sm:mt-20 sm:p-8"
            aria-labelledby="setup-error-title"
          >
            <span className="mb-5 inline-flex size-10 items-center justify-center rounded-xl bg-[#f4f1e7] text-[#927446]">
              <Icon name="alert" />
            </span>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-[#7d8c73]">
              Farm setup
            </p>
            <h1
              id="setup-error-title"
              className="text-xl font-semibold tracking-tight text-[#294531]"
            >
              We couldn’t load farm setup
            </h1>
            {!(
              registry.error instanceof ApiError &&
              registry.error.kind === 'contract'
            ) && (
              <p
                role="alert"
                className="mt-3 mb-6 text-sm leading-relaxed text-[#687862]"
              >
                We couldn’t connect to the farm service. Check that WaziFarm is
                running, then try again.
              </p>
            )}
            <button
              type="button"
              className={`${primary} mt-5`}
              onClick={() => void registry.refetch()}
            >
              Try again
            </button>
          </section>
        ) : mode === 'farm' && registry.data.state === 'complete' ? (
          <section className="rounded-2xl bg-white p-6">
            <h1 className="text-2xl font-bold">
              {registry.data.farm.name} is ready
            </h1>
            <p className="my-4">Your farm and its boundaries are saved.</p>
            <div className="flex flex-wrap gap-3">
              <Link className={primary} to="/dashboard">
                Open farm overview
              </Link>
              <Link className={secondary} to="/setup/plot">
                Add another plot
              </Link>
            </div>
          </section>
        ) : mode === 'plot' && registry.data.state === 'empty' ? (
          <section className="rounded-2xl bg-white p-6">
            <h1 className="text-2xl font-bold">Create your farm first</h1>
            <p className="my-4">Draw the farm and its first plot together.</p>
            <Link className={primary} to="/setup">
              Start farm setup
            </Link>
          </section>
        ) : (
          <SetupWizard key={mode} initial={registry.data} mode={mode} />
        )}
      </div>
    </div>
  );
}
function LeaveDialog({
  stay,
  leave,
  busy,
}: {
  stay: () => void;
  leave: () => void;
  busy: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      aria-labelledby="leave-title"
      onCancel={(e) => {
        e.preventDefault();
        stay();
      }}
      className="m-auto max-w-[calc(100%-2rem)] rounded-2xl bg-white p-6 text-[#243b29] shadow-xl backdrop:bg-black/35"
    >
      <h2 id="leave-title" className="text-xl font-bold">
        {busy ? 'Saving is still in progress' : 'Leave setup?'}
      </h2>
      <p className="my-4">
        {busy
          ? 'Wait for the save result before leaving.'
          : 'Your changes haven’t been saved.'}
      </p>
      <div className="flex gap-3">
        <button className={primary} type="button" onClick={stay}>
          Stay in setup
        </button>
        {!busy && (
          <button className={secondary} type="button" onClick={leave}>
            Leave setup
          </button>
        )}
      </div>
    </dialog>
  );
}
function SetupWizard({
  initial,
  mode,
}: {
  initial: Installation;
  mode: 'farm' | 'plot';
}) {
  const navigate = useNavigate(),
    client = useQueryClient();
  const [draft, setDraft] = useState<SetupDraft>(() => {
    const saved = readDraft(mode) ?? {
      ...initialDraft,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    };
    return mode === 'plot' && initial.state === 'complete'
      ? {
          ...saved,
          farmGeometry: initial.farm.geometry,
          name: initial.farm.name,
          timezone: initial.farm.timezone,
          cropType: '',
        }
      : { ...saved, cropType: '' };
  });
  const steps = mode === 'farm' ? [...allSteps] : allSteps.slice(1);
  const [step, setStep] = useState(0),
    [drawing, setDrawing] = useState(false),
    [error, setError] = useState('');
  const [center, setCenter] = useState<GeoJsonPosition | null>(() => {
    if (mode === 'plot' && initial.state === 'complete')
      return initial.farm.geometry.coordinates[0][0];
    try {
      return centerPosition(draft.latitude, draft.longitude);
    } catch {
      return null;
    }
  });
  const stepHeading = useRef<HTMLHeadingElement>(null);
  const continueButton = useRef<HTMLButtonElement>(null);
  const [checking, setChecking] = useState(false);
  useEffect(() => {
    stepHeading.current?.scrollIntoView({ block: 'start' });
    stepHeading.current?.focus({ preventScroll: true });
  }, [step]);
  function boundaryFinished() {
    requestAnimationFrame(() =>
      continueButton.current?.scrollIntoView({
        block: 'end',
        behavior: 'smooth',
      }),
    );
  }
  const [completed, setCompleted] = useState<Installation | null>(null);
  const [storageUnavailable, setStorageUnavailable] = useState(false);
  const confirmed = useRef(false);
  const save = useFarmSetup(initial, mode);
  const dirty =
    drawing ||
    !!(
      draft.plotName ||
      draft.cropType ||
      draft.plotGeometry ||
      (mode === 'farm' &&
        (draft.name ||
          draft.owner ||
          draft.farmGeometry ||
          draft.latitude ||
          draft.longitude ||
          draft.timezone !== initialDraft.timezone ||
          draft.areaUnit !== initialDraft.areaUnit))
    );
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      !confirmed.current &&
      (dirty || save.busy) &&
      currentLocation.pathname !== nextLocation.pathname,
  );
  useBeforeUnload(
    useCallback(
      (e) => {
        if (!confirmed.current && (dirty || save.busy)) {
          e.preventDefault();
          e.returnValue = '';
        }
      },
      [dirty, save.busy],
    ),
  );
  const baselineFarm =
    save.baseline.state === 'complete'
      ? save.baseline.farm.geometry
      : draft.farmGeometry;
  const others =
    mode === 'plot' ? save.baseline.plots.map((p) => p.geometry) : [];
  const label = steps[step];
  function update(fields: Partial<SetupDraft>) {
    const next = { ...draft, ...fields };
    setDraft(next);
    setStorageUnavailable(!writeDraft(mode, next));
    setError('');
    save.clearError();
  }
  function validateStep() {
    if (label === 'Farm boundary') validateDetails(draft);
    if (label === 'Farm boundary' && !draft.farmGeometry)
      throw new Error('Finish the farm boundary before continuing.');
    if (label === 'Plot boundary' || label === 'Review') {
      if (!draft.plotGeometry || !baselineFarm)
        throw new Error('Finish both boundaries before continuing.');
      validatePlotBoundary(draft.plotGeometry, baselineFarm, others);
    }
    if (label === 'Plot boundary' || label === 'Review') {
      if (!draft.plotName.trim() || draft.plotName.trim().length > 160)
        throw new Error('Enter a plot name up to 160 characters.');
    }
    if (label === 'Review' && mode === 'farm') validateDetails(draft);
  }
  async function finish(saved: Installation, destination = '/dashboard') {
    confirmed.current = true;
    removeDraft(mode);
    await client.cancelQueries({ queryKey: ['farm-context'] });
    client.removeQueries({
      predicate: (query) => query.queryKey[0] !== setupKey[0],
    });
    client.setQueryData(setupKey, saved);
    navigate(destination);
  }
  async function submit() {
    try {
      validateStep();
      const saved = await save.save(draft);
      if (saved) {
        confirmed.current = true;
        setCompleted(saved);
        removeDraft(mode);
      }
    } catch (cause) {
      const original = cause instanceof SetupSaveError ? cause.original : cause;
      if (
        original instanceof ApiError &&
        original.status === 422 &&
        Object.keys(original.fieldErrors).some((key) =>
          key.includes('geometry'),
        )
      ) {
        setStep(mode === 'farm' ? 1 : 0);
        save.clearError();
        setError(
          'Check the plot boundary before continuing. Shared farm edges are allowed.',
        );
      } else {
        setError(cause instanceof Error ? cause.message : 'Check your setup.');
      }
    }
  }
  const originalError =
    save.error instanceof SetupSaveError ? save.error.original : save.error;
  if (completed)
    return (
      <section className="rounded-2xl border border-[#dce5d7] bg-white p-6 sm:p-8">
        <p className="text-sm font-semibold text-[#548f3c]">
          {mode === 'farm' ? 'Farm created' : 'Plot added'}
        </p>
        <h1 className="mt-2 text-2xl font-bold">Next, configure your plot</h1>
        <p className="my-4 text-[#53645a]">
          {completed.state === 'complete' ? completed.farm.name : draft.name}
        </p>
        <p className="mb-6 text-[#53645a]">
          Set up sensors, soil and irrigation in Plot settings.
        </p>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            className={primary}
            onClick={() => void finish(completed, '/settings')}
          >
            Configure plot
          </button>
          <button
            type="button"
            className="rounded-lg border border-[#ccd9c6] px-5 py-3 font-semibold"
            onClick={() => void finish(completed)}
          >
            Open farm overview
          </button>
        </div>
      </section>
    );
  return (
    <>
      <div className="mb-6">
        <p className="text-sm text-[#53645a]">
          Step {step + 1} of {steps.length}
        </p>
        <h1
          ref={stepHeading}
          tabIndex={-1}
          className="mt-2 scroll-mt-5 text-3xl font-bold outline-none"
        >
          {label}
        </h1>
        <ol aria-label="Setup progress" className="mt-5 flex flex-wrap gap-2">
          {steps.map((name, i) => (
            <li
              key={name}
              aria-current={i === step ? 'step' : undefined}
              className={`rounded-full px-3 py-1.5 text-xs ${i === step ? 'bg-[#356b30] text-white' : 'bg-[#e3ebde] text-[#53645a]'}`}
            >
              {i + 1}. {name}
            </li>
          ))}
        </ol>
      </div>
      <section className="rounded-2xl border border-[#dce5d7] bg-white p-5 shadow-sm sm:p-8">
        {storageUnavailable && (
          <p
            role="alert"
            className="mb-5 rounded-lg bg-amber-50 p-3 text-sm text-amber-950"
          >
            Your browser cannot keep this draft after reload. Keep this page
            open until you finish.
          </p>
        )}
        <fieldset
          disabled={save.busy || checking || !!save.reconciled}
          className="grid min-w-0 gap-6"
        >
          {label === 'Farm boundary' && (
            <>
              <FarmDetailsForm draft={draft} update={update} />
              <div className="grid gap-4 sm:grid-cols-[1fr_1fr_auto]">
                <SetupField
                  label="Farm latitude"
                  type="number"
                  value={draft.latitude}
                  onChange={(latitude) => update({ latitude })}
                />
                <SetupField
                  label="Farm longitude"
                  type="number"
                  value={draft.longitude}
                  onChange={(longitude) => update({ longitude })}
                />
                <button
                  type="button"
                  disabled={drawing}
                  className={`${secondary} self-end`}
                  onClick={() => {
                    try {
                      setCenter(
                        centerPosition(draft.latitude, draft.longitude),
                      );
                      setError('');
                    } catch (cause) {
                      setError((cause as Error).message);
                    }
                  }}
                >
                  Locate farm
                </button>
              </div>
              {!center && (
                <p className="text-sm text-[#53645a]">
                  Enter a reference location to position the map.
                </p>
              )}
              {center && (
                <Suspense
                  fallback={<p role="status">Loading boundary editor…</p>}
                >
                  <BoundaryEditor
                    label="Farm boundary"
                    value={draft.farmGeometry}
                    center={center}
                    onFinished={boundaryFinished}
                    onDrawingChange={setDrawing}
                    onChange={(farmGeometry) => update({ farmGeometry })}
                  />
                </Suspense>
              )}
            </>
          )}
          {label === 'Plot boundary' && baselineFarm && (
            <>
              <PlotNameForm draft={draft} update={update} />
              <Suspense
                fallback={<p role="status">Loading boundary editor…</p>}
              >
                <BoundaryEditor
                  label="Plot boundary"
                  value={draft.plotGeometry}
                  center={baselineFarm.coordinates[0][0]}
                  farm={baselineFarm}
                  others={others}
                  onFinished={boundaryFinished}
                  onDrawingChange={setDrawing}
                  onChange={(plotGeometry) => update({ plotGeometry })}
                />
              </Suspense>
            </>
          )}
          {label === 'Review' && <SetupReview draft={draft} mode={mode} />}
        </fieldset>
        {(error || save.error || save.reloadError) && (
          <div
            role="alert"
            className="mt-5 rounded-xl bg-red-50 p-4 text-sm text-red-800"
          >
            <p>{error || save.reloadError || save.error?.message}</p>
            {originalError instanceof ApiError &&
              Object.entries(originalError.fieldErrors).map(([key, message]) =>
                message ===
                (error || save.reloadError || save.error?.message) ? null : (
                  <p key={key}>{message}</p>
                ),
              )}
          </div>
        )}
        {save.mustReload && !save.reconciled && (
          <div className="mt-5 rounded-xl bg-amber-50 p-4 text-sm text-amber-950">
            <p>
              The save outcome or farm layout needs checking. Your draft is
              retained.
            </p>
            <button
              type="button"
              className={`${secondary} mt-3`}
              disabled={save.busy}
              onClick={() => void save.reload(draft)}
            >
              Reload saved farm
            </button>
          </div>
        )}
        {save.reconciled ? (
          <div role="status" className="mt-5 rounded-xl bg-[#edf4e8] p-4">
            <p>
              A farm or matching plot is already saved. Review the farm before
              making another change.
            </p>
            <button
              type="button"
              className={`${primary} mt-3`}
              onClick={() => void finish(save.reconciled!)}
            >
              Review saved farm
            </button>
          </div>
        ) : (
          <div className="mt-8 flex flex-wrap justify-between gap-3 border-t border-[#e1e8dd] pt-5">
            <button
              type="button"
              className={secondary}
              disabled={step === 0 || save.busy || checking || drawing}
              onClick={() => {
                setStep(step - 1);
                setError('');
                save.clearError();
              }}
            >
              Back
            </button>
            {label === 'Review' ? (
              <button
                type="button"
                className={primary}
                disabled={save.busy || save.mustReload}
                onClick={() => void submit()}
              >
                {save.busy
                  ? 'Saving…'
                  : mode === 'farm'
                    ? 'Save farm and first plot'
                    : 'Save plot'}
              </button>
            ) : (
              <button
                type="button"
                className={primary}
                disabled={save.busy || checking || drawing}
                ref={continueButton}
                onClick={async () => {
                  if (checking) return;
                  setChecking(true);
                  try {
                    validateStep();
                    if (label === 'Plot boundary')
                      await farmSetupApi.validate(
                        draft,
                        save.baseline.revision,
                        mode,
                      );
                    save.clearError();
                    setStep(step + 1);
                    setError('');
                  } catch (cause) {
                    setError((cause as Error).message);
                  } finally {
                    setChecking(false);
                  }
                }}
              >
                {checking ? 'Checking boundary…' : 'Continue'}
              </button>
            )}
          </div>
        )}
        {label === 'Review' && draft.plotGeometry && (
          <p className="mt-4 text-xs text-[#53645a]">
            Plot preview: {areaLabel(previewArea(draft.plotGeometry))} · Nothing
            is saved to your farm until you confirm.
          </p>
        )}
      </section>
      {blocker.state === 'blocked' && (
        <LeaveDialog
          busy={save.busy}
          stay={() => blocker.reset()}
          leave={() => blocker.proceed()}
        />
      )}
    </>
  );
}

export function PlotSetupPage() {
  return <FarmSetupPage mode="plot" />;
}
