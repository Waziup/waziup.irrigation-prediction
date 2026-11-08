import { BoundarySettingsEntry } from './components/BoundarySettingsEntry';
import { PlotRemoval } from './components/PlotRemoval';
import { FarmManagement } from './components/FarmManagement';
import { SettingsWorkspace } from './components/SettingsWorkspace';
import { EmptyPlotSettings } from './components/EmptyPlotSettings';
import { PlotSettingsSections } from './components/PlotSettingsSections';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useBeforeUnload, useBlocker } from 'react-router';
import { ApiError } from '../../api/errors';
import { sensorApi } from '../../api/sensorApi';
import type { FieldErrors } from '../../domain/api';
import { useFarmContext } from '../farm-context/FarmContext';
import { initialSettings, toDraft } from './types';
import type { SectionProps, SettingsDraft, SettingsSnapshot } from './types';
import { validateSettingsDraft } from './validation/settingsValidation';
import { SettingsSaveError } from './settingsActions';
import { usePlotSettings } from './hooks/usePlotSettings';
import { useSavePlotSettings } from './hooks/useSavePlotSettings';
import { usePlotTraining } from './hooks/usePlotTraining';
import { TrainingControls } from './components/TrainingControls';

export function SettingsPage() {
  return (
    <>
      <BoundarySettingsEntry />
      <PlotSettingsPage />
    </>
  );
}
function PlotSettingsPage() {
  const context = useFarmContext();
  const [previewActive, setPreviewActive] = useState(false);
  if (previewActive || !context.currentPlot || context.status !== 'ready')
    return (
      <EmptyPlotSettings
        editable={previewActive || context.status === 'error'}
        onEditingChange={setPreviewActive}
      />
    );
  return (
    <PlotSettingsLoader
      key={context.currentPlot.id}
      plotId={context.currentPlot.id}
    />
  );
}
function PlotSettingsLoader({ plotId }: { plotId: string }) {
  const query = usePlotSettings(plotId);
  if (query.isPending || query.isFetching)
    return <EmptyPlotSettings editable={false} />;
  if (query.isError)
    return (
      <div className="grid gap-5">
        <FarmManagement />
        <section
          aria-label="Plot settings unavailable"
          className="rounded-xl border border-[#e0e6d9] bg-white p-5"
        >
          <h2 className="text-sm font-semibold text-[#304b3b]">
            Plot settings couldn’t be loaded
          </h2>
          <p className="mt-2 text-sm text-[#7b8873]">
            Try again to edit this plot’s configuration. Boundary management is
            still available above.
          </p>
          <button
            className="mt-4 min-h-10 rounded-lg border border-[#d5dfcc] px-4 py-2 text-sm font-medium text-[#345c3c] hover:bg-[#f4f7ef]"
            type="button"
            onClick={() => {
              void query.refetch();
            }}
          >
            Retry settings
          </button>
        </section>
      </div>
    );
  return (
    <SettingsEditor
      plotId={plotId}
      snapshot={query.data}
      reload={() => void query.refetch()}
    />
  );
}
function SettingsEditor({
  plotId,
  snapshot,
  reload,
}: {
  plotId: string;
  snapshot: SettingsSnapshot;
  reload: () => void;
}) {
  const context = useFarmContext();
  const holdContext = context.holdContext;
  const [managing, setManaging] = useState(false);
  const holdRelease = useRef<(() => void) | undefined>(undefined);
  useEffect(() => {
    if (managing) return;
    const release = holdContext?.();
    holdRelease.current = release;
    return () => {
      release?.();
      holdRelease.current = undefined;
    };
  }, [holdContext, managing]);

  const plot = context.currentPlot!;
  const [baseline, setBaseline] = useState(snapshot);
  const [savedDraft, setSavedDraft] = useState(() =>
    toDraft(
      snapshot.configured
        ? snapshot.settings
        : initialSettings(plot, context.farm),
    ),
  );
  const [draft, setDraft] = useState(savedDraft);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [message, setMessage] = useState('');
  const saveLock = useRef(false);
  const summary = useRef<HTMLDivElement>(null);
  const save = useSavePlotSettings(plotId);
  const training = usePlotTraining(plotId);
  const devices = useQuery({
    queryKey: ['plot', plotId, 'settings-devices'],
    queryFn: ({ signal }) => sensorApi.getDevices({ signal }, plotId),
    retry: false,
    staleTime: 60_000,
    refetchInterval: 60_000,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  });
  const dirty = JSON.stringify(draft) !== JSON.stringify(savedDraft);
  const running =
    training.status.data?.currentlyTraining ||
    training.status.data?.status === 'running';
  const pending = save.isPending || training.start.isPending;
  const requiresReload =
    save.error instanceof SettingsSaveError && save.error.requiresReload;
  const trainingUnknown =
    training.status.isPending ||
    training.status.isError ||
    !training.status.data;
  const blocker = useBlocker(dirty || pending);
  useEffect(() => {
    if (blocker.state !== 'blocked' || pending) return;
    if (!dirty || window.confirm('Discard unsaved changes?')) blocker.proceed();
    else blocker.reset();
  }, [blocker, dirty, pending]);
  useBeforeUnload(
    useCallback(
      (event: BeforeUnloadEvent) => {
        if (dirty || pending) {
          event.preventDefault();
          event.returnValue = '';
        }
      },
      [dirty, pending],
    ),
  );
  useEffect(() => {
    if (
      Object.keys(errors).some(
        (field) => !['actuatorIds', 'actuator'].includes(field),
      ) ||
      save.error
    )
      summary.current?.focus();
  }, [errors, save.error]);
  const props: SectionProps = {
    draft,
    errors,
    update: (key, value) => {
      setDraft((d) => ({ ...d, [key]: value }));
      setErrors({});
      setMessage('');
    },
  };
  function changed(keys: readonly (keyof SettingsDraft)[]) {
    return keys.some(
      (k) => JSON.stringify(draft[k]) !== JSON.stringify(savedDraft[k]),
    );
  }
  async function submit() {
    if (
      saveLock.current ||
      pending ||
      running ||
      trainingUnknown ||
      requiresReload
    )
      return;
    const validated = validateSettingsDraft(draft);
    setErrors(validated.errors);
    setMessage('');
    if (!validated.input) {
      return;
    }
    saveLock.current = true;
    try {
      const result = await save.mutateAsync({
        baseline,
        input: validated.input,
      });
      const confirmed = toDraft(result.saved.settings);
      setBaseline(result.saved);
      setSavedDraft(confirmed);
      setDraft(confirmed);
      setMessage(result.result.message);
      setErrors({});
    } catch (error) {
      const original =
        error instanceof SettingsSaveError ? error.original : error;
      if (original instanceof ApiError) setErrors(original.fieldErrors);
    } finally {
      saveLock.current = false;
    }
  }
  function discardAndReload() {
    if (
      !dirty ||
      window.confirm('Discard unsaved changes and reload saved settings?')
    )
      reload();
  }
  return (
    <div className="grid gap-5">
      <FarmManagement
        disabled={dirty || pending || !!running || managing || requiresReload}
        onBegin={() => {
          holdRelease.current?.();
          setManaging(true);
        }}
        onEnd={() => setManaging(false)}
      />
      <SettingsWorkspace
        name={plot.name}
        farmName={context.farm?.name}
        dirty={dirty}
      >
        <form
          className="grid gap-5"
          noValidate
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <div
            ref={summary}
            tabIndex={-1}
            className="empty:hidden focus:outline-2 focus:outline-[#365e46]"
            aria-live="polite"
          >
            {Object.keys(errors).some(
              (field) => !['actuatorIds', 'actuator'].includes(field),
            ) && (
              <div role="alert">
                <h2 className="mt-4 mb-2.5 text-xl font-semibold">
                  Review these settings
                </h2>
                <ul className="list-disc pl-5">
                  {Object.entries(errors)
                    .filter(
                      ([field]) => !['actuatorIds', 'actuator'].includes(field),
                    )
                    .map(([field, error]) => (
                      <li key={field}>
                        {field}: {error}
                      </li>
                    ))}
                </ul>
              </div>
            )}
            {save.error && (
              <p className="my-3 leading-relaxed text-gray-600" role="alert">
                {save.error.message}
              </p>
            )}
            {message && (
              <p className="my-3 leading-relaxed text-gray-600" role="status">
                {message}
              </p>
            )}
          </div>
          {running && (
            <p className="my-3 leading-relaxed text-gray-600" role="status">
              Training is running. Settings are read-only until it finishes.
            </p>
          )}
          <fieldset className="m-0 grid min-w-0 gap-5 border-0 p-0">
            <legend className="sr-only">Plot settings</legend>
            <PlotSettingsSections
              actions={
                <>
                  {' '}
                  <div className="flex flex-wrap gap-3">
                    <button
                      className="min-h-11 cursor-pointer rounded-md border border-[#365e46] px-4 py-2.5 disabled:cursor-default disabled:opacity-55 bg-[#24583c] text-white hover:bg-[#153e27]"
                      type="submit"
                      disabled={
                        pending ||
                        !!running ||
                        trainingUnknown ||
                        requiresReload ||
                        (baseline.configured && !dirty)
                      }
                    >
                      {save.isPending
                        ? 'Saving and verifying…'
                        : 'Save all plot settings'}
                    </button>
                    <Link
                      to="/dashboard"
                      className="inline-flex min-h-11 items-center rounded-lg border border-[#d8dee8] px-4 py-2 text-sm text-[#374151]"
                    >
                      Cancel
                    </Link>
                    {requiresReload && (
                      <button
                        className="min-h-11 cursor-pointer rounded-md border border-[#365e46] px-4 py-2.5 disabled:cursor-default disabled:opacity-55 bg-[#f2f7f0] text-[#153e27] hover:bg-[#e8f5e6]"
                        type="button"
                        disabled={pending}
                        onClick={discardAndReload}
                      >
                        Reload saved settings
                      </button>
                    )}
                  </div>
                </>
              }
              disabled={pending || !!running || managing}
              props={props}
              changed={changed}
              devices={devices.data ?? []}
              discoveryError={devices.error?.message ?? null}
              discovering={devices.isPending}
              onRetry={() => void devices.refetch()}
            />
          </fieldset>
        </form>
      </SettingsWorkspace>
      <TrainingControls
        plotName={baseline.configured ? baseline.settings.name : plot.name}
        readiness={
          baseline.configured ? baseline.settings.trainingReadiness : null
        }
        dirty={dirty || requiresReload}
        saving={save.isPending}
        loading={training.status.isFetching}
        error={training.status.error}
        state={training.status.data}
        onRetry={() => void training.status.refetch()}
        onStart={() => {
          if (
            !dirty &&
            baseline.configured &&
            baseline.settings.trainingReadiness.ready &&
            !pending &&
            !requiresReload
          )
            void training.startTraining(baseline).catch(() => {
              /* Rendered from mutation state. */
            });
        }}
        startPending={training.start.isPending}
        startError={training.start.error}
      />
      <PlotRemoval
        disabled={
          dirty ||
          pending ||
          !!running ||
          managing ||
          trainingUnknown ||
          requiresReload
        }
        onBegin={() => {
          holdRelease.current?.();
          setManaging(true);
        }}
        onEnd={() => setManaging(false)}
      />
    </div>
  );
}
