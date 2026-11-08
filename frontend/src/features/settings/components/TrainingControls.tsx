import type {
  TrainingReadiness,
  TrainingState,
} from '../../../domain/training';
export function TrainingControls({
  plotName,
  readiness,
  dirty,
  saving,
  loading,
  error,
  state,
  onRetry,
  onStart,
  startPending,
  startError,
}: {
  plotName: string;
  readiness: TrainingReadiness | null;
  dirty: boolean;
  saving: boolean;
  loading: boolean;
  error: Error | null;
  state: TrainingState | undefined;
  onRetry: () => void;
  onStart: () => void;
  startPending: boolean;
  startError: Error | null;
}) {
  const finished = state?.status === 'finished' || state?.isTrainingFinished;
  const running = state?.currentlyTraining || state?.status === 'running';
  const unavailable =
    !readiness?.ready ||
    dirty ||
    saving ||
    loading ||
    !!error ||
    !state ||
    running ||
    finished ||
    startPending;
  return (
    <section
      className="flex min-w-0 flex-col gap-5 rounded-2xl border border-[#e0e0e0] bg-white p-6 sm:flex-row sm:items-center sm:justify-between [&_p]:text-xs"
      aria-labelledby="settings-training-heading"
    >
      <div className="min-w-0">
        <span className="mb-2 block text-[11px] font-extrabold uppercase tracking-widest text-[#367442]">
          Model training
        </span>
        <h2
          className="mb-3 text-base font-semibold"
          id="settings-training-heading"
        >
          Train {plotName}
        </h2>
        <div role="status" aria-live="polite">
          {dirty ? (
            <p className="my-3 leading-relaxed text-gray-600">
              Unsaved changes. Save all plot settings before training.
            </p>
          ) : !readiness ? null : !readiness.ready ? (
            <p className="my-3 leading-relaxed text-gray-600">
              Setup incomplete:{' '}
              {readiness.missing.join(', ') || 'check saved settings'}.
            </p>
          ) : null}
          {loading && (
            <p className="my-3 leading-relaxed text-gray-600">
              Checking training status…
            </p>
          )}
          {state && state.status !== 'idle' && state.status !== 'finished' && (
            <p className="my-3 leading-relaxed text-gray-600">
              Training status: {running ? 'running' : state.status}.
            </p>
          )}
          {state?.status === 'finished' && (
            <p className="my-3 leading-relaxed text-gray-600">
              {state.recommendationAvailable
                ? 'Training finished; a recommendation is available.'
                : 'Training finished; recommendation is currently unavailable.'}
            </p>
          )}
        </div>
        {state?.error && (
          <p className="my-3 leading-relaxed text-gray-600" role="alert">
            {state.error}
          </p>
        )}
        {state?.recommendationError && (
          <p className="my-3 leading-relaxed text-gray-600" role="alert">
            {state.recommendationError}
          </p>
        )}
        {error && (
          <p className="my-3 leading-relaxed text-gray-600" role="alert">
            {error.message}
          </p>
        )}
        {startError && (
          <p className="my-3 leading-relaxed text-gray-600" role="alert">
            Training could not be confirmed: {startError.message}.
          </p>
        )}
      </div>
      <div className="flex shrink-0 flex-wrap gap-3">
        {!finished && (
          <button
            className="min-h-11 cursor-pointer rounded-md border border-[#365e46] px-4 py-2.5 disabled:cursor-default disabled:opacity-55 bg-[#f2f7f0] text-[#153e27] hover:bg-[#e8f5e6]"
            type="button"
            disabled={!!unavailable}
            onClick={onStart}
          >
            {startPending
              ? 'Starting training…'
              : running
                ? 'Training…'
                : state?.status === 'error'
                  ? 'Train again'
                  : `Train ${plotName}`}
          </button>
        )}
        {(error || startError) && (
          <button
            className="min-h-11 cursor-pointer rounded-md border border-[#365e46] px-4 py-2.5 disabled:cursor-default disabled:opacity-55 bg-[#f2f7f0] text-[#153e27] hover:bg-[#e8f5e6]"
            type="button"
            disabled={loading || startPending}
            onClick={onRetry}
          >
            Refresh training status
          </button>
        )}
      </div>
    </section>
  );
}
