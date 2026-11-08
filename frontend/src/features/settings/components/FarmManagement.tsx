import {
  useMutation,
  useMutationState,
  useQueryClient,
} from '@tanstack/react-query';
import { useNavigate } from 'react-router';
import { ApiError } from '../../../api/errors';
import { useRef, useState } from 'react';
import { plotApi } from '../../../api/plotApi';
import { fromSquareMetres, toSquareMetres } from '../../../domain/plot';
import type { AreaUnit } from '../../../domain/plot';
import { useFarmContext } from '../../farm-context/FarmContext';
import { Field, SelectField } from './Fields';

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '';
}
const buttonClass =
  'inline-flex min-h-10 items-center justify-center rounded-lg bg-[#356b30] px-4 py-2 text-sm font-semibold text-white hover:bg-[#275323] disabled:cursor-default disabled:opacity-45';
export function FarmManagement({
  disabled = false,
  preview = false,
  onBegin,
  onEnd,
}: {
  disabled?: boolean;
  preview?: boolean;
  onBegin?: () => void;
  onEnd?: () => void;
}) {
  const context = useFarmContext();
  const navigate = useNavigate();
  const client = useQueryClient();
  const change = useMutation({
    mutationKey: ['farm-management'],
    mutationFn: (request: () => Promise<unknown>) => request(),
    retry: false,
  });
  const failures = useMutationState({
    filters: { mutationKey: ['farm-management'], status: 'error' },
    select: (mutation) => mutation.state.error,
  });
  const lastFailure = failures.at(-1);
  const reportedError = errorMessage(lastFailure);

  const dialog = useRef<HTMLDialogElement>(null);
  const [kind, setKind] = useState<'plot' | 'farm'>('plot');
  const [busy, setBusy] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [error, setError] = useState('');
  const [form, setForm] = useState({
    name: '',
    area: '',
    unit: 'm2' as AreaUnit,
    latitude: '',
    longitude: '',
    plotName: '',
  });
  const unavailable = disabled || busy || context.status !== 'ready';
  const canOpen = !busy && (preview || !unavailable);
  const hasFarm = context.farm !== null;
  function open(next: 'plot' | 'farm') {
    if (!canOpen || (next === 'plot' && !hasFarm)) return;
    if (next === 'plot' || (next === 'farm' && !hasFarm)) {
      navigate(next === 'plot' ? '/settings/plots/new' : '/settings/farm/new');
      return;
    }
    for (const mutation of client
      .getMutationCache()
      .findAll({ mutationKey: ['farm-management'], status: 'error' }))
      client.getMutationCache().remove(mutation);
    setDialogOpen(true);
    setKind(next);
    setError('');
    const farm = context.farm;
    setForm(
      next === 'farm' && farm
        ? {
            name: farm.name,
            area: String(fromSquareMetres(farm.sizeM2, farm.areaUnit)),
            unit: farm.areaUnit,
            latitude: farm.coordinates ? String(farm.coordinates.latitude) : '',
            longitude: farm.coordinates
              ? String(farm.coordinates.longitude)
              : '',
            plotName: '',
          }
        : {
            name: '',
            area: '',
            unit: 'm2',
            latitude: '',
            longitude: '',
            plotName: '',
          },
    );
    dialog.current?.showModal();
  }
  async function submit() {
    if (unavailable) return;
    // Farm details are view-only.
    if (kind !== 'plot') return;
    const area = Number(form.area);
    if (!form.name.trim() || !Number.isFinite(area) || area < 0) {
      setError('Enter a name and a valid area.');
      return;
    }
    for (const mutation of client
      .getMutationCache()
      .findAll({ mutationKey: ['farm-management'], status: 'error' }))
      client.getMutationCache().remove(mutation);
    setBusy(true);
    setError('');
    try {
      await change.mutateAsync(() =>
        plotApi.addPlot({
          name: form.name.trim(),
          areaM2: toSquareMetres(area, form.unit),
          areaUnit: form.unit,
        }),
      );
      onBegin?.();
      await context.refresh();
      dialog.current?.close();
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'The change could not be confirmed.',
      );
      // Rejected requests keep the dialog draft. Ambiguous failures require a reread.
      if (
        cause instanceof ApiError &&
        cause.kind === 'http' &&
        cause.status &&
        cause.status >= 400 &&
        cause.status < 500
      )
        return;
      onBegin?.();
      await context.refresh().catch(() => undefined);
    } finally {
      setBusy(false);
    }
  }
  async function edit(plotId: string) {
    if (unavailable) return;
    if (plotId === context.currentPlot?.id) {
      document
        .getElementById('plot-settings-workspace')
        ?.scrollIntoView({ block: 'start' });
      return;
    }
    onBegin?.();
    try {
      await context.selectPlot(plotId);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : 'Plot selection failed.',
      );
    } finally {
      onEnd?.();
    }
  }
  return (
    <>
      <section className="rounded-2xl border border-[#e0e0e0] bg-white p-5 shadow-[0_2px_8px_#00000014] sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-bold text-[#333333]">Farm and plots</h2>
          <div className="flex flex-wrap gap-2">
            {hasFarm && (
              <button
                type="button"
                className={buttonClass}
                disabled={!canOpen}
                onClick={() => open('plot')}
              >
                Add plot
              </button>
            )}
            {hasFarm && (
              <button
                type="button"
                className="inline-flex min-h-10 items-center rounded-lg border border-[#d5dfcc] px-4 py-2 text-sm font-medium text-[#42613d] hover:bg-[#f2f6ed] disabled:opacity-45"
                disabled={!canOpen}
                onClick={() => open('farm')}
              >
                Farm details
              </button>
            )}
          </div>
        </div>
        {hasFarm && context.plots.length > 0 && (
          <details className="mt-4 text-sm text-[#65765b]">
            <summary className="cursor-pointer py-1">
              Switch plot · {context.currentPlot?.name ?? 'Choose a plot'}
            </summary>
            <div className="mt-3 grid gap-2">
              {context.plots.map((plot) => (
                <div
                  key={plot.id}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#dce4ed] bg-white px-3.5 py-3 shadow-[0_2px_6px_#00000008]"
                >
                  <div>
                    <h3 className="text-base font-semibold text-[#1f2937]">
                      {plot.name}
                    </h3>
                    <p className="mt-0.5 text-xs text-[#64748b]">
                      {context.farm?.name} ·{' '}
                      {plot.areaM2 > 0
                        ? `${fromSquareMetres(plot.areaM2, plot.areaUnit).toFixed(2)} ${plot.areaUnit}`
                        : 'Not measured'}
                      {plot.id === context.currentPlot?.id
                        ? ' · Currently editing'
                        : ''}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {plot.id === context.currentPlot?.id && (
                      <span className="rounded-full bg-[#e5f3df] px-2.5 py-1 text-[10px] font-bold text-[#548f3c]">
                        ACTIVE
                      </span>
                    )}
                    <button
                      type="button"
                      disabled={unavailable}
                      onClick={() => void edit(plot.id)}
                      className="min-h-8 rounded-full border border-[#d5dce6] px-3 text-xs text-[#475569] hover:bg-[#f3f7f2] disabled:opacity-45"
                    >
                      Edit
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}
        {(error || reportedError) && !dialogOpen && (
          <p className="mt-3 text-sm text-red-700" role="alert">
            {error || reportedError}
          </p>
        )}
      </section>
      <dialog
        ref={dialog}
        aria-label={kind === 'plot' ? 'Add a plot' : 'Farm details'}
        onCancel={(event) => {
          if (busy) event.preventDefault();
        }}
        onClose={() => {
          setDialogOpen(false);
          onEnd?.();
        }}
        className="m-auto w-[min(34rem,calc(100%-2rem))] max-h-[90dvh] overflow-y-auto rounded-2xl border border-[#dce4ed] bg-white p-6 text-[#1f2937] shadow-xl backdrop:bg-black/35"
      >
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
          className="grid gap-5"
        >
          <h2 className="text-xl font-semibold">
            {kind === 'plot' ? 'Add a plot' : 'Farm details'}
          </h2>
          {kind === 'farm' ? (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-5 text-sm [&_dt]:text-xs [&_dt]:text-slate-500 [&_dd]:mt-1.5 [&_dd]:font-medium [&_dd]:text-[#304b3b]">
              <div className="col-span-2">
                <dt>Farm name</dt>
                <dd>{form.name}</dd>
              </div>
              <div className="col-span-2">
                <dt>Farm area</dt>
                <dd>
                  {Number(form.area).toLocaleString(undefined, {
                    maximumFractionDigits: 2,
                  })}{' '}
                  {form.unit === 'm2' ? 'm²' : form.unit}
                </dd>
              </div>
              <div>
                <dt>Latitude</dt>
                <dd>
                  {form.latitude
                    ? Number(form.latitude).toLocaleString(undefined, {
                        maximumFractionDigits: 6,
                      })
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>Longitude</dt>
                <dd>
                  {form.longitude
                    ? Number(form.longitude).toLocaleString(undefined, {
                        maximumFractionDigits: 6,
                      })
                    : '—'}
                </dd>
              </div>
            </dl>
          ) : (
            <fieldset
              disabled={busy}
              className="m-0 grid min-w-0 gap-4 border-0 p-0"
            >
              <Field
                label={kind === 'plot' ? 'Plot name' : 'Farm name'}
                value={form.name}
                onChange={(name) => setForm({ ...form, name })}
              />
              <fieldset className="grid grid-cols-2 gap-4">
                <Field
                  label={kind === 'plot' ? 'Plot area' : 'Farm size'}
                  decimalPlaces={2}
                  value={form.area}
                  type="number"
                  min={0}
                  onChange={(area) => setForm({ ...form, area })}
                />
                <SelectField
                  label="Area unit"
                  value={form.unit}
                  options={['m2', 'ha', 'acre'].map((v) => [
                    v,
                    v === 'm2' ? 'm²' : v,
                  ])}
                  onChange={(values) =>
                    setForm({ ...form, unit: values[0] as AreaUnit })
                  }
                />
              </fieldset>
            </fieldset>
          )}
          {error && (
            <p role="alert" className="text-sm text-red-700">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => dialog.current?.close()}
              className="min-h-10 rounded-lg border border-gray-300 px-4 text-sm"
            >
              {kind === 'farm' ? 'Close' : 'Cancel'}
            </button>
            {kind !== 'farm' && (
              <button
                type="submit"
                disabled={unavailable}
                title={
                  unavailable
                    ? 'Load the farm registry from the updated backend before saving.'
                    : undefined
                }
                className={buttonClass}
              >
                {busy
                  ? 'Saving…'
                  : kind === 'plot'
                    ? 'Create and configure plot'
                    : 'Save changes'}
              </button>
            )}
          </div>
        </form>
      </dialog>
    </>
  );
}
