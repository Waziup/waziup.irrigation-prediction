import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { pumpApi } from '../../../api/pumpApi';
import { ApiError } from '../../../api/errors';
import { useFarmContext } from '../../farm-context/FarmContext';

function requestId() {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `pump-${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
}

export function PumpControls({
  plotId,
  embedded = false,
}: {
  plotId: string;
  embedded?: boolean;
}) {
  const context = useFarmContext();
  const client = useQueryClient();
  const [amount, setAmount] = useState('1');
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState('');
  const [failed, setFailed] = useState(false);
  const [operationId, setOperationId] = useState<string | null>(null);
  const request = useRef<{ amount: string; key: string } | null>(null);
  const status = useQuery({
    queryKey: ['plot', plotId, 'pump-controls'],
    queryFn: ({ signal }) => pumpApi.status(plotId, { signal }),
    refetchInterval: 15_000,
    retry: false,
    refetchOnWindowFocus: true,
  });
  const data = status.data?.plotId === plotId ? status.data : undefined;
  useEffect(() => {
    if (!operationId) return;
    return client.getQueryCache().subscribe((event) => {
      const key = event.query.queryKey;
      if (key[0] !== 'plot' || key[1] !== plotId || key[2] !== 'pump-controls')
        return;
      const latest = client.getQueryData<NonNullable<typeof data>>(key);
      if (latest?.operationId !== operationId) return;
      if (latest.operationStatus === 'verified') {
        setAmount('');
        request.current = null;
        setOperationId(null);
        setFailed(false);
        setMessage('Done');
      } else if (latest.operationStatus === 'failed') {
        setOperationId(null);
        setFailed(true);
        setMessage(
          'Delivery could not be confirmed. Check the pump and flow meter.',
        );
      }
    });
  }, [client, plotId, operationId]);
  const configured = !!data?.configured;
  const disabled = pending || context.status !== 'ready';
  const canStart = configured && data?.mode === 'manual' && !data.busy;
  const validAmount =
    Number.isFinite(Number(amount)) &&
    Number(amount) >= 0.01 &&
    Number(amount) <= 1000;
  const change = (value: string) => {
    setAmount(value);
    setMessage('');
    request.current = null;
  };
  async function command(state: 'on' | 'off') {
    if (disabled) return;
    if (!request.current || request.current.amount !== amount) {
      request.current = { amount, key: requestId() };
    }
    const release = context.holdContext?.();
    setPending(true);
    setMessage('');
    setFailed(false);
    try {
      const result = await pumpApi.command(
        plotId,
        state,
        amount,
        state === 'on' ? request.current.key : requestId(),
      );
      if (result.plotId !== plotId) throw new Error('Unexpected plot response');
      if (state === 'on') setOperationId(result.operationId);
      setMessage(state === 'on' ? '' : 'OFF command sent.');
      request.current = null;
      await status.refetch();
      void client.invalidateQueries({ queryKey: ['farm-dashboard'] });
    } catch (error) {
      setFailed(true);
      const payload = error instanceof ApiError ? error.payload : null;
      setMessage(
        payload &&
          typeof payload === 'object' &&
          'message' in payload &&
          typeof payload.message === 'string'
          ? payload.message
          : 'The command could not be confirmed. Check the controller before trying again.',
      );
    } finally {
      release?.();
      setPending(false);
    }
  }
  const fieldDisabled = disabled || !canStart;
  return (
    <section
      aria-label="Water pump"
      className={
        embedded
          ? 'min-w-0'
          : 'mb-6 rounded-xl border border-[#e0e6d9] bg-white px-5 py-4'
      }
    >
      <div
        className={
          embedded
            ? 'flex min-w-0 flex-col items-start gap-3'
            : 'flex flex-wrap items-center justify-between gap-4'
        }
      >
        <h2 className="text-base font-semibold text-[#253247]">Water pump</h2>
        <div className="flex flex-wrap items-center gap-3">
          <label
            htmlFor={`pump-amount-${plotId}`}
            className="text-sm text-[#53645a]"
          >
            Irrigation amount
          </label>
          <div className="flex items-center gap-1 rounded-lg border border-slate-200 bg-[#f7f9fb] p-1">
            <button
              type="button"
              aria-label="Decrease irrigation amount"
              disabled={fieldDisabled || Number(amount) <= 0.01}
              onClick={() =>
                change(
                  Math.max(
                    0.01,
                    Number((Number(amount) - 0.1).toFixed(2)),
                  ).toString(),
                )
              }
              className="size-10 rounded-md border border-slate-200 bg-white text-lg disabled:opacity-40"
            >
              −
            </button>
            <input
              id={`pump-amount-${plotId}`}
              type="number"
              inputMode="decimal"
              min="0.01"
              max="1000"
              step="0.01"
              value={amount}
              disabled={fieldDisabled}
              onChange={(event) => change(event.target.value)}
              className="min-h-10 w-16 bg-transparent text-center text-sm font-semibold disabled:text-slate-400"
            />
            <button
              type="button"
              aria-label="Increase irrigation amount"
              disabled={fieldDisabled || Number(amount) >= 1000}
              onClick={() =>
                change(
                  Math.min(
                    1000,
                    Number((Number(amount) + 0.1).toFixed(2)),
                  ).toString(),
                )
              }
              className="size-10 rounded-md border border-slate-200 bg-white text-lg disabled:opacity-40"
            >
              +
            </button>
            <span className="px-2 text-xs text-slate-500">m³</span>
          </div>
        </div>
        <div className="flex items-center gap-5">
          <button
            type="button"
            disabled={disabled || !canStart || !validAmount}
            onClick={() => void command('on')}
            className="min-h-12 min-w-24 rounded-lg bg-[#287b53] px-4 py-2 text-sm font-bold text-white disabled:opacity-40"
          >
            ON
          </button>
          <button
            type="button"
            disabled={disabled || !configured}
            onClick={() => void command('off')}
            className="min-h-12 min-w-24 rounded-lg bg-[#cf3636] px-4 py-2 text-sm font-bold text-white disabled:opacity-40"
          >
            OFF
          </button>
        </div>
      </div>
      {status.isError && (
        <p className="mt-3 text-xs text-[#53645a]">
          Pump controls could not be loaded.{' '}
          <button
            type="button"
            onClick={() => void status.refetch()}
            className="underline"
          >
            Retry
          </button>
        </p>
      )}
      {data && !configured && (
        <p className="mt-3 text-xs text-[#53645a]">
          Select an irrigation actuator in Plot settings to use these controls.
        </p>
      )}
      {configured && data?.mode !== 'manual' && (
        <p className="mt-3 text-xs text-[#53645a]">
          Select Semi-Auto to start irrigation yourself. OFF remains available.
        </p>
      )}
      {data?.busy && (
        <p className="mt-3 text-xs text-[#53645a]">
          An irrigation request is active or awaiting delivery confirmation.
        </p>
      )}
      {message && (
        <p
          role={failed ? 'alert' : 'status'}
          className={
            message === 'Done'
              ? 'mt-3 text-sm font-semibold text-[#287b53]'
              : 'mt-3 text-xs text-[#53645a]'
          }
        >
          {message}
        </p>
      )}
    </section>
  );
}
