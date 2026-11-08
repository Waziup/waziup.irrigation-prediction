import { useId } from 'react';
import type { ReactNode } from 'react';
import type { ResourceState } from '../types';
import { LoadingMessage } from '../../app-shell/LoadingBoundary';
export function ResourcePanel<T>({
  title,
  state,
  onRetry,
  children,
  bare = false,
  footer,
}: {
  title: string;
  bare?: boolean;
  footer?: ReactNode;
  state: ResourceState<T>;
  onRetry: () => void;
  children: (data: T) => ReactNode;
}) {
  const headingId = useId();
  return (
    <section
      className={
        bare
          ? 'min-w-0'
          : 'min-w-0 rounded-2xl border border-[#e0e0e0] bg-white p-4 shadow-[0_1px_2px_#00000014] sm:p-5 [&>h2]:mt-0 [&_details]:my-5 [&_summary]:cursor-pointer [&_summary]:py-2 [&_summary]:text-[#2d5a27]'
      }
      aria-labelledby={headingId}
    >
      <h2 className="mb-3 text-base font-bold text-[#111827]" id={headingId}>
        {title}
      </h2>
      {state.status === 'loading' ? (
        <LoadingMessage message={`Loading ${title.toLowerCase()}…`} />
      ) : state.status === 'error' ? (
        <>
          <p className="my-3 leading-relaxed text-gray-600" role="alert">
            {state.error.message}
          </p>
          <button
            type="button"
            className="min-h-11 cursor-pointer rounded-lg border border-[#aac9a5] bg-[#f5faf4] px-4 py-2.5 font-semibold text-[#2d5a27] hover:bg-[#e8f5e6] disabled:cursor-wait disabled:opacity-65"
            onClick={onRetry}
          >
            Retry {title.toLowerCase()}
          </button>
        </>
      ) : (
        children(state.data)
      )}
      {footer}
    </section>
  );
}
