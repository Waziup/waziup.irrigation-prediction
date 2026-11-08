import { Suspense } from 'react';
import type { ReactNode } from 'react';

export function LoadingMessage({
  message = 'Loading page…',
}: {
  message?: string;
}) {
  return (
    <div className="flex items-center gap-3 py-4 text-gray-600" role="status">
      <span
        className="size-5 animate-spin rounded-full border-2 border-gray-300 border-t-[#2d5a27] motion-reduce:animate-none"
        aria-hidden="true"
      />
      <span>{message}</span>
    </div>
  );
}
export function LoadingBoundary({ children }: { children: ReactNode }) {
  return <Suspense fallback={<LoadingMessage />}>{children}</Suspense>;
}
