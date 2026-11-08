import type { ReactNode } from 'react';

export function DataSection({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <section className="min-w-0 rounded-2xl border border-[#e0e0e0] bg-white p-4 shadow-[0_1px_2px_#00000014] sm:p-5">
      <h2 className="mb-3 text-base font-bold text-[#111827]">{title}</h2>
      {children ?? <span className="text-gray-500">—</span>}
    </section>
  );
}
export function EmptyValues({ labels }: { labels: readonly string[] }) {
  return (
    <dl className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,180px),1fr))] gap-5">
      {labels.map((label) => (
        <div key={label}>
          <dt className="text-sm text-gray-600">{label}</dt>
          <dd className="mt-2 text-2xl text-[#2d5a27]">—</dd>
        </div>
      ))}
    </dl>
  );
}
