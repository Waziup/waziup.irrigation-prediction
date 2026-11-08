const readingLabels = [
  'Soil tension',
  'Active field threshold',
  'Soil temperature',
  'Soil VWC',
] as const;
export function ReadingCards({ values }: { values?: readonly string[] }) {
  return (
    <div className="grid grid-cols-2 gap-2.5 min-[1000px]:grid-cols-4">
      {readingLabels.map((label, index) => (
        <div
          key={label}
          className="flex min-w-0 flex-col min-h-26 justify-between gap-6 rounded-[13px] border border-gray-200 bg-white px-3.5 py-4"
        >
          <span className="text-xs text-[#64748b]">{label}</span>
          <strong className="text-2xl font-semibold text-[#253247]">
            {values?.[index] ?? '—'}
          </strong>
        </div>
      ))}
    </div>
  );
}
