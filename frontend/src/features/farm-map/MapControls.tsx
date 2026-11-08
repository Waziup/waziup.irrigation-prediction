export type Imagery = 'illustrated' | 'light' | 'satellite';
export const mapButton =
  'inline-flex min-h-10 items-center justify-center gap-2 rounded-lg border border-[#dce4dc] bg-white px-3.5 py-2 text-sm font-medium text-[#244636] transition hover:border-[#a6bda9] hover:bg-[#f3f7f2] disabled:cursor-not-allowed disabled:opacity-45';
export function MapControls({
  imagery,
  onImagery,
  onFit,
}: {
  imagery: Imagery;
  onImagery: (mode: Imagery) => void;
  onFit: () => void;
}) {
  return (
    <div
      className="flex flex-wrap items-center gap-2"
      aria-label="Map controls"
    >
      <div className="flex rounded-xl border border-white/70 bg-white/95 p-1 shadow-lg shadow-[#183424]/10">
        <button
          type="button"
          aria-label="Show street map"
          aria-pressed={imagery === 'light'}
          onClick={() => onImagery('light')}
          className="min-h-10 rounded-lg px-4 text-xs font-semibold text-[#52675b] transition aria-pressed:bg-[#203f30] aria-pressed:text-white"
        >
          Map
        </button>
        <button
          type="button"
          aria-label="Use illustrated map"
          aria-pressed={imagery === 'illustrated'}
          onClick={() => onImagery('illustrated')}
          className="min-h-10 rounded-lg px-4 text-xs font-semibold text-[#52675b] transition aria-pressed:bg-[#203f30] aria-pressed:text-white"
        >
          Boundaries
        </button>
        <button
          type="button"
          aria-label="Show satellite map"
          aria-pressed={imagery === 'satellite'}
          onClick={() => onImagery('satellite')}
          className="min-h-10 rounded-lg px-3 text-xs font-semibold text-[#52675b] transition aria-pressed:bg-[#203f30] aria-pressed:text-white"
        >
          Satellite
        </button>
      </div>
      <button
        type="button"
        className="hidden size-12 sm:flex items-center justify-center rounded-xl border border-white/70 bg-white/95 text-[#244636] shadow-lg shadow-[#183424]/10 hover:bg-white"
        aria-label="Show whole farm"
        title="Show whole farm"
        onClick={onFit}
      >
        <svg
          width="19"
          height="19"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          aria-hidden="true"
        >
          <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M8 12h8m-4-4v8" />
        </svg>
      </button>
    </div>
  );
}
