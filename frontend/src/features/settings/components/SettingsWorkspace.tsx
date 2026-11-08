import type { ReactNode } from 'react';
export function SettingsWorkspace({
  name,
  farmName,
  dirty,
  children,
}: {
  name?: string | undefined;
  farmName?: string | undefined;
  dirty?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      id="plot-settings-workspace"
      className="grid gap-3 rounded-[18px] border border-[#bfcfbd] bg-[#edf3eb] p-3.5"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 px-2 py-2">
        <div>
          <span className="text-[11px] font-extrabold uppercase tracking-widest text-[#367442]">
            {dirty ? 'Unsaved changes' : 'Configuration'}
          </span>
          <h2 className="mt-1 text-xl font-semibold text-[#1f2937]">
            {name ? `${name} settings` : 'Plot settings'}
          </h2>
          {farmName && (
            <p className="mt-1 text-xs text-[#647568]">Farm: {farmName}</p>
          )}
        </div>
      </header>
      {children}
    </section>
  );
}
