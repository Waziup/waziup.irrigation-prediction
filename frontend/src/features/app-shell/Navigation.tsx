import { Brand } from '../../components/Brand';
import type { RefObject } from 'react';
import { Link, NavLink } from 'react-router';
import { Icon } from '../../components/Icon';

function NavigationLinks({
  label,
  onNavigate,
}: {
  label: string;
  onNavigate?: () => void;
}) {
  return (
    <>
      <Link
        to="/dashboard"
        onClick={onNavigate}
        className="flex items-center gap-2.5 px-2.5 pb-6 text-xl font-semibold tracking-tight text-[#f0f5e9] no-underline"
      >
        <Brand />
      </Link>
      <nav aria-label={label} className="grid gap-2">
        <NavLink
          to="/dashboard"
          onClick={onNavigate}
          className="flex min-h-12 items-center gap-3 rounded-xl px-3.5 py-3 text-[13px] font-medium text-[#b0c0a7] no-underline transition hover:bg-[#2b4634] hover:text-white aria-[current=page]:bg-[#3a543a] aria-[current=page]:text-[#f0f5e7]"
        >
          <Icon name="dashboard" />
          <span>Dashboard</span>
        </NavLink>
        <NavLink
          to="/settings"
          onClick={onNavigate}
          className="flex min-h-12 items-center gap-3 rounded-xl px-3.5 py-3 text-[13px] font-medium text-[#b0c0a7] no-underline transition hover:bg-[#2b4634] hover:text-white aria-[current=page]:bg-[#3a543a] aria-[current=page]:text-[#f0f5e7]"
        >
          <Icon name="settings" />
          <span>Farm settings</span>
        </NavLink>
      </nav>
    </>
  );
}

export function Navigation({
  dialogRef,
}: {
  dialogRef: RefObject<HTMLDialogElement | null>;
}) {
  const close = () => dialogRef.current?.close();
  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-56 flex-col overflow-y-auto bg-[#213d2c] px-4 py-7 md:flex">
        <NavigationLinks label="Primary navigation" />
      </aside>
      <dialog
        ref={dialogRef}
        aria-label="Application navigation"
        className="fixed inset-y-0 left-0 m-0 h-dvh max-h-none w-72 max-w-[calc(100%-2rem)] overflow-y-auto border-0 border-r border-gray-200 bg-[#213d2c] p-4 text-white shadow-xl backdrop:bg-black/40"
      >
        <button
          type="button"
          aria-label="Close navigation"
          className="mb-4 ml-auto flex min-h-11 min-w-11 items-center justify-center rounded-lg border border-gray-200 text-[#2d5a27] hover:bg-gray-50"
          onClick={close}
        >
          <Icon name="close" />
        </button>
        <NavigationLinks label="Mobile navigation" onNavigate={close} />
      </dialog>
    </>
  );
}
export function PlotNavigation() {
  return (
    <nav
      aria-label="Selected plot sections"
      className="mb-5 flex flex-wrap gap-5 border-b border-[#d8dee8]"
    >
      <NavLink
        to="/plot"
        end
        className="inline-flex items-center gap-2 border-b-2 border-transparent px-1 py-3 text-sm text-[#64748b] no-underline hover:text-[#174f2a] aria-[current=page]:border-[#315f48] aria-[current=page]:font-semibold aria-[current=page]:text-[#174f2a]"
      >
        <Icon name="plot" />
        Overview
      </NavLink>
      <NavLink
        to="/phenology"
        className="inline-flex items-center gap-2 border-b-2 border-transparent px-1 py-3 text-sm text-[#64748b] no-underline hover:text-[#174f2a] aria-[current=page]:border-[#315f48] aria-[current=page]:font-semibold aria-[current=page]:text-[#174f2a]"
      >
        <Icon name="growth" />
        Growth &amp; water demand
      </NavLink>
    </nav>
  );
}
