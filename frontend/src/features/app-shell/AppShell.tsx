import { FarmContext } from '../farm-context/FarmContext';
import { PlotHeader } from '../plot-overview/components/PlotHeader';
import { useContext, useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Link, Outlet, useLocation, useNavigation } from 'react-router';
import { Header } from './Header';
import { Navigation, PlotNavigation } from './Navigation';
import { LoadingBoundary, LoadingMessage } from './LoadingBoundary';
import { useShellRoute } from './hooks/useShellRoute';

export function AppShell({
  contextPending = false,
  contentKey,
  pageContent,
}: {
  contextPending?: boolean;
  contentKey?: string;
  pageContent?: ReactNode;
}) {
  const context = useContext(FarmContext);
  const { title, isPlotView } = useShellRoute();
  const navigation = useNavigation();
  const { pathname } = useLocation();
  const main = useRef<HTMLElement>(null);
  const navigationDialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const desktop = window.matchMedia('(min-width: 48rem)');
    const closeOnDesktop = () => {
      if (desktop.matches) navigationDialog.current?.close();
    };
    desktop.addEventListener('change', closeOnDesktop);
    return () => desktop.removeEventListener('change', closeOnDesktop);
  }, []);
  const pending = navigation.state !== 'idle' || contextPending;
  useEffect(() => {
    main.current?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [pathname]);
  return (
    <div className="min-h-dvh bg-[#f7f8f3]">
      <a
        href="#main-content"
        className="fixed top-2 left-2 z-50 -translate-y-[200%] rounded-lg bg-white p-3 text-[#2d5a27] shadow-sm focus:translate-y-0"
        onClick={(event) => {
          event.preventDefault();
          main.current?.focus();
        }}
      >
        Skip to content
      </a>
      <Navigation dialogRef={navigationDialog} />
      <div className="min-w-0 md:ml-56">
        <Header
          title={title}
          onOpenNavigation={() => navigationDialog.current?.showModal()}
        />
        <main
          id="main-content"
          ref={main}
          className={`mr-auto w-full px-4 pt-6 pb-16 focus:outline-none md:px-8 xl:px-10 xl:pt-8 ${pathname === '/settings' ? 'max-w-260' : 'max-w-360'}`}
          data-page={pathname}
          tabIndex={-1}
          aria-labelledby="page-title"
          aria-busy={pending}
        >
          {navigation.state !== 'idle' && <LoadingMessage />}
          {isPlotView && (
            <>
              <div className="mb-7 flex flex-wrap items-start justify-between gap-4">
                {context?.currentPlot && (
                  <div>
                    <p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                      Selected plot
                    </p>
                    <PlotHeader plot={context.currentPlot} />
                  </div>
                )}
                <div className="flex gap-2">
                  <Link
                    to="/dashboard"
                    className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs font-semibold text-[#345c3d]"
                  >
                    ← Farm overview
                  </Link>
                  <Link
                    to="/settings"
                    aria-label="Edit plot settings"
                    className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs font-semibold text-[#345c3d]"
                  >
                    Plot settings
                  </Link>
                </div>
              </div>
              <PlotNavigation />
            </>
          )}
          <LoadingBoundary>
            <div key={contentKey}>{pageContent ?? <Outlet />}</div>
          </LoadingBoundary>
        </main>
      </div>
    </div>
  );
}
