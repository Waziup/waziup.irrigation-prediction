import { useEffect } from 'react';
import { useLocation } from 'react-router';

const titles: Record<string, string> = {
  '/': 'Farm overview',
  '/dashboard': 'Farm overview',
  '/settings/boundaries': 'Farm boundaries',
  '/plot': 'Plot overview',
  '/phenology': 'Growth & water demand',
  '/settings': 'Settings',
};
export function useShellRoute() {
  const { pathname } = useLocation();
  const title = titles[pathname] ?? 'Page not found';
  useEffect(() => {
    document.title = `${title} · WaziFarm`;
  }, [title]);
  return {
    title,
    isPlotView: pathname === '/plot' || pathname === '/phenology',
  };
}
