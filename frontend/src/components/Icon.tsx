import type { SVGProps } from 'react';

export type IconName =
  | 'leaf'
  | 'dashboard'
  | 'settings'
  | 'plot'
  | 'growth'
  | 'alert'
  | 'menu'
  | 'close'
  | 'rain'
  | 'temperature'
  | 'humidity'
  | 'wind';
const paths: Record<IconName, string> = {
  rain: 'M5 14a4 4 0 0 1 0-8 6 6 0 0 1 11-1 4.5 4.5 0 0 1 2 9M7 17l-1 3m6-3-1 3m6-3-1 3',
  temperature: 'M9 14.5V5a3 3 0 0 1 6 0v9.5a5 5 0 1 1-6 0ZM12 8v10m0 0v1',
  humidity:
    'M12 3C9 7 5 11 5 15a7 7 0 0 0 14 0c0-4-4-8-7-12ZM8 15a4 4 0 0 0 4 4',
  wind: 'M3 8h12a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h6a3 3 0 1 1-3 3',
  menu: 'M3 6h18M3 12h18M3 18h18',
  close: 'm6 6 12 12M6 18 18 6',
  leaf: 'M20 4c-8-1-15 2-15 8a7 7 0 0 0 7 7c6 0 9-7 8-15ZM4 21 15 10',
  dashboard: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  settings:
    'M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1M5.6 18.4l2.1-2.1m8.6-8.6 2.1-2.1M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z',
  plot: 'M3 5h18v14H3zM3 12h18M11 5v14',
  growth: 'M4 20V10m7 10V5m7 15V2M2 20h20',
  alert: 'M12 3 2 21h20L12 3Zm0 6v5m0 3v1',
};
export function Icon({
  name,
  className,
  ...props
}: SVGProps<SVGSVGElement> & { name: IconName }) {
  return (
    <svg
      className={className ? `shrink-0 ${className}` : 'shrink-0'}
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      <path d={paths[name]} />
    </svg>
  );
}
