import type { ReactNode } from 'react';
export type BadgeTone = 'neutral' | 'success' | 'warning' | 'danger';
const tones: Record<BadgeTone, string> = {
  neutral: 'bg-gray-100 text-gray-600',
  success: 'bg-[#e8f5e6] text-[#245d2a]',
  warning: 'bg-[#fff8e1] text-[#775300]',
  danger: 'bg-[#fde8e8] text-[#a52020]',
};
export function StatusBadge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: BadgeTone;
}) {
  return (
    <span
      className={`inline-flex shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold ${tones[tone]}`}
    >
      {children}
    </span>
  );
}
