import type { FarmPlotCard } from '../../domain/farm';

export const mapStatuses = {
  unavailable: {
    label: 'Advice unavailable',
    color: '#94a3b8',
    dot: 'bg-[#94a3b8]',
  },
  stale: {
    label: 'Advice needs updating',
    color: '#94a3b8',
    dot: 'bg-[#94a3b8]',
  },
  active: {
    label: 'Irrigation in progress',
    color: '#2563eb',
    dot: 'bg-[#2563eb]',
  },
  critical: { label: 'Water needed', color: '#c62828', dot: 'bg-[#c62828]' },
  advise: {
    label: 'Irrigation advised',
    color: '#e6a800',
    dot: 'bg-[#e6a800]',
  },
  watch: { label: 'Needs attention', color: '#e6a800', dot: 'bg-[#e6a800]' },
  healthy: { label: 'Healthy', color: '#3a7d34', dot: 'bg-[#3a7d34]' },
} as const;
export type MapStatus = keyof typeof mapStatuses;
export function mapStatus(plot: FarmPlotCard): MapStatus {
  if (plot.dataState !== 'fresh') return plot.dataState;
  if (plot.operation?.status === 'active' || plot.equipment.state === 'active')
    return 'active';
  const urgencies = [plot.urgency, plot.alert?.urgency];
  if (
    urgencies.includes('critical') ||
    plot.conditionStatus === 'threshold_exceeded'
  )
    return 'critical';
  if (urgencies.includes('advise') || plot.conditionStatus === 'at_threshold')
    return 'advise';
  if (urgencies.includes('watch')) return 'watch';
  return plot.conditionStatus === 'within_threshold'
    ? 'healthy'
    : 'unavailable';
}
