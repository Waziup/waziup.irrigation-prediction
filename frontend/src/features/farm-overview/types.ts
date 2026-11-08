import type { FarmMap } from '../../domain/map';
import type { FarmDashboard } from '../../domain/farm';

export interface FarmOverviewContentProps {
  dashboard: FarmDashboard;
  farmMap?: FarmMap;
  mapError?: string;
  weather?: import('react').ReactNode;
  currentPlotId: string;
  onSelectPlot: (plotId: string) => void;
  onConfigurePlot?: (plotId: string) => void;
  disabled?: boolean;
}
export {
  formatAmount,
  formatDashboardTime,
  urgencyPresentation,
} from '../../shared/presentation';
