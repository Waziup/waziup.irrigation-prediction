import type { Farm } from '../../domain/farm';
import type { Plot, PlotContext } from '../../domain/plot';

export interface FarmContextValue {
  farm: Farm | null;
  currentPlot: Plot | null;
  plots: Plot[];
  snapshot: PlotContext | null;
  status: 'loading' | 'refreshing' | 'switching' | 'ready' | 'error';
  error: Error | null;
  selectionError: Error | null;
  revision: number;
  contextHeld?: boolean;
  holdContext?: () => () => void;
  selectPlot: (plotId: string) => Promise<void>;
  refresh: () => Promise<void>;
}
