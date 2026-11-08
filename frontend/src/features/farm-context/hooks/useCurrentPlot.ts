import { useFarmContext } from '../FarmContext';
export function useCurrentPlot() {
  return useFarmContext().currentPlot;
}
