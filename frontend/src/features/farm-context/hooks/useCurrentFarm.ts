import { useFarmContext } from '../FarmContext';
export function useCurrentFarm() {
  return useFarmContext().farm;
}
