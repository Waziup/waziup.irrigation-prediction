import { createContext, useContext } from 'react';
import type { FarmContextValue } from './types';

export const FarmContext = createContext<FarmContextValue | null>(null);
export function useFarmContext(): FarmContextValue {
  const value = useContext(FarmContext);
  if (!value)
    throw new Error('Farm context must be used inside FarmContextProvider.');
  return value;
}
