import { createContext } from 'react';

// Presentation only: unloaded selects must not display a saved/default value.
export const SettingsDataContext = createContext(true);
