import { InstallationGate } from '../farm-setup/InstallationGate';
import { AppShell } from '../app-shell/AppShell';
import { FarmContextProvider } from './FarmContextProvider';
import { useFarmContext } from './FarmContext';

function ContextShell() {
  const context = useFarmContext();
  return (
    <AppShell
      contextPending={context.status !== 'ready' && context.status !== 'error'}
      contentKey={context.currentPlot?.id ?? 'none'}
    />
  );
}
export function FarmContextLayout() {
  return (
    <InstallationGate>
      <FarmContextProvider>
        <ContextShell />
      </FarmContextProvider>
    </InstallationGate>
  );
}
