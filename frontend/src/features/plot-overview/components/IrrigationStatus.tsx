import type { LegacyIrrigationConfiguration } from '../../../domain/sensor';
import type { PlotSettings } from '../../../domain/settings';
import type { ResourceState } from '../types';
import { irrigationModeLabels } from '../types';
import { ResourcePanel } from './ResourcePanel';
export function IrrigationStatus({
  state,
  settings,
  onRetry,
}: {
  state: ResourceState<LegacyIrrigationConfiguration>;
  settings: ResourceState<
    { configured: true; settings: PlotSettings } | { configured: false }
  >;
  onRetry: () => void;
}) {
  const mode =
    settings.status === 'ready' && settings.data.configured
      ? settings.data.settings.irrigationMode
      : null;
  return (
    <ResourcePanel title="Irrigation status" state={state} onRetry={onRetry}>
      {(configuration) => (
        <dl className="my-4 grid grid-cols-[repeat(auto-fit,minmax(min(100%,180px),1fr))] gap-5 [&_dt]:text-sm [&_dt]:text-gray-600 [&_dd]:mt-2 [&_dd]:wrap-anywhere">
          <div>
            <dt>Mode</dt>
            <dd>{mode ? irrigationModeLabels[mode] : 'Unavailable'}</dd>
          </div>
          <div>
            <dt>Actuator configuration</dt>
            <dd>
              {configuration.hasConfiguredFlowActuator
                ? 'Actuator configured'
                : 'No actuator configured'}
            </dd>
          </div>
          <div>
            <dt>Live pump state</dt>
            <dd>Unavailable</dd>
          </div>
        </dl>
      )}
    </ResourcePanel>
  );
}
