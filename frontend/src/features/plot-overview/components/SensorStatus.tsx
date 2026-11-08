import type { Device } from '../../../domain/sensor';
import type { PlotSettings } from '../../../domain/settings';
import { assignedSensors } from '../types';
import type { ResourceState } from '../types';
import { ResourcePanel } from './ResourcePanel';
import { StatusBadge } from '../../../components/StatusBadge';
export function SensorStatus({
  settings,
  devices,
  onRetry,
}: {
  settings: ResourceState<
    { configured: true; settings: PlotSettings } | { configured: false }
  >;
  devices: ResourceState<Device[]>;
  onRetry: () => void;
}) {
  return (
    <ResourcePanel title="Sensor assignment" state={settings} onRetry={onRetry}>
      {(result) => {
        if (!result.configured)
          return (
            <p className="my-3 leading-relaxed text-gray-600">
              Save plot settings to assign sensors.
            </p>
          );
        const assignments = assignedSensors(
          result.settings,
          devices.status === 'ready' ? devices.data : null,
        );
        return (
          <>
            {devices.status === 'error' && (
              <p className="my-3 leading-relaxed text-gray-600" role="status">
                Device discovery unavailable. Saved assignments are retained.
              </p>
            )}
            {devices.status === 'loading' && (
              <p className="my-3 leading-relaxed text-gray-600" role="status">
                Loading device names…
              </p>
            )}
            {assignments.length ? (
              <ul className="m-0 grid list-none gap-3 p-0 [&>li]:flex [&>li]:items-baseline [&>li]:justify-between [&>li]:gap-3 [&>li]:wrap-anywhere">
                {assignments.map((item) => (
                  <li key={`${item.role}:${item.id}`}>
                    <span>
                      <strong>{item.role}</strong> · {item.label}
                    </span>
                    <StatusBadge>
                      {item.discovered ? 'Assigned' : 'Saved source'}
                    </StatusBadge>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="my-3 leading-relaxed text-gray-600">
                No sensors assigned.
              </p>
            )}
            <p className="my-3 text-sm text-gray-600">
              Assignments and device discovery do not establish sensor
              freshness. Current readings are shown separately.
            </p>
          </>
        );
      }}
    </ResourcePanel>
  );
}
