import { mapStatuses } from './mapStyles';
export function MapLegend() {
  return (
    <ul
      aria-label="Map legend"
      className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-[#52606d]"
    >
      {(['critical', 'watch', 'healthy', 'active', 'unavailable'] as const).map(
        (status) => (
          <li key={status} className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className={`size-3 rounded-full ${mapStatuses[status].dot}`}
            />
            {mapStatuses[status].label}
          </li>
        ),
      )}
    </ul>
  );
}
