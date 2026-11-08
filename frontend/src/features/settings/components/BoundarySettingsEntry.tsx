import { Link } from 'react-router';
import { mapButton } from '../../farm-map/MapControls';

export function BoundarySettingsEntry() {
  return (
    <section className="mb-5 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-[#dde5d6] bg-white p-5">
      <div>
        <h2 className="text-sm font-semibold text-[#304b3b]">
          Farm &amp; plot boundaries
        </h2>
        <p className="mt-1 text-xs text-[#849079]">
          Draw and manage the outlines used on your farm dashboard.
        </p>
      </div>
      <Link to="/settings/boundaries" className={`${mapButton} no-underline`}>
        Manage boundaries
      </Link>
    </section>
  );
}
