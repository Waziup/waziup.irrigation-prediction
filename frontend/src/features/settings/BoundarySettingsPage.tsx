import { lazy, Suspense, useState } from 'react';
import { Link } from 'react-router';
import { useFarmMap } from '../farm-map/hooks/useFarmMap';
import { useFarmContext } from '../farm-context/FarmContext';
import type { BoundaryTarget } from '../farm-map/boundaryActions';
import { mapButton } from '../farm-map/MapControls';
const BoundaryEditor = lazy(() =>
  import('../farm-map/BoundaryEditPanel').then((module) => ({
    default: module.BoundaryEditPanel,
  })),
);

export function BoundarySettingsPage() {
  const query = useFarmMap();
  const context = useFarmContext();
  const [editing, setEditing] = useState<BoundaryTarget | null>(null);
  if (editing && query.data)
    return (
      <Suspense fallback={<p role="status">Loading boundary editor…</p>}>
        <BoundaryEditor
          data={query.data}
          target={editing}
          onClose={() => setEditing(null)}
        />
      </Suspense>
    );
  if (query.isError)
    return (
      <section className="grid gap-3">
        <h2 className="text-lg font-semibold">Boundaries unavailable</h2>
        <p role="alert">{query.error.message}</p>
        <button
          className={mapButton}
          onClick={() => {
            void query.refetch();
          }}
        >
          Retry
        </button>
      </section>
    );
  if (!query.data || context.status !== 'ready')
    return <p role="status">Loading farm boundaries…</p>;
  const data = query.data;
  if (!data.farm)
    return (
      <section>
        <h2 className="text-lg font-semibold">Create your farm</h2>
        <Link to="/settings/farm/new">Add farm and first plot</Link>
      </section>
    );
  return (
    <div className="grid gap-5">
      <header>
        <Link to="/settings" className="text-xs text-[#7b8b71] no-underline">
          ← Farm settings
        </Link>
        <h2 className="mt-3 text-2xl font-semibold text-[#243f2c]">
          Farm &amp; plot boundaries
        </h2>
        <p className="mt-2 text-sm text-[#819077]">
          Adjust the saved farm and plot outlines.
        </p>
      </header>
      <section className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-[#dde5d6] bg-white p-5">
        <div>
          <h3 className="font-semibold text-[#304b3b]">{data.farm.name}</h3>
          <p className="mt-1 text-xs text-[#819077]">
            {(data.farm.areaM2 / 10000).toFixed(2)} ha · Farm boundary
          </p>
        </div>
        <button
          className={mapButton}
          onClick={() => setEditing({ entity: 'farm' })}
        >
          Edit farm boundary
        </button>
      </section>
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-[#304b3b]">Plot boundaries</h3>
        <Link to="/settings/plots/new" className={`${mapButton} no-underline`}>
          Add plot
        </Link>
      </div>
      <section className="divide-y divide-[#e5eadf] overflow-hidden rounded-xl border border-[#dde5d6] bg-white">
        {data.plots.map((plot) => (
          <article
            key={plot.plotId}
            className="flex flex-wrap items-center justify-between gap-3 p-5"
          >
            <div>
              <h3 className="text-sm font-semibold text-[#304b3b]">
                {plot.name}
              </h3>
              <p className="mt-1 text-xs text-[#819077]">
                {plot.cropType || 'No crop selected'} ·{' '}
                {(plot.areaM2 / 10000).toFixed(2)} ha
              </p>
            </div>
            <button
              className={mapButton}
              onClick={() =>
                setEditing({ entity: 'plot', plotId: plot.plotId })
              }
              aria-label={`Edit boundary ${plot.name}`}
            >
              Edit boundary
            </button>
          </article>
        ))}
      </section>
      <p className="text-xs text-[#849079]">
        Every saved plot requires a valid boundary within its farm. Areas are
        calculated from the saved outlines.
      </p>
    </div>
  );
}
