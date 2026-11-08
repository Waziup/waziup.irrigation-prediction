import type { PolygonGeometry } from '../../../domain/geometry';

export function BoundaryPreview({
  farm,
  plot,
}: {
  farm: PolygonGeometry;
  plot: PolygonGeometry;
}) {
  const ring = farm.coordinates[0];
  const minX = Math.min(...ring.map((p) => p[0])),
    maxX = Math.max(...ring.map((p) => p[0]));
  const minY = Math.min(...ring.map((p) => p[1])),
    maxY = Math.max(...ring.map((p) => p[1]));
  const longitudeScale = Math.cos((((minY + maxY) / 2) * Math.PI) / 180);
  const width = (maxX - minX) * longitudeScale,
    height = maxY - minY;
  const scale = Math.min(480 / width, 220 / height);
  const offsetX = (520 - width * scale) / 2,
    offsetY = (260 - height * scale) / 2;
  const points = (geometry: PolygonGeometry) =>
    geometry.coordinates[0]
      .map(
        (p) =>
          `${offsetX + (p[0] - minX) * longitudeScale * scale},${offsetY + (maxY - p[1]) * scale}`,
      )
      .join(' ');
  return (
    <figure className="overflow-hidden rounded-xl border border-[#dce5d7] bg-[#edf3e8] sm:col-span-2">
      <svg
        viewBox="0 0 520 260"
        role="img"
        aria-label="Review farm and plot boundaries"
        className="max-h-72 w-full"
      >
        <polygon
          points={points(farm)}
          className="fill-white/60 stroke-[#356b30]"
          strokeWidth="2"
          strokeDasharray="6 4"
        />
        <polygon
          points={points(plot)}
          className="fill-[#8ab17d]/50 stroke-[#2d5a27]"
          strokeWidth="2"
        />
      </svg>
      <figcaption className="px-4 pb-3 text-xs text-[#53645a]">
        Dashed outline: farm · Filled outline: your plot
      </figcaption>
    </figure>
  );
}
