import type { ChartPlugin } from './chartTypes';
function present(value: number | null): value is number {
  return value !== null && Number.isFinite(value);
}
export const soilDecorations: ChartPlugin = {
  id: 'soilChartDecorations',
  beforeDraw(chart, _args, options) {
    const area = chart.chartArea;
    const y = chart.scales.y;
    if (!area || !y || !options.enabled) return;
    const refs = options.references;
    const bands: [number | null, number | null, string][] = [
      [y.min, refs.lower, '#bfdbfe55'],
      [refs.lower, refs.upper, '#bbf7d055'],
      [refs.upper, refs.wilting, '#fed7aa55'],
      [refs.wilting, y.max, '#fecaca55'],
    ];
    const ctx = chart.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.rect(area.left, area.top, area.width, area.height);
    ctx.clip();
    for (const [from, to, color] of bands) {
      if (!present(from) || !present(to) || to <= from) continue;
      const bottom = y.getPixelForValue(Math.max(y.min, Math.min(y.max, from)));
      const top = y.getPixelForValue(Math.max(y.min, Math.min(y.max, to)));
      ctx.fillStyle = color;
      ctx.fillRect(area.left, top, area.width, bottom - top);
    }
    ctx.restore();
  },
  afterDatasetsDraw(chart, _args, options) {
    const area = chart.chartArea;
    const y = chart.scales.y;
    const x = chart.scales.x;
    if (!area || !y || !x) return;
    const ctx = chart.ctx;
    const refs = options.references;
    for (const [label, value] of [
      ['Saturation', refs.saturation],
      ['Field capacity lower', refs.lower],
      ['Field capacity upper', refs.upper],
      ['Permanent wilting point', refs.wilting],
      ['Irrigation trigger', refs.threshold],
    ] as const) {
      if (!options.enabled || !present(value) || value < y.min || value > y.max)
        continue;
      const pixel = y.getPixelForValue(value);
      ctx.save();
      ctx.strokeStyle = '#94a3b8';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(area.left, pixel);
      ctx.lineTo(area.right, pixel);
      ctx.stroke();
      ctx.font = '11px system-ui';
      const width = ctx.measureText(label).width + 8;
      ctx.fillStyle = '#ffffffee';
      ctx.fillRect(area.right - width, pixel - 10, width, 18);
      ctx.fillStyle = '#334155';
      ctx.fillText(label, area.right - width + 4, pixel + 3);
      ctx.restore();
    }
    const now = options.nowTimestamp;
    if (now !== undefined) {
      const divider = x.getPixelForValue(now);
      if (divider >= area.left && divider <= area.right) {
        ctx.save();
        ctx.strokeStyle = '#64748b';
        ctx.beginPath();
        ctx.moveTo(divider, area.top);
        ctx.lineTo(divider, area.bottom);
        ctx.stroke();
        ctx.fillStyle = '#475569';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('Today', divider, area.bottom + 40);

        ctx.restore();
      }
    }
  },
};
