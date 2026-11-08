// Deliberately small adapter for the unchanged legacy Chart.js 4.4.0 UMD build.
export interface LineDataset {
  label: string;
  data: (number | null)[];
  borderColor: string;
  borderWidth: number;
  pointRadius: number;
  pointHoverRadius?: number;
  borderDash?: number[];
  tension?: number;
  yAxisID: string;
  fill?: boolean;
  spanGaps?: boolean;
}
export interface ChartDrawing {
  ctx: CanvasRenderingContext2D;
  chartArea?: {
    left: number;
    right: number;
    top: number;
    bottom: number;
    width: number;
    height: number;
  };
  scales: Record<
    string,
    { min: number; max: number; getPixelForValue: (value: number) => number }
  >;
}
export interface SoilReferences {
  saturation: number | null;
  lower: number | null;
  upper: number | null;
  wilting: number | null;
  threshold: number | null;
}
export interface SoilOptions {
  enabled: boolean;
  historyCount: number;
  nowTimestamp?: number | undefined;
  references: SoilReferences;
}
export interface ChartPlugin {
  id: string;
  beforeDraw?: (
    chart: ChartDrawing,
    args: unknown,
    options: SoilOptions,
  ) => void;
  afterDatasetsDraw?: (
    chart: ChartDrawing,
    args: unknown,
    options: SoilOptions,
  ) => void;
}
export interface ForecastConfiguration {
  type: 'line';
  data: { labels: (string | number)[]; datasets: LineDataset[] };
  plugins: ChartPlugin[];
  options: {
    responsive: boolean;
    maintainAspectRatio: boolean;
    animation: false;
    interaction: { mode: 'index'; intersect: boolean };
    layout: { padding: { bottom: number } };
    plugins: {
      soilChartDecorations: SoilOptions;
      legend: { position: 'top' };
      tooltip?: {
        callbacks: { title: (items: { parsed: { x: number } }[]) => string };
      };
    };
    scales: Record<
      string,
      {
        type?: string;
        min?: number;
        max?: number;
        display?: boolean;
        position?: string;
        beginAtZero?: boolean;
        title?: { display: boolean; text: string };
        grid?: { drawOnChartArea?: boolean };
        ticks?: {
          maxTicksLimit: number;
          maxRotation?: number;
          callback?: (value: string | number) => string;
        };
      }
    >;
  };
}
export interface ChartInstance {
  data: ForecastConfiguration['data'];
  options: ForecastConfiguration['options'];
  update: (mode: 'none') => void;
  destroy: () => void;
}
export interface ChartConstructor {
  new (canvas: HTMLCanvasElement, config: ForecastConfiguration): ChartInstance;
  version: string;
}
