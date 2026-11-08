import type { ChartConstructor } from './chartTypes';
let loading: Promise<ChartConstructor> | undefined;
export function loadChart(): Promise<ChartConstructor> {
  const chartWindow = window as Window & { Chart?: ChartConstructor };
  if (chartWindow.Chart?.version === '4.4.0')
    return Promise.resolve(chartWindow.Chart);
  if (loading) return loading;
  loading = new Promise<ChartConstructor>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = new URL(
      `${import.meta.env.BASE_URL}vendor/chart.umd.min.js`,
      document.baseURI,
    ).href;
    script.async = true;
    script.onload = () => {
      if (chartWindow.Chart?.version === '4.4.0') resolve(chartWindow.Chart);
      else {
        script.remove();
        reject(
          new Error('The expected chart library could not be initialized.'),
        );
      }
    };
    script.onerror = () => {
      script.remove();
      reject(new Error('The chart library could not be loaded.'));
    };
    document.head.append(script);
  }).catch((error) => {
    loading = undefined;
    throw error;
  });
  return loading;
}
