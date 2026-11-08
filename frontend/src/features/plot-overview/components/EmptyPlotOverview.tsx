import { GrowthDetails } from './GrowthDetails';
import { ReadingCards } from './ReadingCards';
import { DataSection, EmptyValues } from '../../../components/DataSection';
export function EmptyPlotOverview() {
  return (
    <div className="grid gap-6">
      <DataSection title="Irrigation recommendation">
        <EmptyValues labels={['Next action', 'Urgency', 'Threshold']} />
      </DataSection>
      <section>
        <h2 className="mb-3 text-lg font-bold text-[#111827]">
          Current readings
        </h2>
        <ReadingCards />
      </section>
      <DataSection title="Plot alerts" />
      <DataSection title="Predicted sensor values">
        <div
          role="img"
          aria-label="Observed and forecast soil moisture"
          className="h-85 rounded-lg border border-gray-200 bg-gray-50 min-[651px]:h-107.5"
        />
      </DataSection>
      <DataSection title="Earth observation">
        <EmptyValues labels={['NDVI', 'NDRE', 'Latest observation']} />
      </DataSection>
    </div>
  );
}
export function EmptyGrowthOverview() {
  return (
    <div className="grid gap-6">
      <DataSection title="Growth stage timeline">
        <EmptyValues labels={['Crop', 'Planting date', 'Current stage']} />
      </DataSection>
      <GrowthDetails summary={null} advice={null}>
        <DataSection title="Earth observation">
          <EmptyValues labels={['NDVI', 'NDRE', 'Latest observation']} />
        </DataSection>
      </GrowthDetails>
    </div>
  );
}
