import type { ConditionStatus } from './recommendation';
import type { AlertSummary, Urgency } from './alert';
import type { OperationSummary, IrrigationMode } from './operation';
import type { AreaUnit, Coordinates } from './plot';

export interface Farm {
  name: string;
  sizeM2: number;
  areaUnit: AreaUnit;
  timezone: string;
  gatewayId: string | null;
  configured: boolean | null;
  owner: string | null;
  coordinates: Coordinates | null;
}
export interface FarmInput extends Coordinates {
  name: string;
  sizeM2: number;
  areaUnit: AreaUnit;
  timezone: string;
}
export interface FarmPlotCard {
  plotId: string;
  name: string;
  cropType: string | null;
  growthStage: string | null;
  tensionCbar: number | null;
  thresholdCbar: number | null;
  thresholdMarginCbar: number | null;
  conditionStatus: ConditionStatus;
  nextAction: string;
  nextActionTime: string | null;
  urgency: Urgency;
  dataState: 'fresh' | 'stale' | 'unavailable';
  setupState?: 'settings_required' | 'unplanted' | 'complete';
  dataSources?: string[];
  dataReason?: string | null;
  equipment: {
    configured: boolean;
    state: 'active' | 'not_configured' | 'advisory_only' | 'ready';
    mode: IrrigationMode | null;
  };
  alert: { id: string; urgency: Urgency } | null;
  operation: {
    id: string;
    status: OperationSummary['status'];
    amountM3: number | null;
  } | null;
}
export interface FarmDashboard {
  schemaVersion: string;
  generatedAt: string;
  farm: Farm;
  summary: {
    totalPlots: number;
    plotsNeedingAttention: number;
    plannedWaterM3: number;
    equipmentConfigured: number;
    equipmentActive: number;
    staleOrUnavailablePlots: number;
    alertCounts: { watch: number; advise: number; critical: number };
  };
  plots: FarmPlotCard[];
  activeAlerts: AlertSummary[];
  todaysPlan: OperationSummary[];
}
