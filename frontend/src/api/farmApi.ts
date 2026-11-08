import { conditionStatuses } from '../domain/recommendation';
import type { RequestOptions } from '../domain/api';
import type { FarmDashboard, FarmInput, FarmPlotCard } from '../domain/farm';
import { irrigationModes, operationStatuses } from '../domain/operation';
import { urgencies } from '../domain/alert';
import { apiClient } from './client';
import {
  checkedCount,
  farmForm,
  parseAlert,
  parseContext,
  parseFarm,
  parseOperation,
  parsePlot,
} from './adapters';
import {
  array,
  boolean,
  id,
  member,
  nonnegative,
  nullable,
  number,
  object,
  string,
  timestamp,
} from './validation';
function parseCard(value: unknown): FarmPlotCard {
  const d = object(value);
  const equipment = object(d.equipment);
  const alert = d.alert == null ? null : object(d.alert);
  const operation = d.operation == null ? null : object(d.operation);
  return {
    plotId: id(d.plot_id),
    name: string(d.name),
    cropType: d.crop_type === '' ? null : nullable(d.crop_type, string),
    growthStage: nullable(d.growth_stage, string),
    tensionCbar: nullable(d.current_tension_cbar, number),
    thresholdCbar: nullable(d.threshold_cbar, number),
    thresholdMarginCbar: nullable(d.threshold_margin_cbar, number),
    conditionStatus: member(d.condition_status, conditionStatuses),
    nextAction: string(d.next_action),
    nextActionTime: nullable(d.next_action_time, string),
    urgency: member(d.urgency, urgencies),
    dataState: member(d.data_state, ['fresh', 'stale', 'unavailable']),
    setupState:
      d.setup_state == null
        ? 'complete'
        : member(d.setup_state, ['settings_required', 'unplanted', 'complete']),
    dataSources: d.data_sources == null ? [] : array(d.data_sources, string),
    dataReason: nullable(d.data_reason, string),
    equipment: {
      configured: boolean(equipment.configured),
      state: member(equipment.state, [
        'active',
        'not_configured',
        'advisory_only',
        'ready',
      ]),
      mode: nullable(equipment.mode, (v) => member(v, irrigationModes)),
    },
    alert: alert
      ? { id: id(alert.alert_id), urgency: member(alert.urgency, urgencies) }
      : null,
    operation: operation
      ? {
          id: id(operation.operation_id),
          status: member(operation.status, operationStatuses),
          amountM3: nullable(operation.amount_m3, nonnegative),
        }
      : null,
  };
}
export function parseFarmDashboard(value: unknown): FarmDashboard {
  const d = object(value);
  const summary = object(d.summary);
  const counts = object(summary.alert_counts);
  return {
    schemaVersion: member(d.schema_version, ['1.0']),
    generatedAt: timestamp(d.generated_at),
    farm: parseFarm(d.farm),
    summary: {
      totalPlots: checkedCount(summary.total_plots),
      plotsNeedingAttention: checkedCount(summary.plots_needing_attention),
      plannedWaterM3: nonnegative(summary.planned_water_m3),
      equipmentConfigured: checkedCount(summary.equipment_configured),
      equipmentActive: checkedCount(summary.equipment_active),
      staleOrUnavailablePlots: checkedCount(
        summary.plots_with_stale_or_unavailable_data,
      ),
      alertCounts: {
        watch: checkedCount(counts.watch),
        advise: checkedCount(counts.advise),
        critical: checkedCount(counts.critical),
      },
    },
    plots: array(d.plots, parseCard),
    activeAlerts: array(d.active_alerts, parseAlert),
    todaysPlan: array(d.todays_plan, parseOperation),
  };
}
export const farmApi = {
  getRegistry: (options?: RequestOptions) =>
    apiClient.json('getFarmRegistry', parseContext, options),
  getFarmDashboard: (options?: RequestOptions) =>
    apiClient.json('farmDashboard', parseFarmDashboard, options),
  createFarm(
    input: FarmInput & { plotName: string },
    options?: RequestOptions,
  ) {
    const form = farmForm(input);
    form.set('plot_name', input.plotName);
    return apiClient.json(
      'farms',
      (value) => {
        const d = object(value);
        return { farm: parseFarm(d.farm), plot: parsePlot(d.plot) };
      },
      { ...options, method: 'POST', form },
    );
  },
  updateFarm(input: FarmInput, options?: RequestOptions) {
    const form = farmForm(input);
    form.delete('size'); // Boundary-derived area cannot be edited as metadata.
    return apiClient.json(
      'updateFarm',
      (value) => parseFarm(object(value).farm),
      { ...options, method: 'POST', form },
    );
  },
};
