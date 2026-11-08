import type { RequestOptions } from '../domain/api';
import { irrigationModes } from '../domain/operation';
import {
  areaUnits,
  fromSquareMetres,
  isValidCoordinates,
  thresholdModes,
} from '../domain/plot';
import type {
  PlotSettings,
  PlotSettingsInput,
  SaveSettingsResult,
  PlotSettingsSnapshot,
  PlotSettingsSaveResult,
} from '../domain/settings';
import {
  flowConfirmationModes,
  validatePlotSettings,
} from '../domain/settings';
import { apiClient, parseJson } from './client';
import { ApiError } from './errors';
import {
  array,
  boolean,
  id,
  integer,
  invalid,
  jsonObject,
  measurement,
  member,
  nonnegative,
  nullable,
  number,
  object,
  string,
} from './validation';
export function parsePlotSettings(value: unknown): PlotSettings {
  const envelope = object(value);
  if (envelope.status_code !== 200) invalid('status_code', '200');
  const d = object(envelope.data);
  const gps = object(d.Gps_info);
  const readiness = object(d.Training_readiness);
  const coordinates = {
    latitude: number(gps.latitude ?? gps.lattitude),
    longitude: number(gps.longitude),
  };
  if (!isValidCoordinates(coordinates))
    invalid('Gps_info', 'valid coordinates');
  return {
    plotId: id(d.Plot_id),
    name: string(d.Name),
    zoneName: string(d.Zone_name),
    coordinates,
    areaM2: nonnegative(d.Plot_area_m2),
    areaUnit: member(d.Plot_area_unit, areaUnits),
    sensorKind: string(d.Sensor_kind),
    moistureSensorIds: array(d.DeviceAndSensorIdsMoisture, string),
    temperatureSensorIds: array(d.DeviceAndSensorIdsTemp, string),
    actuatorIds: array(d.DeviceAndSensorIdsFlow, string),
    flowConfirmationSensorIds: array(
      d.DeviceAndSensorIdsFlowConfirmation,
      string,
    ),
    flowConfirmationMode: member(
      d.Flow_confirmation_mode,
      flowConfirmationModes,
    ),
    slope: number(d.Slope),
    thresholdCbar: number(d.Threshold),
    thresholdMode: member(d.Threshold_mode, thresholdModes),
    irrigationType: string(d.Irrigation_type),
    irrigationMode: member(d.Irrigation_mode, irrigationModes),
    applicationEfficiency: number(d.Application_efficiency),
    effectiveRainfallFraction: number(d.Effective_rainfall_fraction),
    lookaheadHours: nonnegative(d.Look_ahead_time),
    startDate: string(d.Start_date),
    period: integer(d.Period),
    soilType: string(d.Soil_type),
    soilTextureClass: nullable(d.Soil_texture_class, integer),
    soilCalibration: jsonObject(d.Soil_calibration),
    permanentWiltingPoint: number(d.PermanentWiltingPoint),
    fieldCapacityUpper: number(d.FieldCapacityUpper),
    fieldCapacityLower: number(d.FieldCapacityLower),
    saturation: number(d.Saturation),
    retentionCurve: array(d.Soil_water_retention_curve, (entry) => {
      const point = object(entry);
      // CSV-backed saved rows contain numeric strings; preserve VWC's existing scale.
      const tensionCbar = measurement(point['Soil tension']);
      const vwc = measurement(point.VWC);
      if (tensionCbar === null || vwc === null)
        invalid('retentionCurve', 'numeric tension and VWC');
      return { tensionCbar, vwc };
    }),
    cropType: string(d.Crop_type),
    plantingDate: string(d.Planting_date),
    initialGdd: number(d.Initial_gdd),
    trainingReadiness: {
      ready: boolean(readiness.ready),
      missing: array(readiness.missing, string),
      dataSource: string(readiness.data_source),
    },
  };
}
export function settingsForm(input: PlotSettingsInput): URLSearchParams {
  const fieldErrors = validatePlotSettings(input);
  if (Object.keys(fieldErrors).length)
    throw new ApiError('Plot settings contain invalid values.', {
      kind: 'validation',
      fieldErrors,
    });
  const form = new URLSearchParams({
    name: input.name,
    zone_name: input.zoneName,
    gps: `${input.coordinates.latitude}, ${input.coordinates.longitude}`,
    plot_area_m2: String(fromSquareMetres(input.areaM2, input.areaUnit)),
    area_unit: input.areaUnit,
    sensor_kind: input.sensorKind,
    slope: String(input.slope),
    thres: String(input.thresholdCbar),
    threshold_mode: input.thresholdMode,
    irrigation_type: input.irrigationType,
    irrigation_mode: input.irrigationMode,
    flow_confirmation_mode: input.flowConfirmationMode,
    application_efficiency: String(input.applicationEfficiency),
    effective_rainfall_fraction: String(input.effectiveRainfallFraction),
    lookahead: String(input.lookaheadHours),
    start: input.startDate,
    period: String(input.period),
    soil: input.soilType,
    pwp: String(input.permanentWiltingPoint),
    fcu: String(input.fieldCapacityUpper),
    fcl: String(input.fieldCapacityLower),
    sat: String(input.saturation),
    soil_texture_class:
      input.soilTextureClass === null ? '' : String(input.soilTextureClass),
    soil_calibration: JSON.stringify(input.soilCalibration),
    crop_type: input.cropType,
    planting_date: input.plantingDate,
    initial_gdd: String(input.initialGdd),
    ret:
      'Soil tension,VWC\n' +
      input.retentionCurve
        .map((point) => `${point.tensionCbar},${point.vwc}`)
        .join('\n') +
      '\n',
  });
  for (const [key, values] of [
    ['selectedOptionsMoisture', input.moistureSensorIds],
    ['selectedOptionsTemp', input.temperatureSensorIds],
    ['selectedOptionsFlow', input.actuatorIds],
    ['selectedOptionsFlowConfirmation', input.flowConfirmationSensorIds],
  ] as const)
    for (const value of values) form.append(key, value);
  return form;
}
export function parseSaveSettings(text: string): SaveSettingsResult {
  if (text.trim() === 'Configuration has been successfully saved!')
    return { status: 'saved', message: text.trim() };
  const d = object(parseJson(text, 'setConfig'));
  return {
    status:
      member(d.status, ['ok', 'pending']) === 'pending' ? 'pending' : 'saved',
    message: string(d.message),
  };
}
function resourceMetadata(value: unknown, plotId: string) {
  const d = object(value);
  const returnedId = id(d.plot_id);
  if (returnedId !== plotId) invalid('plot_id', 'the requested settings plot');
  const configVersion = string(d.config_version);
  if (!configVersion.trim())
    invalid('config_version', 'a nonempty version token');
  return { plotId: returnedId, configVersion };
}
export const settingsApi = {
  async getPlotSettings(
    plotId: string,
    options?: RequestOptions,
  ): Promise<PlotSettingsSnapshot> {
    const expectedId = id(plotId);
    try {
      return await apiClient.json(
        'returnConfig',
        (value) => {
          const metadata = resourceMetadata(value, expectedId);
          const settings = parsePlotSettings(value);
          if (settings.plotId !== expectedId)
            invalid('Plot_id', 'the requested settings plot');
          return { ...metadata, configured: true as const, settings };
        },
        { ...options, query: { plot_id: expectedId } },
      );
    } catch (error) {
      if (
        error instanceof ApiError &&
        error.status === 400 &&
        error.payload &&
        typeof error.payload === 'object' &&
        object(error.payload).error ===
          'No config data present. Perform configuration.'
      )
        return {
          ...resourceMetadata(error.payload, expectedId),
          configured: false,
        };
      throw error;
    }
  },
  async savePlotSettings(
    plotId: string,
    input: PlotSettingsInput,
    configVersion: string,
    options?: RequestOptions,
  ): Promise<PlotSettingsSaveResult> {
    const expectedId = id(plotId);
    if (!configVersion.trim())
      throw new ApiError(
        'Reload settings before saving: configuration version is missing.',
        { kind: 'validation' },
      );
    const form = settingsForm(input);
    form.set('plot_id', expectedId);
    form.set('config_version', configVersion);
    const text = await apiClient.text('setConfig', {
      ...options,
      method: 'POST',
      form,
    });
    const metadata = resourceMetadata(parseJson(text, 'setConfig'), expectedId);
    return { ...parseSaveSettings(text), ...metadata };
  },
};
