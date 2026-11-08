// Shared decoders for records returned by multiple existing endpoints.
import type { Farm, FarmInput } from '../domain/farm';
import type { Plot, PlotContext } from '../domain/plot';
import {
  areaUnits,
  fromSquareMetres,
  isValidCoordinates,
  toSquareMetres,
} from '../domain/plot';
import type { AlertSummary } from '../domain/alert';
import { urgencies } from '../domain/alert';
import type { OperationSummary } from '../domain/operation';
import { irrigationModes, operationStatuses } from '../domain/operation';
import {
  array,
  boolean,
  id,
  integer,
  invalid,
  jsonObject,
  member,
  nonnegative,
  nullable,
  number,
  object,
  string,
} from './validation';
export function parseFarm(value: unknown): Farm {
  const data = object(value, 'farm');
  const areaUnit = member(data.area_unit, areaUnits, 'farm.area_unit');
  const coordinates =
    data.latitude == null || data.longitude == null
      ? null
      : { latitude: number(data.latitude), longitude: number(data.longitude) };
  if (coordinates && !isValidCoordinates(coordinates))
    invalid('farm.coordinates', 'valid coordinates');
  return {
    name: string(data.name),
    sizeM2: toSquareMetres(nonnegative(data.size), areaUnit),
    areaUnit,
    timezone: string(data.timezone),
    gatewayId: nullable(data.gateway_id, string),
    configured: nullable(data.configured, boolean),
    owner: nullable(data.owner, string),
    coordinates,
  };
}
export function farmForm(input: FarmInput): URLSearchParams {
  if (!input.name.trim() || !isValidCoordinates(input))
    invalid('farm input', 'name and valid coordinates');
  return new URLSearchParams({
    name: input.name,
    size: String(fromSquareMetres(input.sizeM2, input.areaUnit)),
    area_unit: input.areaUnit,
    latitude: String(input.latitude),
    longitude: String(input.longitude),
    timezone: input.timezone,
  });
}
export function parsePlot(value: unknown): Plot {
  const data = object(value, 'plot');
  const reference =
    data.reference_location == null ? null : object(data.reference_location);
  const coordinates = reference
    ? {
        latitude: number(reference.latitude),
        longitude: number(reference.longitude),
      }
    : null;
  if (coordinates && !isValidCoordinates(coordinates))
    invalid('plot.reference_location', 'valid coordinates');
  return {
    id: id(data.plot_id),
    name: string(data.name),
    cropType: nullable(data.crop_type, string),
    coordinates,
    areaM2: nonnegative(data.area),
    areaUnit: member(data.area_unit, areaUnits),
  };
}
export function parseContext(value: unknown): PlotContext {
  const data = object(value);
  const plots = array(data.plots, parsePlot, 'plots');
  const currentPlotId = id(data.current_plot_id);
  if (
    new Set(plots.map((plot) => plot.id)).size !== plots.length ||
    !plots.some((plot) => plot.id === currentPlotId)
  )
    invalid('plots', 'unique plot IDs containing current_plot_id');
  return { farm: parseFarm(data.farm), plots, currentPlotId };
}
export function parseAlert(value: unknown): AlertSummary {
  const data = object(value, 'alert');
  return {
    id: id(data.alert_id),
    plotId: id(data.plot_id),
    urgency: member(data.urgency, urgencies),
    payload: jsonObject(data.payload),
  };
}
export function parseOperation(value: unknown): OperationSummary {
  const data = object(value, 'operation');
  return {
    id: id(data.operation_id),
    source: nullable(data.source, string),
    recommendationUrgency:
      data.recommendation == null
        ? null
        : nullable(object(data.recommendation).urgency, (value) =>
            member(value, urgencies),
          ),
    mode: nullable(data.mode, (value) => member(value, irrigationModes)),
    plotId: id(data.plot_id),
    status: member(data.status, operationStatuses),
    amountM3: nullable(data.amount_m3, nonnegative),
    plannedStart: nullable(data.planned_start, string),
    plannedEnd: nullable(data.planned_end, string),
  };
}
export function checkedCount(value: unknown): number {
  return integer(value, 'count');
}
