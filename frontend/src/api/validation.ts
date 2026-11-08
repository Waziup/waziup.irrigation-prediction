import type { JsonObject, JsonValue, Unavailable } from '../domain/api';
import { ApiError } from './errors';
export function invalid(path: string, expected: string): never {
  throw new ApiError(`Invalid API response at ${path}: expected ${expected}.`, {
    kind: 'contract',
  });
}
export function object(
  value: unknown,
  path = 'response',
): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    return invalid(path, 'an object');
  return value as Record<string, unknown>;
}
export function string(value: unknown, path = 'value'): string {
  if (typeof value !== 'string') return invalid(path, 'a string');
  return value;
}
export function id(value: unknown, path = 'id'): string {
  const result = string(value, path);
  if (!result.trim()) return invalid(path, 'a nonempty stable ID');
  return result;
}
export function number(value: unknown, path = 'value'): number {
  if (typeof value !== 'number' || !Number.isFinite(value))
    return invalid(path, 'a finite number');
  return value;
}
export function nonnegative(value: unknown, path = 'value'): number {
  const result = number(value, path);
  return result >= 0 ? result : invalid(path, 'a nonnegative number');
}
export function integer(value: unknown, path = 'value'): number {
  const result = nonnegative(value, path);
  return Number.isInteger(result)
    ? result
    : invalid(path, 'a nonnegative integer');
}
export function boolean(value: unknown, path = 'value'): boolean {
  if (typeof value !== 'boolean') return invalid(path, 'a boolean');
  return value;
}
export function nullable<T>(
  value: unknown,
  parse: (value: unknown) => T,
): T | null {
  return value === null || value === undefined ? null : parse(value);
}
export function array<T>(
  value: unknown,
  parse: (value: unknown) => T,
  path = 'value',
): T[] {
  if (!Array.isArray(value)) return invalid(path, 'an array');
  return value.map(parse);
}
export function member<const T extends readonly string[]>(
  value: unknown,
  values: T,
  path = 'status',
): T[number] {
  const result = string(value, path);
  return values.includes(result)
    ? (result as T[number])
    : invalid(path, values.join(' | '));
}
export function timestamp(value: unknown): string {
  const result = string(value, 'timestamp');
  return result.trim() && Number.isFinite(Date.parse(result))
    ? result
    : invalid('timestamp', 'a valid date/time');
}
export function jsonValue(value: unknown): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string')
    return value;
  if (typeof value === 'number') return number(value);
  if (Array.isArray(value)) return value.map(jsonValue);
  return jsonObject(value);
}
export function jsonObject(value: unknown): JsonObject {
  return Object.fromEntries(
    Object.entries(object(value)).map(([key, entry]) => [
      key,
      jsonValue(entry),
    ]),
  );
}
export function unavailable(
  value: Record<string, unknown>,
): Unavailable | null {
  if (value.available !== false) return null;
  return {
    available: false,
    reason: nullable(value.reason ?? value.error, string),
  };
}
// A few legacy measurements are numeric strings. Null/blank mean missing,
// but booleans and malformed strings must never become measured zero.
export function measurement(value: unknown): number | null {
  if (
    value === null ||
    value === undefined ||
    (typeof value === 'string' && !value.trim())
  )
    return null;
  if (typeof value === 'string') return number(Number(value), 'measurement');
  return number(value, 'measurement');
}
