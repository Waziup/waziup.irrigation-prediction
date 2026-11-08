export type JsonValue =
  null | boolean | number | string | JsonValue[] | JsonObject;
export interface JsonObject {
  [key: string]: JsonValue;
}
export interface RequestOptions {
  signal?: AbortSignal;
}
export interface Unavailable {
  available: false;
  reason: string | null;
}
export type FieldErrors = Record<string, string>;
