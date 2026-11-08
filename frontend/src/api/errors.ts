import type { FieldErrors } from '../domain/api';
export type ApiErrorKind =
  'configuration' | 'network' | 'http' | 'contract' | 'validation' | 'context';
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly endpoint: string | null;
  readonly fieldErrors: FieldErrors;
  readonly payload: unknown;
  constructor(
    message: string,
    options: {
      kind: ApiErrorKind;
      status?: number;
      endpoint?: string;
      fieldErrors?: FieldErrors;
      payload?: unknown;
      cause?: unknown;
    },
  ) {
    super(message, { cause: options.cause });
    this.name = 'ApiError';
    this.kind = options.kind;
    this.status = options.status ?? null;
    this.endpoint = options.endpoint ?? null;
    this.fieldErrors = options.fieldErrors ?? {};
    this.payload = options.payload;
  }
}
export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}
