import type { JsonValue, RequestOptions } from '../domain/api';
import { ApiError, isAbortError } from './errors';
import { object } from './validation';
export interface HttpOptions extends RequestOptions {
  method?: 'GET' | 'POST';
  form?: URLSearchParams;
  jsonBody?: JsonValue;
  query?: Record<string, string>;
}
export interface ApiClient {
  json<T>(
    endpoint: string,
    decode: (value: unknown) => T,
    options?: HttpOptions,
  ): Promise<T>;
  text(endpoint: string, options?: HttpOptions): Promise<string>;
  empty(endpoint: string, options?: HttpOptions): Promise<void>;
}
export function parseJson(text: string, endpoint: string): unknown {
  try {
    return JSON.parse(text);
  } catch (cause) {
    throw new ApiError(`Expected JSON from ${endpoint}.`, {
      kind: 'contract',
      endpoint,
      cause,
    });
  }
}
export function resolveApiUrl(
  base: string,
  endpoint: string,
  pageUrl?: string,
): URL {
  if (!base.trim())
    throw new ApiError(
      'Configure VITE_API_BASE_URL before calling the backend.',
      { kind: 'configuration' },
    );
  // Resource modules pass endpoint names, never arbitrary URLs or traversal paths.
  if (!/^[A-Za-z][A-Za-z0-9]*$/.test(endpoint))
    throw new ApiError('Invalid endpoint name.', {
      kind: 'configuration',
      endpoint,
    });
  try {
    const root = new URL(base, pageUrl);
    if (
      !['http:', 'https:'].includes(root.protocol) ||
      root.username ||
      root.password ||
      root.search ||
      root.hash
    )
      throw new Error('Invalid API base');
    root.pathname = root.pathname.replace(/\/$/, '') + '/';
    return new URL(endpoint, root);
  } catch (cause) {
    throw new ApiError(
      'API base must resolve to an HTTP(S) directory without query or hash.',
      { kind: 'configuration', cause },
    );
  }
}
function httpError(
  text: string,
  response: Response,
  endpoint: string,
): ApiError {
  let payload: unknown = text;
  try {
    payload = JSON.parse(text);
  } catch {
    /* HTML/plain-text proxy failures are valid error bodies. */
  }
  let message = `API request failed (${response.status}) at ${endpoint}.`;
  const fieldErrors: Record<string, string> = {};
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    const body = object(payload);
    const detail = body.message ?? body.error ?? body.reason;
    if (typeof detail === 'string') message = detail;
    if (
      Array.isArray(body.missing) &&
      body.missing.every((entry) => typeof entry === 'string')
    )
      message += ` Missing: ${body.missing.join(', ')}.`;
    if (
      (body.errors ?? body.fields) &&
      typeof (body.errors ?? body.fields) === 'object' &&
      !Array.isArray(body.errors ?? body.fields)
    ) {
      for (const [key, value] of Object.entries(
        object(body.errors ?? body.fields),
      ))
        if (typeof value === 'string') fieldErrors[key] = value;
    }
  } else if (
    typeof payload === 'string' &&
    payload.trim() &&
    !payload.trim().startsWith('<')
  )
    message += ` ${payload.slice(0, 500)}`;
  return new ApiError(message, {
    kind: 'http',
    status: response.status,
    endpoint,
    payload,
    fieldErrors,
  });
}
export function createApiClient(
  config: { baseUrl?: string; pageUrl?: string; fetch?: typeof fetch } = {},
): ApiClient {
  async function request(
    endpoint: string,
    options: HttpOptions = {},
  ): Promise<{ text: string; status: number }> {
    const url = resolveApiUrl(
      config.baseUrl ?? import.meta.env.VITE_API_BASE_URL ?? '',
      endpoint,
      config.pageUrl ??
        (typeof document === 'undefined' ? undefined : document.baseURI),
    );
    for (const [key, value] of Object.entries(options.query ?? {}))
      url.searchParams.set(key, value);
    if (options.form && options.method !== 'POST')
      throw new ApiError('Form requests require POST.', {
        kind: 'configuration',
        endpoint,
      });
    if (
      options.jsonBody !== undefined &&
      (options.method !== 'POST' || options.form)
    )
      throw new ApiError(
        'JSON requests require POST and cannot include a form.',
        {
          kind: 'configuration',
          endpoint,
        },
      );
    let response: Response;
    let text: string;
    try {
      response = await (config.fetch ?? globalThis.fetch)(url, {
        method: options.method ?? 'GET',
        headers: {
          Accept: 'application/json',
          ...(options.form
            ? { 'Content-Type': 'application/x-www-form-urlencoded' }
            : {}),
          ...(options.jsonBody !== undefined
            ? { 'Content-Type': 'application/json' }
            : {}),
        },
        credentials: 'same-origin',
        ...(options.signal ? { signal: options.signal } : {}),
        ...(options.form ? { body: options.form } : {}),
        ...(options.jsonBody !== undefined
          ? { body: JSON.stringify(options.jsonBody) }
          : {}),
      });
      text = await response.text();
    } catch (cause) {
      if (isAbortError(cause) || options.signal?.aborted) throw cause;
      throw new ApiError(`Unable to reach ${endpoint}.`, {
        kind: 'network',
        endpoint,
        cause,
      });
    }
    if (!response.ok) throw httpError(text, response, endpoint);
    return { text, status: response.status };
  }
  return {
    async json(endpoint, decode, options) {
      const result = await request(endpoint, options);
      if (result.status === 204 || !result.text.trim())
        throw new ApiError(`Expected a JSON body from ${endpoint}.`, {
          kind: 'contract',
          endpoint,
        });
      try {
        return decode(parseJson(result.text, endpoint));
      } catch (cause) {
        if (cause instanceof ApiError && cause.kind === 'contract')
          throw new ApiError(cause.message, {
            kind: 'contract',
            endpoint,
            status: result.status,
            cause,
          });
        throw cause;
      }
    },
    async text(endpoint, options) {
      return (await request(endpoint, options)).text;
    },
    async empty(endpoint, options) {
      const result = await request(endpoint, options);
      if (result.text.trim())
        throw new ApiError(`Expected an empty response from ${endpoint}.`, {
          kind: 'contract',
          endpoint,
        });
    },
  };
}
export const apiClient = createApiClient();

// Only endpoints with a documented {available:false} 404 opt into this path.
// A missing route returning HTML or an unrelated 404 remains an HTTP error.
export async function readWithUnavailable<T>(
  endpoint: string,
  decode: (value: unknown) => T,
  options?: HttpOptions,
): Promise<T> {
  try {
    return await apiClient.json(endpoint, decode, options);
  } catch (error) {
    if (
      error instanceof ApiError &&
      error.status === 404 &&
      error.payload !== null &&
      typeof error.payload === 'object' &&
      !Array.isArray(error.payload) &&
      object(error.payload).available === false
    )
      return decode(error.payload);
    throw error;
  }
}
