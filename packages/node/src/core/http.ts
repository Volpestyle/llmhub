import { ErrorKind, FetchLike, Provider } from "./types.js";
import { InferenceKitError } from "./errors.js";

export interface HttpRequestOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: unknown;
  signal?: AbortSignal;
  provider: Provider;
  fetchImpl?: FetchLike;
}

export async function httpRequest<T = unknown>(
  url: string,
  options: HttpRequestOptions,
): Promise<{ data: T; response: Response }> {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new InferenceKitError({
      kind: ErrorKind.Unsupported,
      message: "global fetch is not available; supply kitConfig.httpClient",
      provider: options.provider,
    });
  }

  const response = await fetchImpl(url, {
    method: options.method ?? "GET",
    headers: {
      "content-type": "application/json",
      ...options.headers,
    },
    body:
      options.body === undefined
        ? undefined
        : typeof options.body === "string"
          ? options.body
          : JSON.stringify(options.body),
    signal: options.signal,
  });

  if (!response.ok) {
    const text = await safeReadText(response);
    throw new InferenceKitError({
      kind: classifyStatus(response.status),
      message: text || `Request failed with status ${response.status}`,
      provider: options.provider,
      upstreamStatus: response.status,
      upstreamCode: extractErrorCode(text),
      requestId: response.headers.get("x-request-id") ?? undefined,
    });
  }

  const data = (await response.json()) as T;
  return { data, response };
}

async function safeReadText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return "";
  }
}

function classifyStatus(status: number): ErrorKind {
  if (status === 401 || status === 403) {
    return ErrorKind.ProviderAuth;
  }
  if (status === 404) {
    return ErrorKind.ProviderNotFound;
  }
  if (status === 429) {
    return ErrorKind.ProviderRateLimit;
  }
  if (status >= 500) {
    return ErrorKind.ProviderUnavailable;
  }
  return ErrorKind.Unknown;
}

function extractErrorCode(body: string): string | undefined {
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.error === "string") {
      return parsed.error;
    }
    if (parsed?.error?.code) {
      return parsed.error.code;
    }
  } catch {
    // ignore
  }
  return undefined;
}
