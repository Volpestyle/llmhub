import { ErrorKind, HubErrorPayload, Provider } from "./types.js";

export class LLMHubError extends Error {
  public readonly kind: ErrorKind;
  public readonly provider?: Provider;
  public readonly upstreamStatus?: number;
  public readonly upstreamCode?: string;
  public readonly requestId?: string;
  public readonly cause?: unknown;

  constructor(payload: HubErrorPayload) {
    super(payload.message);
    this.name = "LLMHubError";
    this.kind = payload.kind;
    this.provider = payload.provider;
    this.upstreamStatus = payload.upstreamStatus;
    this.upstreamCode = payload.upstreamCode;
    this.requestId = payload.requestId;
    this.cause = payload.cause;
  }
}

export function toHubError(err: unknown): LLMHubError {
  if (err instanceof LLMHubError) {
    return err;
  }
  const message =
    err instanceof Error ? err.message : "Unexpected provider error";
  return new LLMHubError({
    kind: ErrorKind.Unknown,
    message,
    cause: err,
  });
}

export function ensureHubError(
  provider: Provider,
  message: string,
  upstreamStatus?: number,
  upstreamCode?: string,
  requestId?: string,
): LLMHubError {
  return new LLMHubError({
    kind: classifyStatus(upstreamStatus),
    message,
    provider,
    upstreamStatus,
    upstreamCode,
    requestId,
  });
}

function classifyStatus(status?: number): ErrorKind {
  if (!status) {
    return ErrorKind.Unknown;
  }
  if (status === 401 || status === 403) {
    return ErrorKind.ProviderAuth;
  }
  if (status === 404) {
    return ErrorKind.ProviderNotFound;
  }
  if (status === 429) {
    return ErrorKind.ProviderRateLimit;
  }
  if (status >= 500 && status < 600) {
    return ErrorKind.ProviderUnavailable;
  }
  return ErrorKind.Unknown;
}
