import { Hub } from "../core/types.js";
import { GenerateInput, ListModelsParams, Provider } from "../core/types.js";
import { toHubError, LLMHubError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";

export interface RequestLike {
  method?: string;
  query?: Record<string, string | string[] | undefined>;
  body?: GenerateInput | string | null;
  headers?: Record<string, string>;
  get?(name: string): string | undefined;
}

export interface ResponseLike {
  status(code: number): ResponseLike;
  json(payload: unknown): void;
  setHeader?(name: string, value: string): void;
  write?(chunk: string): void;
  end?(chunk?: string): void;
  flush?(): void;
}

export interface ModelsHandlerOptions {
  ttlMs?: number;
}

export interface GenerateHandlerOptions {
  stream?: boolean;
}

export function httpHandlers(hub: Hub) {
  return {
    models:
      (opts?: ModelsHandlerOptions) =>
      async (req: RequestLike, res: ResponseLike) => {
        try {
          const params = buildListModelsParams(req);
          const models = await hub.listModels(params);
          res.status(200).json(models);
        } catch (err) {
          sendJsonError(res, err);
        }
      },
    generate:
      () => async (req: RequestLike, res: ResponseLike) => {
        try {
          const input = normalizeGenerateInput(req.body ?? null);
          const output = await hub.generate({ ...input, stream: false });
          res.status(200).json(output);
        } catch (err) {
          sendJsonError(res, err);
        }
      },
    generateSSE:
      () => async (req: RequestLike, res: ResponseLike) => {
        try {
          const input = normalizeGenerateInput(
            req.method === "GET" ? parseQueryPayload(req) : req.body ?? null,
          );
          const iterable = hub.streamGenerate({ ...input, stream: true });
          prepareSSE(res);
          for await (const chunk of iterable) {
            writeSSE(res, "chunk", chunk);
            res.flush?.();
          }
          writeSSE(res, "done", { ok: true });
          res.end?.();
        } catch (err) {
          sendSSEError(res, err);
        }
      },
  };
}

function buildListModelsParams(req: RequestLike): ListModelsParams | undefined {
  const params: ListModelsParams = {};
  const providers = parseProviders(req.query?.providers);
  if (providers) {
    params.providers = providers;
  }
  if (shouldRefresh(req.query?.refresh)) {
    params.refresh = true;
  }
  return Object.keys(params).length ? params : undefined;
}

function parseProviders(
  input: unknown,
): Provider[] | undefined {
  const raw = normalizeQueryValue(input);
  if (!raw) {
    return undefined;
  }
  const providers = raw
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => toProvider(value))
    .filter((provider): provider is Provider => Boolean(provider));
  return providers.length ? providers : undefined;
}

function shouldRefresh(input: unknown): boolean {
  const raw = normalizeQueryValue(input);
  if (raw === undefined) {
    return false;
  }
  const normalized = raw.toLowerCase();
  return (
    normalized === "" ||
    normalized === "1" ||
    normalized === "true" ||
    normalized === "yes" ||
    normalized === "on"
  );
}

function normalizeQueryValue(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (Array.isArray(value)) {
    return value.length ? String(value[0]) : undefined;
  }
  return String(value);
}

function toProvider(value: string): Provider | undefined {
  switch (value.toLowerCase()) {
    case Provider.OpenAI:
      return Provider.OpenAI;
    case Provider.Anthropic:
      return Provider.Anthropic;
    case Provider.Google:
      return Provider.Google;
    case Provider.XAI:
      return Provider.XAI;
    default:
      return undefined;
  }
}

function normalizeGenerateInput(payload: unknown): GenerateInput {
  let parsed: GenerateInput | string | null | Record<string, unknown> = payload as
    | GenerateInput
    | string
    | null
    | Record<string, unknown>;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      throw new LLMHubError({
        kind: ErrorKind.Validation,
        message: "Body must be valid JSON",
      });
    }
  }
  if (!parsed || typeof parsed !== "object") {
    throw new LLMHubError({
      kind: ErrorKind.Validation,
      message: "Request body must be a GenerateInput object",
    });
  }
  const input = parsed as Record<string, unknown>;
  if (typeof input.provider !== "string") {
    throw new LLMHubError({
      kind: ErrorKind.Validation,
      message: "provider is required and must be a string",
    });
  }
  if (typeof input.model !== "string") {
    throw new LLMHubError({
      kind: ErrorKind.Validation,
      message: "model is required and must be a string",
    });
  }
  if (!Array.isArray(input.messages)) {
    throw new LLMHubError({
      kind: ErrorKind.Validation,
      message: "messages is required and must be an array",
    });
  }
  return input as unknown as GenerateInput;
}

function parseQueryPayload(req: RequestLike): unknown {
  const raw = req.query?.payload;
  if (!raw) {
    throw new LLMHubError({
      kind: ErrorKind.Validation,
      message: "stream endpoint expects ?payload=<json>",
    });
  }
  try {
    return JSON.parse(String(raw));
  } catch {
    throw new LLMHubError({
      kind: ErrorKind.Validation,
      message: "Invalid JSON payload in query parameter",
    });
  }
}

function prepareSSE(res: ResponseLike) {
  res.setHeader?.("Content-Type", "text/event-stream");
  res.setHeader?.("Cache-Control", "no-cache, no-transform");
  res.setHeader?.("Connection", "keep-alive");
  res.status(200);
}

function writeSSE(res: ResponseLike, event: string, data: unknown) {
  res.write?.(`event: ${event}\n`);
  res.write?.(`data: ${JSON.stringify(data)}\n\n`);
}

function sendJsonError(res: ResponseLike, err: unknown) {
  const hubErr = toHubError(err);
  res
    .status(mapStatus(hubErr))
    .json({ error: { kind: hubErr.kind, message: hubErr.message } });
}

function sendSSEError(res: ResponseLike, err: unknown) {
  const hubErr = toHubError(err);
  prepareSSE(res);
  writeSSE(res, "error", {
    kind: hubErr.kind,
    message: hubErr.message,
    requestId: hubErr.requestId,
  });
  res.end?.();
}

function mapStatus(err: LLMHubError): number {
  switch (err.kind) {
    case ErrorKind.Validation:
      return 400;
    case ErrorKind.ProviderAuth:
      return 401;
    case ErrorKind.ProviderRateLimit:
      return 429;
    case ErrorKind.ProviderUnavailable:
      return 503;
    default:
      return 500;
  }
}
