import { Kit } from "../core/types.js";
import {
  GenerateInput,
  ImageGenerateInput,
  ListModelsParams,
  MeshGenerateInput,
  Provider,
  TranscribeInput,
} from "../core/types.js";
import { toKitError, AiKitError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";

export interface RequestLike {
  method?: string;
  query?: Record<string, string | string[] | undefined>;
  body?: GenerateInput | ImageGenerateInput | MeshGenerateInput | TranscribeInput | string | null;
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

export function httpHandlers(kit: Kit) {
  return {
    models:
      (opts?: ModelsHandlerOptions) =>
      async (req: RequestLike, res: ResponseLike) => {
        try {
          const params = buildListModelsParams(req);
          const models = await kit.listModels(params);
          res.status(200).json(models);
        } catch (err) {
          sendJsonError(res, err);
        }
      },
    generate:
      () => async (req: RequestLike, res: ResponseLike) => {
        try {
          const input = normalizeGenerateInput(req.body ?? null);
          const output = await kit.generate({ ...input, stream: false });
          res.status(200).json(output);
        } catch (err) {
          sendJsonError(res, err);
        }
      },
    image:
      () => async (req: RequestLike, res: ResponseLike) => {
        try {
          const input = normalizeImageInput(req.body ?? null);
          const output = await kit.generateImage(input);
          res.status(200).json(output);
        } catch (err) {
          sendJsonError(res, err);
        }
      },
    mesh:
      () => async (req: RequestLike, res: ResponseLike) => {
        try {
          const input = normalizeMeshInput(req.body ?? null);
          const output = await kit.generateMesh(input);
          res.status(200).json(output);
        } catch (err) {
          sendJsonError(res, err);
        }
      },
    transcribe:
      () => async (req: RequestLike, res: ResponseLike) => {
        try {
          const input = normalizeTranscribeInput(req.body ?? null);
          const output = await kit.transcribe(input);
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
          const iterable = kit.streamGenerate({ ...input, stream: true });
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
    case Provider.Bedrock:
      return Provider.Bedrock;
    case Provider.XAI:
      return Provider.XAI;
    case Provider.Ollama:
      return Provider.Ollama;
    case Provider.Local:
      return Provider.Local;
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
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: "Body must be valid JSON",
      });
    }
  }
  if (!parsed || typeof parsed !== "object") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "Request body must be a GenerateInput object",
    });
  }
  const input = parsed as Record<string, unknown>;
  if (typeof input.provider !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "provider is required and must be a string",
    });
  }
  if (typeof input.model !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "model is required and must be a string",
    });
  }
  if (!Array.isArray(input.messages)) {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "messages is required and must be an array",
    });
  }
  return input as unknown as GenerateInput;
}

function normalizeImageInput(payload: unknown): ImageGenerateInput {
  let parsed: ImageGenerateInput | string | null | Record<string, unknown> = payload as
    | ImageGenerateInput
    | string
    | null
    | Record<string, unknown>;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: "Body must be valid JSON",
      });
    }
  }
  if (!parsed || typeof parsed !== "object") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "Request body must be an ImageGenerateInput object",
    });
  }
  const input = parsed as Record<string, unknown>;
  if (typeof input.provider !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "provider is required and must be a string",
    });
  }
  if (typeof input.model !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "model is required and must be a string",
    });
  }
  if (typeof input.prompt !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "prompt is required and must be a string",
    });
  }
  return input as unknown as ImageGenerateInput;
}

function normalizeMeshInput(payload: unknown): MeshGenerateInput {
  let parsed: MeshGenerateInput | string | null | Record<string, unknown> = payload as
    | MeshGenerateInput
    | string
    | null
    | Record<string, unknown>;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: "Body must be valid JSON",
      });
    }
  }
  if (!parsed || typeof parsed !== "object") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "Request body must be a MeshGenerateInput object",
    });
  }
  const input = parsed as Record<string, unknown>;
  if (typeof input.provider !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "provider is required and must be a string",
    });
  }
  if (typeof input.model !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "model is required and must be a string",
    });
  }
  if (typeof input.prompt !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "prompt is required and must be a string",
    });
  }
  return input as unknown as MeshGenerateInput;
}

function normalizeTranscribeInput(payload: unknown): TranscribeInput {
  let parsed: TranscribeInput | string | null | Record<string, unknown> = payload as
    | TranscribeInput
    | string
    | null
    | Record<string, unknown>;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: "Body must be valid JSON",
      });
    }
  }
  if (!parsed || typeof parsed !== "object") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "Request body must be a TranscribeInput object",
    });
  }
  const input = parsed as Record<string, unknown>;
  if (
    typeof input.responseFormat !== "string" &&
    typeof input.response_format === "string"
  ) {
    input.responseFormat = input.response_format;
  }
  if (
    !Array.isArray(input.timestampGranularities) &&
    input.timestamp_granularities !== undefined
  ) {
    input.timestampGranularities =
      typeof input.timestamp_granularities === "string"
        ? [input.timestamp_granularities]
        : input.timestamp_granularities;
  }
  if (typeof input.provider !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "provider is required and must be a string",
    });
  }
  if (typeof input.model !== "string") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "model is required and must be a string",
    });
  }
  if (!input.audio || typeof input.audio !== "object") {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "audio is required and must be an object",
    });
  }
  return input as unknown as TranscribeInput;
}

function parseQueryPayload(req: RequestLike): unknown {
  const raw = req.query?.payload;
  if (!raw) {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "stream endpoint expects ?payload=<json>",
    });
  }
  try {
    return JSON.parse(String(raw));
  } catch {
    throw new AiKitError({
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
  const kitErr = toKitError(err);
  res
    .status(mapStatus(kitErr))
    .json({ error: { kind: kitErr.kind, message: kitErr.message } });
}

function sendSSEError(res: ResponseLike, err: unknown) {
  const kitErr = toKitError(err);
  prepareSSE(res);
  writeSSE(res, "error", {
    kind: kitErr.kind,
    message: kitErr.message,
    requestId: kitErr.requestId,
  });
  res.end?.();
}

function mapStatus(err: AiKitError): number {
  switch (err.kind) {
    case ErrorKind.Validation:
      return 400;
    case ErrorKind.Unsupported:
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
