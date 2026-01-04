import { createHash, createHmac } from "crypto";
import { ProviderAdapter } from "../core/provider.js";
import { AiKitError } from "../core/errors.js";
import {
  BedrockProviderConfig,
  ErrorKind,
  FetchLike,
  GenerateInput,
  GenerateOutput,
  ModelMetadata,
  Provider,
  StreamChunk,
  ToolCall,
  ToolDefinition,
  Usage,
} from "../core/types.js";

interface BedrockModelSummary {
  modelId?: string;
  modelName?: string;
  providerName?: string;
  inputModalities?: string[];
  outputModalities?: string[];
}

interface BedrockListResponse {
  modelSummaries?: BedrockModelSummary[];
  models?: BedrockModelSummary[];
}

type BedrockContentBlock =
  | { text: string }
  | {
      image: {
        format: string;
        source: { bytes: string };
      };
    }
  | {
      toolUse: {
        toolUseId: string;
        name: string;
        input: unknown;
      };
    }
  | {
      toolResult: {
        toolUseId: string;
        content: Array<{ text: string }>;
      };
    };

interface BedrockMessage {
  role: "user" | "assistant";
  content: BedrockContentBlock[];
}

interface BedrockConverseResponse {
  output?: {
    message?: {
      role?: string;
      content?: BedrockContentBlock[];
    };
  };
  stopReason?: string;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    totalTokens?: number;
  };
}

interface AwsCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
}

interface ResolvedBedrockConfig {
  region: string;
  endpoint: string;
  runtimeEndpoint: string;
  controlPlaneService: string;
  runtimeService: string;
  credentials: AwsCredentials;
}

const ENV_REGION = process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? "";
const ENV_ACCESS_KEY =
  process.env.AWS_ACCESS_KEY_ID ?? process.env.AWS_ACCESS_KEY ?? "";
const ENV_SECRET_KEY =
  process.env.AWS_SECRET_ACCESS_KEY ?? process.env.AWS_SECRET_KEY ?? "";
const ENV_SESSION_TOKEN = process.env.AWS_SESSION_TOKEN ?? "";

export class BedrockAdapter implements ProviderAdapter {
  readonly provider = Provider.Bedrock;
  private readonly config: ResolvedBedrockConfig;
  private readonly fetchImpl?: FetchLike;

  constructor(config: BedrockProviderConfig, fetchImpl?: FetchLike) {
    this.fetchImpl = fetchImpl;
    this.config = resolveBedrockConfig(config);
  }

  async listModels(): Promise<ModelMetadata[]> {
    const url = `${this.config.endpoint}/foundation-models`;
    const { data } = await this.fetchSignedJSON<BedrockListResponse>(
      url,
      { method: "GET" },
      this.config.controlPlaneService,
    );
    const summaries = data.modelSummaries ?? data.models ?? [];
    return summaries
      .map((summary) => this.toModelMetadata(summary))
      .filter((model): model is ModelMetadata => Boolean(model));
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    if (input.responseFormat?.type === "json_schema") {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: "Bedrock structured outputs are not supported",
        provider: this.provider,
      });
    }
    const payload = await this.buildConversePayload(input);
    const url = `${this.config.runtimeEndpoint}/model/${encodeURIComponent(
      input.model,
    )}/converse`;
    const { data } = await this.fetchSignedJSON<BedrockConverseResponse>(
      url,
      {
        method: "POST",
        body: payload,
        signal: input.signal,
      },
      this.config.runtimeService,
    );
    return this.normalizeConverseResponse(data);
  }

  async *streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const output = await this.generate({ ...input, stream: false });
    if (output.text) {
      yield { type: "delta", textDelta: output.text };
    }
    if (output.toolCalls) {
      for (const call of output.toolCalls) {
        yield { type: "tool_call", call };
      }
    }
    yield {
      type: "message_end",
      usage: output.usage,
      finishReason: output.finishReason,
    };
  }

  private toModelMetadata(summary: BedrockModelSummary): ModelMetadata | null {
    const modelId = summary.modelId?.trim();
    if (!modelId) {
      return null;
    }
    const inputModalities = new Set(
      (summary.inputModalities ?? []).map((value) => value.toLowerCase()),
    );
    const outputModalities = new Set(
      (summary.outputModalities ?? []).map((value) => value.toLowerCase()),
    );
    return {
      id: modelId,
      displayName: summary.modelName ?? modelId,
      provider: this.provider,
      family: deriveFamily(modelId, summary.providerName),
      capabilities: {
        text: outputModalities.has("text"),
        vision: inputModalities.has("image"),
        tool_use: supportsToolUse(modelId),
        structured_output: false,
        reasoning: false,
      },
    };
  }

  private async buildConversePayload(input: GenerateInput) {
    const system: Array<{ text: string }> = [];
    const messages: BedrockMessage[] = [];
    for (const message of input.messages) {
      if (message.role === "system") {
        const text = extractText(message.content);
        if (text) {
          system.push({ text });
        }
        continue;
      }
      if (message.role === "tool") {
        const toolContent = message.content
          .filter((part) => part.type === "text")
          .map((part) => ({ text: part.text }));
        messages.push({
          role: "user",
          content: [
            {
              toolResult: {
                toolUseId: message.toolCallId ?? "",
                content: toolContent,
              },
            },
          ],
        });
        continue;
      }
      const content: BedrockContentBlock[] = [];
      for (const part of message.content) {
        if (part.type === "text") {
          content.push({ text: part.text });
        } else if (part.type === "tool_use") {
          content.push({
            toolUse: {
              toolUseId: part.id,
              name: part.name,
              input: part.input,
            },
          });
        } else if (part.type === "image") {
          const image = await this.resolveImageContent(part.image);
          content.push({ image });
        }
      }
      messages.push({
        role: message.role === "assistant" ? "assistant" : "user",
        content,
      });
    }

    const payload: Record<string, unknown> = {
      messages,
    };
    if (system.length) {
      payload.system = system;
    }
    const inferenceConfig: Record<string, unknown> = {};
    if (input.maxTokens !== undefined) {
      inferenceConfig.maxTokens = input.maxTokens;
    }
    if (input.temperature !== undefined) {
      inferenceConfig.temperature = input.temperature;
    }
    if (input.topP !== undefined) {
      inferenceConfig.topP = input.topP;
    }
    if (Object.keys(inferenceConfig).length) {
      payload.inferenceConfig = inferenceConfig;
    }
    const toolConfig = buildToolConfig(input.tools, input.toolChoice);
    if (toolConfig) {
      payload.toolConfig = toolConfig;
    }
    return payload;
  }

  private async resolveImageContent(image: {
    url?: string;
    base64?: string;
    mediaType?: string;
  }): Promise<{ format: string; source: { bytes: string } }> {
    if (image.base64) {
      return {
        format: formatFromMediaType(image.mediaType),
        source: { bytes: image.base64 },
      };
    }
    if (!image.url) {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: "Bedrock image content requires a base64 payload or URL",
        provider: this.provider,
      });
    }
    const { data, mediaType } = await this.fetchImageAsBase64(image.url);
    return {
      format: formatFromMediaType(image.mediaType ?? mediaType),
      source: { bytes: data },
    };
  }

  private async fetchImageAsBase64(url: string) {
    const fetchImpl = this.fetchImpl ?? globalThis.fetch;
    if (!fetchImpl) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: "global fetch is not available",
        provider: this.provider,
      });
    }
    const response = await fetchImpl(url);
    if (!response.ok) {
      throw new AiKitError({
        kind: ErrorKind.Validation,
        message: `Failed to load image URL (${response.status})`,
        provider: this.provider,
        upstreamStatus: response.status,
      });
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    return {
      data: buffer.toString("base64"),
      mediaType: response.headers.get("content-type") ?? undefined,
    };
  }

  private normalizeConverseResponse(response: BedrockConverseResponse): GenerateOutput {
    const blocks = response.output?.message?.content ?? [];
    const text = blocks
      .filter((block): block is { text: string } => "text" in block)
      .map((block) => block.text)
      .join("");
    const toolCalls: ToolCall[] = blocks
      .filter(
        (block): block is { toolUse: { toolUseId: string; name: string; input: unknown } } =>
          "toolUse" in block,
      )
      .map((block) => ({
        id: block.toolUse.toolUseId,
        name: block.toolUse.name,
        argumentsJson: JSON.stringify(block.toolUse.input ?? {}),
      }));
    return {
      text: text || undefined,
      toolCalls: toolCalls.length ? toolCalls : undefined,
      finishReason:
        response.stopReason ?? (toolCalls.length ? "tool_calls" : "stop"),
      usage: mapUsage(response.usage),
      raw: response,
    };
  }

  private async fetchSignedJSON<T>(
    url: string,
    init: { method: string; body?: unknown; signal?: AbortSignal },
    service: string,
  ): Promise<{ data: T; response: Response }> {
    const fetchImpl = this.fetchImpl ?? globalThis.fetch;
    if (!fetchImpl) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: "global fetch is not available",
        provider: this.provider,
      });
    }
    const body =
      init.body === undefined
        ? ""
        : typeof init.body === "string"
          ? init.body
          : JSON.stringify(init.body);
    const headers = signAwsRequest({
      method: init.method,
      url,
      body,
      region: this.config.region,
      service,
      credentials: this.config.credentials,
      headers: {
        "content-type": "application/json",
        accept: "application/json",
      },
    });
    const response = await fetchImpl(url, {
      method: init.method,
      headers,
      body: body && init.method !== "GET" ? body : undefined,
      signal: init.signal,
    });
    if (!response.ok) {
      await throwBedrockError(this.provider, response);
    }
    const data = (await response.json()) as T;
    return { data, response };
  }
}

function resolveBedrockConfig(config: BedrockProviderConfig): ResolvedBedrockConfig {
  const region = config.region?.trim() || ENV_REGION;
  if (!region) {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "Bedrock region is required",
      provider: Provider.Bedrock,
    });
  }
  const accessKeyId = config.accessKeyId?.trim() || ENV_ACCESS_KEY;
  const secretAccessKey = config.secretAccessKey?.trim() || ENV_SECRET_KEY;
  if (!accessKeyId || !secretAccessKey) {
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "Bedrock AWS credentials are required",
      provider: Provider.Bedrock,
    });
  }
  const sessionToken = config.sessionToken?.trim() || ENV_SESSION_TOKEN || undefined;
  const endpoint =
    config.endpoint?.trim() || `https://bedrock.${region}.amazonaws.com`;
  const runtimeEndpoint =
    config.runtimeEndpoint?.trim() ||
    `https://bedrock-runtime.${region}.amazonaws.com`;
  return {
    region,
    endpoint,
    runtimeEndpoint,
    controlPlaneService: config.controlPlaneService?.trim() || "bedrock",
    runtimeService: config.runtimeService?.trim() || "bedrock-runtime",
    credentials: { accessKeyId, secretAccessKey, sessionToken },
  };
}

function extractText(content: GenerateInput["messages"][number]["content"]): string {
  return content
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
}

function deriveFamily(modelId: string, providerName?: string): string {
  if (providerName) {
    return providerName.toLowerCase();
  }
  const [prefix] = modelId.split(".");
  return prefix || modelId;
}

function supportsToolUse(modelId: string): boolean {
  return modelId.startsWith("anthropic.") || modelId.startsWith("cohere.");
}

function buildToolConfig(
  tools?: ToolDefinition[],
  choice?: GenerateInput["toolChoice"],
): Record<string, unknown> | undefined {
  if (!tools?.length) {
    return undefined;
  }
  const config: Record<string, unknown> = {
    tools: tools.map((tool) => ({
      toolSpec: {
        name: tool.name,
        description: tool.description,
        inputSchema: { json: tool.parameters },
      },
    })),
  };
  if (choice) {
    config.toolChoice = normalizeToolChoice(choice);
  }
  return config;
}

function normalizeToolChoice(choice: GenerateInput["toolChoice"]) {
  if (choice === "auto") {
    return { auto: {} };
  }
  if (choice === "none") {
    return { none: {} };
  }
  return { tool: { name: choice.name } };
}

function mapUsage(
  usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number },
): Usage | undefined {
  if (!usage) return undefined;
  return {
    inputTokens: usage.inputTokens,
    outputTokens: usage.outputTokens,
    totalTokens:
      usage.totalTokens ??
      (usage.inputTokens && usage.outputTokens
        ? usage.inputTokens + usage.outputTokens
        : undefined),
  };
}

function formatFromMediaType(mediaType?: string): string {
  const normalized = mediaType?.toLowerCase() ?? "";
  if (normalized.includes("png")) return "png";
  if (normalized.includes("jpeg") || normalized.includes("jpg")) return "jpeg";
  if (normalized.includes("webp")) return "webp";
  if (normalized.includes("gif")) return "gif";
  return "png";
}

function signAwsRequest(options: {
  method: string;
  url: string;
  body: string;
  region: string;
  service: string;
  credentials: AwsCredentials;
  headers?: Record<string, string>;
}): Record<string, string> {
  const parsed = new URL(options.url);
  const now = new Date();
  const amzDate = toAmzDate(now);
  const dateStamp = amzDate.slice(0, 8);
  const payloadHash = sha256Hex(options.body);
  const headers: Record<string, string> = {
    ...(options.headers ?? {}),
    host: parsed.host,
    "x-amz-date": amzDate,
    "x-amz-content-sha256": payloadHash,
  };
  if (options.credentials.sessionToken) {
    headers["x-amz-security-token"] = options.credentials.sessionToken;
  }
  const { canonicalHeaders, signedHeaders } = buildCanonicalHeaders(headers);
  const canonicalRequest = [
    options.method.toUpperCase(),
    canonicalUri(parsed.pathname),
    canonicalQuery(parsed.searchParams),
    canonicalHeaders,
    signedHeaders,
    payloadHash,
  ].join("\n");
  const scope = `${dateStamp}/${options.region}/${options.service}/aws4_request`;
  const stringToSign = [
    "AWS4-HMAC-SHA256",
    amzDate,
    scope,
    sha256Hex(canonicalRequest),
  ].join("\n");
  const signingKey = deriveSigningKey(
    options.credentials.secretAccessKey,
    dateStamp,
    options.region,
    options.service,
  );
  const signature = hmacHex(signingKey, stringToSign);
  headers.Authorization = `AWS4-HMAC-SHA256 Credential=${options.credentials.accessKeyId}/${scope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;
  return headers;
}

function buildCanonicalHeaders(headers: Record<string, string>) {
  const entries = Object.entries(headers).map(([key, value]) => [
    key.toLowerCase(),
    normalizeHeaderValue(value),
  ]);
  entries.sort(([a], [b]) => a.localeCompare(b));
  const canonicalHeaders = entries
    .map(([key, value]) => `${key}:${value}\n`)
    .join("");
  const signedHeaders = entries.map(([key]) => key).join(";");
  return { canonicalHeaders, signedHeaders };
}

function canonicalUri(pathname: string) {
  if (!pathname) return "/";
  return pathname
    .split("/")
    .map((segment) => encodeRFC3986(segment))
    .join("/");
}

function canonicalQuery(params: URLSearchParams) {
  const entries: Array<[string, string]> = [];
  params.forEach((value, key) => {
    entries.push([key, value]);
  });
  entries.sort(([aKey, aVal], [bKey, bVal]) => {
    if (aKey === bKey) {
      return aVal.localeCompare(bVal);
    }
    return aKey.localeCompare(bKey);
  });
  return entries
    .map(
      ([key, value]) =>
        `${encodeRFC3986(key)}=${encodeRFC3986(value)}`,
    )
    .join("&");
}

function encodeRFC3986(value: string) {
  return encodeURIComponent(value).replace(
    /[!'()*]/g,
    (char) => `%${char.charCodeAt(0).toString(16).toUpperCase()}`,
  );
}

function normalizeHeaderValue(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function sha256Hex(value: string) {
  return createHash("sha256").update(value).digest("hex");
}

function hmacHex(key: Buffer, value: string) {
  return createHmac("sha256", key).update(value).digest("hex");
}

function deriveSigningKey(
  secretAccessKey: string,
  dateStamp: string,
  region: string,
  service: string,
): Buffer {
  const kDate = createHmac("sha256", `AWS4${secretAccessKey}`)
    .update(dateStamp)
    .digest();
  const kRegion = createHmac("sha256", kDate).update(region).digest();
  const kService = createHmac("sha256", kRegion).update(service).digest();
  return createHmac("sha256", kService).update("aws4_request").digest();
}

function toAmzDate(date: Date) {
  const iso = date.toISOString().replace(/[:-]|\.\d{3}/g, "");
  return iso.slice(0, 15) + "Z";
}

async function throwBedrockError(
  provider: Provider,
  response: Response,
): Promise<never> {
  const text = await response.text().catch(() => "");
  const { message, code } = extractBedrockError(text);
  throw new AiKitError({
    kind: classifyStatus(response.status),
    message: message || `Bedrock error (${response.status})`,
    provider,
    upstreamStatus: response.status,
    upstreamCode: code,
    requestId:
      response.headers.get("x-amzn-requestid") ??
      response.headers.get("x-amzn-request-id") ??
      undefined,
  });
}

function extractBedrockError(body: string): { message?: string; code?: string } {
  if (!body) {
    return {};
  }
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.message === "string") {
      return { message: parsed.message, code: parsed.code };
    }
    if (typeof parsed?.error === "string") {
      return { message: parsed.error };
    }
    if (typeof parsed?.error?.message === "string") {
      return { message: parsed.error.message, code: parsed.error.code };
    }
    if (typeof parsed?.__type === "string") {
      return { message: parsed.message, code: parsed.__type };
    }
  } catch {
    return { message: body };
  }
  return { message: body };
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
