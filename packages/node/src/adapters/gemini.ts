import { ProviderAdapter } from "../core/provider.js";
import {
  FetchLike,
  GenerateInput,
  GenerateOutput,
  GoogleProviderConfig,
  ImageGenerateInput,
  ImageGenerateOutput,
  ModelMetadata,
  Provider,
  ResponseFormat,
  StreamChunk,
  ToolCall,
  ToolDefinition,
  Usage,
} from "../core/types.js";
import { AiKitError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";
import { parseEventData, streamSSE } from "../core/stream.js";

interface GeminiModelList {
  models: Array<{
    name: string;
    displayName: string;
    description?: string;
    supportedGenerationMethods?: string[];
    inputTokenLimit?: number;
    outputTokenLimit?: number;
  }>;
}

interface GeminiContent {
  role: string;
  parts: Array<GeminiPart>;
}

type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { fileData: { mimeType?: string; fileUri: string } }
  | {
      functionCall: {
        name: string;
        args?: Record<string, unknown>;
      };
    }
  | {
      functionResponse: {
        name: string;
        response?: unknown;
      };
    };

interface GeminiGenerateResponse {
  candidates?: Array<{
    content?: {
      parts?: GeminiPart[];
    };
    finishReason?: string;
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
}

interface GeminiStreamEvent {
  candidates?: GeminiGenerateResponse["candidates"];
  usageMetadata?: GeminiGenerateResponse["usageMetadata"];
}

export class GoogleAdapter implements ProviderAdapter {
  readonly provider = Provider.Google;
  private readonly baseURL: string;
  private readonly apiKey: string;
  private readonly fetchImpl?: FetchLike;

  constructor(config: GoogleProviderConfig, fetchImpl?: FetchLike) {
    this.baseURL =
      config.baseURL ?? "https://generativelanguage.googleapis.com";
    this.apiKey = config.apiKey;
    this.fetchImpl = fetchImpl;
  }

  async listModels(): Promise<ModelMetadata[]> {
    const data = await this.fetchJSON<GeminiModelList>(`/v1beta/models`, {
      method: "GET",
    });
    return data.models.map((model) => {
      const id = model.name.replace("models/", "");
      return {
        id,
        displayName: model.displayName ?? id,
        provider: this.provider,
        family: id.split("-").slice(0, 3).join("-"),
        capabilities: {
          text: true,
          vision: true,
          tool_use: true,
          structured_output: true,
          reasoning: false,
        },
        contextWindow: model.inputTokenLimit,
      };
    });
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const response = await this.fetchJSON<GeminiGenerateResponse>(
      this.buildModelPath(input.model, ":generateContent"),
      {
        method: "POST",
        body: this.buildPayload({ ...input, stream: false }),
        signal: input.signal,
      },
    );
    return this.normalizeResponse(response);
  }

  async generateImage(
    input: ImageGenerateInput,
  ): Promise<ImageGenerateOutput> {
    const payload = buildImagePayload(input);
    const response = await this.fetchJSON<GeminiGenerateResponse>(
      this.buildModelPath(input.model, ":generateContent"),
      {
        method: "POST",
        body: payload,
      },
    );
    const image = extractInlineImage(response);
    if (!image) {
      throw new AiKitError({
        kind: ErrorKind.Unknown,
        message: "Gemini image response missing inline data",
        provider: this.provider,
      });
    }
    return {
      mime: image.mimeType ?? "image/png",
      data: image.data,
      raw: response,
    };
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const body = this.buildPayload({ ...input, stream: true });
    return this.streamContent(
      this.buildModelPath(input.model, ":streamGenerateContent"),
      body,
      input.signal,
    );
  }

  private buildPayload(input: GenerateInput & { stream?: boolean }) {
    const { systemInstruction, contents } = buildGeminiContents(input.messages);
    const generationConfig = buildGenerationConfig(
      input.responseFormat,
      input.temperature,
      input.topP,
      input.maxTokens,
    );
    return {
      contents,
      systemInstruction,
      generationConfig,
      tools: buildTools(input.tools),
      toolConfig: buildToolConfig(input.toolChoice),
      safetySettings: undefined,
      stream: input.stream ?? false,
    };
  }

  private normalizeResponse(response: GeminiGenerateResponse): GenerateOutput {
    const candidate = response.candidates?.[0];
    const parts = candidate?.content?.parts ?? [];
    const text = parts
      .filter((part): part is { text: string } => "text" in part)
      .map((part) => part.text)
      .join("");
    const toolCalls = parts
      .filter((part): part is { functionCall: { name: string; args?: Record<string, unknown> } } =>
        Boolean((part as any).functionCall),
      )
      .map((part, index) => ({
        id: part.functionCall.name ?? `call_${index}`,
        name: part.functionCall.name ?? `call_${index}`,
        argumentsJson: JSON.stringify(part.functionCall.args ?? {}),
      }));
    return {
      text: text || undefined,
      toolCalls: toolCalls.length ? toolCalls : undefined,
      finishReason: candidate?.finishReason ?? "stop",
      usage: mapGeminiUsage(response.usageMetadata),
      raw: response,
    };
  }

  private async *streamContent(
    path: string,
    body: unknown,
    signal?: AbortSignal,
  ): AsyncIterable<StreamChunk> {
    const response = await this.fetchRaw(`${path}?alt=sse`, {
      method: "POST",
      body: JSON.stringify(body),
      signal,
      headers: {
        "content-type": "application/json",
      },
    });
    let usage: Usage | undefined;
    for await (const event of streamSSE(response)) {
      if (!event.data || event.data === "[DONE]") {
        continue;
      }
      const payload = parseEventData<GeminiStreamEvent>(event);
      if (!payload) continue;
      if (payload.usageMetadata) {
        usage = mapGeminiUsage(payload.usageMetadata);
      }
      for (const candidate of payload.candidates ?? []) {
        const parts = candidate?.content?.parts ?? [];
        for (const part of parts) {
          if ("text" in part && part.text) {
            yield { type: "delta", textDelta: part.text };
          } else if ("functionCall" in part) {
            yield {
              type: "tool_call",
              call: {
                id: part.functionCall.name,
                name: part.functionCall.name,
                argumentsJson: JSON.stringify(part.functionCall.args ?? {}),
              },
            };
          }
        }
      }
    }
    yield {
      type: "message_end",
      usage,
      finishReason: "stop",
    };
  }

  private buildModelPath(model: string, method: string) {
    const normalized = model.startsWith("models/") ? model : `models/${model}`;
    return `/v1beta/${normalized}${method}`;
  }

  private async fetchJSON<T>(
    path: string,
    init: RequestInit & { body?: Record<string, unknown> },
  ): Promise<T> {
    const response = await this.fetchRaw(path, {
      ...init,
      body:
        init.body && typeof init.body !== "string"
          ? JSON.stringify(init.body)
          : init.body,
      headers: {
        "content-type": "application/json",
        ...(init.headers ?? {}),
      },
    });
    return (await response.json()) as T;
  }

  private async fetchRaw(path: string, init: RequestInit): Promise<Response> {
    const fetchImpl = this.fetchImpl ?? globalThis.fetch;
    if (!fetchImpl) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: "global fetch is not available",
        provider: this.provider,
      });
    }
    const separator = path.includes("?") ? "&" : "?";
    const url = `${this.baseURL}${path}${separator}key=${encodeURIComponent(this.apiKey)}`;
    const response = await fetchImpl(url, init);
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new AiKitError({
        kind:
          response.status === 401 || response.status === 403
            ? ErrorKind.ProviderAuth
            : response.status === 429
              ? ErrorKind.ProviderRateLimit
              : response.status >= 500
                ? ErrorKind.ProviderUnavailable
                : ErrorKind.Unknown,
        message: text || `Gemini error (${response.status})`,
        provider: this.provider,
        upstreamStatus: response.status,
      });
    }
    return response;
  }
}

function buildGeminiContents(messages: GenerateInput["messages"]) {
  const systemParts: string[] = [];
  const contents: GeminiContent[] = [];
  for (const message of messages) {
    if (message.role === "system") {
      systemParts.push(
        message.content
          .map((part) => (part.type === "text" ? part.text : ""))
          .join(""),
      );
      continue;
    }
    if (message.role === "tool") {
      contents.push({
        role: "user",
        parts: message.content
          .filter((part) => part.type === "text")
          .map((part) => ({
            functionResponse: {
              name: message.name ?? message.toolCallId ?? "tool",
              response: safeParseJSON(part.text),
            },
          })),
      });
      continue;
    }
    contents.push({
      role: message.role === "assistant" ? "model" : "user",
      parts: message.content
        .filter((part) => part.type !== "tool_use")
        .map((part) => {
        if (part.type === "text") {
          return { text: part.text };
        }
        if (part.image.base64) {
          return {
            inlineData: {
              mimeType: part.image.mediaType ?? "image/png",
              data: part.image.base64,
            },
          };
        }
        return {
          fileData: {
            mimeType: part.image.mediaType,
            fileUri: part.image.url ?? "",
          },
        };
      }),
    });
  }
  return {
    systemInstruction: systemParts.length
      ? {
          role: "system",
          parts: systemParts.map((text) => ({ text })),
        }
      : undefined,
    contents,
  };
}

function buildImagePayload(input: ImageGenerateInput) {
  const parts: GeminiPart[] = [{ text: input.prompt }];
  for (const image of input.inputImages ?? []) {
    if (image.base64) {
      parts.push({
        inlineData: {
          mimeType: image.mediaType ?? "image/png",
          data: image.base64,
        },
      });
    } else if (image.url) {
      parts.push({
        fileData: {
          mimeType: image.mediaType,
          fileUri: image.url,
        },
      });
    }
  }
  return {
    contents: [{ role: "user", parts }],
  };
}

function extractInlineImage(response: GeminiGenerateResponse) {
  for (const candidate of response.candidates ?? []) {
    for (const part of candidate.content?.parts ?? []) {
      if ("inlineData" in part && part.inlineData?.data) {
        return part.inlineData;
      }
    }
  }
  return null;
}

function buildGenerationConfig(
  format: ResponseFormat | undefined,
  temperature?: number,
  topP?: number,
  maxTokens?: number,
) {
  const config: Record<string, unknown> = {};
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (format?.type === "json_schema") {
    config.responseMimeType = "application/json";
    config.responseSchema = format.jsonSchema.schema;
  }
  return Object.keys(config).length ? config : undefined;
}

function buildTools(tools?: ToolDefinition[]) {
  if (!tools?.length) return undefined;
  return [
    {
      functionDeclarations: tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      })),
    },
  ];
}

function buildToolConfig(choice?: GenerateInput["toolChoice"]) {
  if (!choice) return undefined;
  if (choice === "auto") {
    return {
      functionCallConfig: { mode: "AUTO" },
    };
  }
  if (choice === "none") {
    return {
      functionCallConfig: { mode: "NONE" },
    };
  }
  return {
    functionCallConfig: {
      mode: "ANY",
      allowedFunctionNames: [choice.name],
    },
  };
}

function mapGeminiUsage(meta?: GeminiGenerateResponse["usageMetadata"]) {
  if (!meta) return undefined;
  return {
    inputTokens: meta.promptTokenCount,
    outputTokens: meta.candidatesTokenCount,
    totalTokens: meta.totalTokenCount,
  };
}

function safeParseJSON(value: string) {
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}
