import { ProviderAdapter } from "../core/provider.js";
import {
  AnthropicProviderConfig,
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
import { LLMHubError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";
import { parseEventData, streamSSE } from "../core/stream.js";

interface AnthropicListResponse {
  data: Array<{
    id: string;
  }>;
}

interface AnthropicMessageResponse {
  id: string;
  content: Array<AnthropicContentBlock>;
  stop_reason?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
}

type AnthropicContentBlock =
  | { type: "text"; text: string }
  | {
      type: "tool_use";
      id: string;
      name: string;
      input: unknown;
    };

interface AnthropicStreamEvent {
  type: string;
  index?: number;
  content_block?: AnthropicContentBlock;
  delta?: {
    type: string;
    text?: string;
    partial_json?: string;
  };
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  message?: {
    stop_reason?: string;
  };
}

export interface AnthropicAdapterOverrides {
  providerOverride?: Provider;
  baseURLOverride?: string;
}

export class AnthropicAdapter implements ProviderAdapter {
  readonly provider: Provider;
  private readonly baseURL: string;
  private readonly apiKey: string;
  private readonly version: string;
  private readonly fetchImpl?: FetchLike;

  constructor(
    config: AnthropicProviderConfig,
    fetchImpl?: FetchLike,
    overrides?: AnthropicAdapterOverrides,
  ) {
    this.provider = overrides?.providerOverride ?? Provider.Anthropic;
    this.baseURL =
      overrides?.baseURLOverride ??
      config.baseURL ??
      "https://api.anthropic.com";
    this.apiKey = config.apiKey;
    this.version = config.version ?? "2023-06-01";
    this.fetchImpl = fetchImpl;
  }

  async listModels(): Promise<ModelMetadata[]> {
    const response = await this.fetchJSON<AnthropicListResponse>("/v1/models", {
      method: "GET",
    });
    return response.data.map((item) =>
      this.enrichModel(item.id),
    );
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const payload = this.buildPayload({ ...input, stream: false });
    const response = await this.fetchJSON<AnthropicMessageResponse>(
      "/v1/messages",
      {
        method: "POST",
        body: payload,
        signal: input.signal,
        useStructuredOutputsBeta: this.usesStructuredOutput(input),
      },
    );
    return this.normalizeResponse(response);
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const payload = this.buildPayload({ ...input, stream: true });
    return this.streamMessages(payload, input.signal, this.usesStructuredOutput(input));
  }

  private async *streamMessages(
    payload: unknown,
    signal?: AbortSignal,
    useStructuredOutputsBeta?: boolean,
  ): AsyncIterable<StreamChunk> {
    const response = await this.fetchRaw("/v1/messages", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
      useStructuredOutputsBeta,
    });
    const toolStates = new Map<
      number,
      { id: string; name: string; arguments: string }
    >();
    let usage: Usage | undefined;
    let finishReason: string | undefined;
    for await (const event of streamSSE(response)) {
      if (!event.data) {
        continue;
      }
      if (event.data === "[DONE]") {
        break;
      }
      const payload = parseEventData<AnthropicStreamEvent>(event);
      if (!payload) continue;
      if (payload.usage) {
        usage = mapAnthropicUsage(payload.usage);
      }
      switch (payload.type) {
        case "content_block_delta":
          if (payload.delta?.type === "text_delta" && payload.delta.text) {
            yield { type: "delta", textDelta: payload.delta.text };
          } else if (
            payload.delta?.type === "input_json_delta" &&
            payload.index !== undefined
          ) {
            const state = toolStates.get(payload.index);
            if (state && payload.delta.partial_json) {
              state.arguments += payload.delta.partial_json;
              yield {
                type: "tool_call",
                call: {
                  id: state.id,
                  name: state.name,
                  argumentsJson: state.arguments,
                },
                delta: payload.delta.partial_json,
              };
            }
          }
          break;
        case "content_block_start":
          if (
            payload.content_block?.type === "tool_use" &&
            payload.index !== undefined
          ) {
            toolStates.set(payload.index, {
              id: payload.content_block.id,
              name: payload.content_block.name,
              arguments: JSON.stringify(payload.content_block.input ?? {}),
            });
          }
          break;
        case "content_block_stop":
          if (payload.index !== undefined) {
            const state = toolStates.get(payload.index);
            if (state) {
              yield {
                type: "tool_call",
                call: {
                  id: state.id,
                  name: state.name,
                  argumentsJson: state.arguments,
                },
              };
            }
          }
          break;
        case "message_delta":
          finishReason = payload.message?.stop_reason ?? finishReason;
          break;
        case "message_stop":
          yield {
            type: "message_end",
            finishReason: finishReason ?? "stop",
            usage,
          };
          break;
        default:
          break;
      }
    }
  }

  private buildPayload(input: GenerateInput & { stream?: boolean }) {
    const { systemPrompt, messages } = splitAnthropicMessages(input.messages);
    const toolConfig = buildAnthropicTools(input.tools, input.toolChoice);
    const payload: Record<string, unknown> = {
      model: input.model,
      system: systemPrompt ?? undefined,
      messages,
      max_tokens: input.maxTokens ?? 1024,
      temperature: input.temperature,
      top_p: input.topP,
      metadata: input.metadata,
      tools: toolConfig.tools,
      tool_choice: toolConfig.toolChoice,
      stream: input.stream ?? false,
    };
    if (input.responseFormat?.type === "json_schema") {
      payload.output_format = {
        type: "json_schema",
        schema: input.responseFormat.jsonSchema.schema,
      };
    }
    return payload;
  }

  private usesStructuredOutput(input: GenerateInput): boolean {
    return input.responseFormat?.type === "json_schema";
  }

  private normalizeResponse(response: AnthropicMessageResponse): GenerateOutput {
    const text = response.content
      .filter((block): block is { type: "text"; text: string } => block.type === "text")
      .map((block) => block.text)
      .join("");
    const toolCalls: ToolCall[] = response.content
      .filter(
        (block): block is { type: "tool_use"; id: string; name: string; input: unknown } =>
          block.type === "tool_use",
      )
      .map((block) => ({
        id: block.id,
        name: block.name,
        argumentsJson: JSON.stringify(block.input ?? {}),
      }));
    return {
      text: text || undefined,
      toolCalls: toolCalls.length ? toolCalls : undefined,
      finishReason: response.stop_reason ?? (toolCalls.length ? "tool_calls" : "stop"),
      usage: mapAnthropicUsage(response.usage),
      raw: response,
    };
  }

  private enrichModel(modelId: string): ModelMetadata {
    return {
      id: modelId,
      displayName: modelId,
      provider: this.provider,
      family: modelId.split("-").slice(0, 3).join("-"),
      capabilities: {
        text: true,
        vision: false,
        tool_use: true,
        structured_output: true,
        reasoning: false,
      },
    };
  }

  private async fetchJSON<T>(
    path: string,
    init: RequestInit & { body?: unknown; useStructuredOutputsBeta?: boolean },
  ): Promise<T> {
    const response = await this.fetchRaw(path, {
      ...init,
      body:
        init.body && typeof init.body !== "string"
          ? JSON.stringify(init.body)
          : (init.body as string | undefined),
      useStructuredOutputsBeta: init.useStructuredOutputsBeta,
    });
    return (await response.json()) as T;
  }

  private async fetchRaw(path: string, init: RequestInit & { useStructuredOutputsBeta?: boolean }): Promise<Response> {
    const fetchImpl = this.fetchImpl ?? globalThis.fetch;
    if (!fetchImpl) {
      throw new LLMHubError({
        kind: ErrorKind.Unsupported,
        message: "global fetch is not available",
        provider: this.provider,
      });
    }
    const headers: Record<string, string> = {
      "content-type": "application/json",
      "x-api-key": this.apiKey,
      "anthropic-version": this.version,
    };
    if (init.useStructuredOutputsBeta) {
      headers["anthropic-beta"] = "structured-outputs-2025-11-13";
    }
    const response = await fetchImpl(`${this.baseURL}${path}`, {
      ...init,
      headers: {
        ...headers,
        ...(init.headers ?? {}),
      },
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new LLMHubError({
        kind:
          response.status === 401 || response.status === 403
            ? ErrorKind.ProviderAuth
            : response.status === 429
              ? ErrorKind.ProviderRateLimit
              : response.status >= 500
                ? ErrorKind.ProviderUnavailable
                : ErrorKind.Unknown,
        message: text || `Anthropic error (${response.status})`,
        provider: this.provider,
        upstreamStatus: response.status,
      });
    }
    return response;
  }
}

function splitAnthropicMessages(messages: GenerateInput["messages"]) {
  const systemParts: string[] = [];
  const chatMessages: Array<{
    role: "user" | "assistant";
    content: unknown[];
  }> = [];
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
      chatMessages.push({
        role: "user",
        content: message.content.map((part) => ({
          type: "tool_result",
          tool_use_id: message.toolCallId ?? "",
          content: part.type === "text" ? part.text : "",
        })),
      });
      continue;
    }
    chatMessages.push({
      role: message.role,
      content: message.content.map((part) => {
        if (part.type === "text") {
          return { type: "text", text: part.text };
        }
        if (part.image.base64) {
          return {
            type: "image",
            source: {
              type: "base64",
              media_type: part.image.mediaType ?? "image/png",
              data: part.image.base64,
            },
          };
        }
        return {
          type: "image",
          source: {
            type: "url",
            url: part.image.url,
          },
        };
      }),
    });
  }
  return {
    systemPrompt: systemParts.length ? systemParts.join("\n") : undefined,
    messages: chatMessages,
  };
}

function buildAnthropicTools(
  tools?: ToolDefinition[],
  choice?: GenerateInput["toolChoice"],
) {
  const toolList = tools ?? [];
  return {
    tools: toolList.length
      ? toolList.map((tool) => ({
          name: tool.name,
          description: tool.description,
          input_schema: tool.parameters,
        }))
      : undefined,
    toolChoice: choiceToAnthropic(choice),
  };
}

function choiceToAnthropic(choice?: GenerateInput["toolChoice"]) {
  if (!choice) return undefined;
  if (choice === "auto") {
    return "auto";
  }
  if (choice === "none") {
    return "none";
  }
  return { type: "tool", name: choice.name };
}

function mapAnthropicUsage(
  usage?: { input_tokens?: number; output_tokens?: number },
): Usage | undefined {
  if (!usage) return undefined;
  return {
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    totalTokens:
      usage.input_tokens && usage.output_tokens
        ? usage.input_tokens + usage.output_tokens
        : undefined,
  };
}
