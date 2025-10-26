import { ProviderAdapter } from "../core/provider.js";
import {
  GenerateInput,
  GenerateOutput,
  ModelMetadata,
  OpenAIProviderConfig,
  Provider,
  ResponseFormat,
  StreamChunk,
  ToolCall,
  ToolDefinition,
  Usage,
  FetchLike,
} from "../core/types.js";
import { lookupCuratedModel } from "../overlays/curatedModels.js";
import { LLMHubError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";
import { streamSSE, parseEventData } from "../core/stream.js";

interface OpenAIModelList {
  data: Array<{ id: string }>;
}

interface ChatCompletion {
  id: string;
  choices: Array<{
    index: number;
    finish_reason?: string | null;
    message?: {
      content?: Array<{ type: string; text?: string }> | string | null;
      tool_calls?: Array<{
        id: string;
        function: { name: string; arguments: string };
      }>;
    };
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

interface ChatCompletionChunk {
  id: string;
  choices: Array<{
    index: number;
    delta: {
      content?:
        | Array<{ type?: string; text?: string }>
        | string
        | null;
      tool_calls?: Array<{
        index: number;
        id?: string;
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
      role?: string;
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

interface ResponsesOutput {
  id: string;
  status?: string;
  output?: Array<{
    id: string;
    type: string;
    role?: string;
    content?: Array<{
      type: string;
      text?: string;
      id?: string;
      name?: string;
      arguments?: unknown;
    }>;
  }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
}

interface ResponsesSSEPayload {
  response?: ResponsesOutput;
  response_id?: string;
  id?: string;
  status?: string;
  delta?: {
    type?: string;
    text?: string;
    arguments?: string;
    name?: string;
  };
  output_index?: number;
  content_index?: number;
  tool_call_id?: string;
}

interface ToolState {
  id: string;
  name?: string;
  arguments: string;
}

export interface OpenAIAdapterOverrides {
  providerOverride?: Provider;
  baseURLOverride?: string;
}

export class OpenAIAdapter implements ProviderAdapter {
  readonly provider: Provider;
  private readonly config: OpenAIProviderConfig;
  private readonly fetchImpl?: FetchLike;
  private readonly baseURL: string;

  constructor(
    config: OpenAIProviderConfig,
    fetchImpl?: FetchLike,
    overrides?: OpenAIAdapterOverrides,
  ) {
    this.provider = overrides?.providerOverride ?? Provider.OpenAI;
    this.config = config;
    this.fetchImpl = fetchImpl;
    this.baseURL =
      overrides?.baseURLOverride ?? config.baseURL ?? "https://api.openai.com";
  }

  async listModels(): Promise<ModelMetadata[]> {
    const response = await this.fetchJSON<OpenAIModelList>("/v1/models", {
      method: "GET",
    });
    return response.data.map((model) => this.enrichModel(model.id));
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    if (this.shouldUseResponses(input)) {
      const data = await this.fetchJSON<ResponsesOutput>("/v1/responses", {
        method: "POST",
        body: this.buildResponsesPayload({ ...input, stream: false }),
        signal: input.signal,
      });
      return this.normalizeResponsesOutput(data);
    }
    const data = await this.fetchJSON<ChatCompletion>("/v1/chat/completions", {
      method: "POST",
      body: this.buildChatPayload({ ...input, stream: false }),
      signal: input.signal,
    });
    return this.normalizeChatOutput(data);
  }

  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    return this.shouldUseResponses(input)
      ? this.streamResponses(input)
      : this.streamChat(input);
  }

  private async *streamResponses(
    input: GenerateInput,
  ): AsyncIterable<StreamChunk> {
    const response = await this.fetchRaw("/v1/responses", {
      method: "POST",
      body: JSON.stringify(
        this.buildResponsesPayload({ ...input, stream: true }),
      ),
      signal: input.signal,
    });
    const toolStates = new Map<string, ToolState>();
    for await (const event of streamSSE(response)) {
      if (!event.data || event.data === "[DONE]") {
        continue;
      }
      const payload = parseEventData<ResponsesSSEPayload>(event);
      if (!payload) {
        continue;
      }
      switch (event.event) {
        case "response.output_text.delta":
          if (payload.delta?.text) {
            yield { type: "delta", textDelta: payload.delta.text };
          }
          break;
        case "response.tool_call.delta":
        case "response.output_tool_call.delta":
          if (!payload.tool_call_id) {
            break;
          }
          this.updateToolState(toolStates, payload.tool_call_id, payload);
          const state = toolStates.get(payload.tool_call_id);
          if (state) {
            yield {
              type: "tool_call",
              call: {
                id: state.id,
                name: state.name ?? "",
                argumentsJson: state.arguments,
              },
              delta: payload.delta?.arguments,
            };
          }
          break;
        case "response.completed":
          yield {
            type: "message_end",
            usage: mapResponsesUsage(payload.response?.usage),
            finishReason: payload.response?.status ?? "stop",
          };
          break;
        case "response.error":
          throw new LLMHubError({
            kind: ErrorKind.ProviderUnavailable,
            message: "OpenAI streaming error",
            provider: this.provider,
          });
        default:
          break;
      }
    }
  }

  private async *streamChat(input: GenerateInput): AsyncIterable<StreamChunk> {
    const response = await this.fetchRaw("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(this.buildChatPayload({ ...input, stream: true })),
      signal: input.signal,
    });
    const toolStates = new Map<number, ToolState>();
    let finishReason: string | undefined;
    let usage: Usage | undefined;
    for await (const event of streamSSE(response)) {
      if (!event.data || event.data === "[DONE]") {
        continue;
      }
      const payload = parseEventData<ChatCompletionChunk>(event);
      if (!payload) continue;
      if (payload.usage) {
        usage = mapChatUsage(payload.usage);
      }
      for (const choice of payload.choices ?? []) {
        finishReason = choice.finish_reason ?? finishReason;
        const delta = choice.delta;
        if (!delta) continue;
        const parts = Array.isArray(delta.content)
          ? delta.content
          : delta.content
            ? [{ type: "text", text: String(delta.content) }]
            : [];
        for (const part of parts) {
          if (typeof part === "string") {
            yield { type: "delta", textDelta: part };
          } else if (part?.type === "text" && part.text) {
            yield { type: "delta", textDelta: part.text };
          }
        }
        if (delta.tool_calls) {
          for (const toolDiff of delta.tool_calls) {
            const state = toolStates.get(toolDiff.index) ?? {
              id: toolDiff.id ?? `tool_${toolDiff.index}`,
              name: toolDiff.function?.name,
              arguments: "",
            };
            if (toolDiff.id) {
              state.id = toolDiff.id;
            }
            if (toolDiff.function?.name) {
              state.name = toolDiff.function.name;
            }
            if (toolDiff.function?.arguments) {
              state.arguments += toolDiff.function.arguments;
            }
            toolStates.set(toolDiff.index, state);
            yield {
              type: "tool_call",
              call: {
                id: state.id,
                name: state.name ?? "",
                argumentsJson: state.arguments,
              },
              delta: toolDiff.function?.arguments,
            };
          }
        }
      }
    }
    yield {
      type: "message_end",
      finishReason: finishReason ?? "stop",
      usage,
    };
  }

  private buildChatPayload(input: GenerateInput) {
    return {
      model: input.model,
      messages: mapMessagesToChat(input.messages),
      temperature: input.temperature,
      top_p: input.topP,
      max_tokens: input.maxTokens,
      stream: input.stream ?? false,
      tools: mapTools(input.tools),
      tool_choice: mapToolChoice(input.toolChoice),
      response_format: mapResponseFormat(input.responseFormat),
    };
  }

  private buildResponsesPayload(input: GenerateInput) {
    return {
      model: input.model,
      input: mapMessagesToResponses(input.messages),
      tools: mapTools(input.tools),
      tool_choice: mapToolChoice(input.toolChoice),
      response_format: mapResponseFormat(input.responseFormat),
      temperature: input.temperature,
      top_p: input.topP,
      max_output_tokens: input.maxTokens,
      metadata: input.metadata,
      stream: input.stream ?? false,
    };
  }

  private normalizeChatOutput(data: ChatCompletion): GenerateOutput {
    const choice = data.choices?.[0];
    const toolCalls =
      choice?.message?.tool_calls?.map((call) => ({
        id: call.id,
        name: call.function.name,
        argumentsJson: call.function.arguments,
      })) ?? [];
    return {
      text: extractChatText(choice),
      toolCalls: toolCalls.length ? toolCalls : undefined,
      finishReason: choice?.finish_reason ?? "stop",
      usage: mapChatUsage(data.usage),
      raw: data,
    };
  }

  private normalizeResponsesOutput(data: ResponsesOutput): GenerateOutput {
    const toolCalls: ToolCall[] = [];
    const textParts: string[] = [];
    for (const output of data.output ?? []) {
      for (const content of output.content ?? []) {
        if (content.type === "output_text" && content.text) {
          textParts.push(content.text);
        } else if (content.type === "tool_call") {
          toolCalls.push({
            id: content.id ?? `tool_${toolCalls.length}`,
            name: content.name ?? "",
            argumentsJson: JSON.stringify(content.arguments ?? {}),
          });
        }
      }
    }
    return {
      text: textParts.join(""),
      toolCalls: toolCalls.length ? toolCalls : undefined,
      finishReason: data.status ?? "stop",
      usage: mapResponsesUsage(data.usage),
      raw: data,
    };
  }

  private enrichModel(modelId: string): ModelMetadata {
    const curated = lookupCuratedModel(this.provider, modelId);
    if (curated) {
      return curated;
    }
    return {
      id: modelId,
      displayName: modelId,
      provider: this.provider,
      family: deriveFamily(modelId),
      capabilities: {
        text: true,
        vision: false,
        tool_use: true,
        structured_output: false,
        reasoning: false,
      },
    };
  }

  private shouldUseResponses(input: GenerateInput): boolean {
    return (
      input.responseFormat?.type === "json_schema" ||
      this.config.defaultUseResponses === true
    );
  }

  private async fetchJSON<T>(
    path: string,
    init: RequestInit & { body?: any },
  ): Promise<T> {
    const response = await this.fetchRaw(path, {
      ...init,
      body:
        init.body && typeof init.body !== "string"
          ? JSON.stringify(init.body)
          : init.body,
    });
    const json = await response.json();
    return json as T;
  }

  private async fetchRaw(
    path: string,
    init: RequestInit,
  ): Promise<Response> {
    const fetchImpl = this.fetchImpl ?? globalThis.fetch;
    if (!fetchImpl) {
      throw new LLMHubError({
        kind: ErrorKind.Unsupported,
        message: "global fetch is not available",
        provider: this.provider,
      });
    }
    const url = `${this.baseURL}${path}`;
    const response = await fetchImpl(url, {
      ...init,
      headers: {
        "content-type": "application/json",
        Authorization: `Bearer ${this.config.apiKey}`,
        ...(this.config.organization
          ? { "OpenAI-Organization": this.config.organization }
          : {}),
        ...(init.headers ?? {}),
      },
      signal: (init as any).signal,
    });
    if (!response.ok) {
      const errorBody = await response.text().catch(() => "");
      throw new LLMHubError({
        kind:
          response.status === 401 || response.status === 403
            ? ErrorKind.ProviderAuth
            : response.status === 429
              ? ErrorKind.ProviderRateLimit
              : response.status >= 500
                ? ErrorKind.ProviderUnavailable
                : ErrorKind.Unknown,
        message: errorBody || `OpenAI request failed (${response.status})`,
        provider: this.provider,
        upstreamStatus: response.status,
        requestId: response.headers.get("x-request-id") ?? undefined,
        upstreamCode: safeExtractErrorCode(errorBody),
      });
    }
    return response;
  }

  private updateToolState(
    toolStates: Map<string, ToolState>,
    id: string,
    payload: ResponsesSSEPayload,
  ) {
    const state = toolStates.get(id) ?? { id, arguments: "" };
    if (payload.delta?.name) {
      state.name = payload.delta.name;
    }
    if (payload.delta?.arguments) {
      state.arguments += payload.delta.arguments;
    }
    toolStates.set(id, state);
  }
}

function mapMessagesToChat(messages: GenerateInput["messages"]) {
  return messages.map((message) => {
    if (message.role === "tool") {
      return {
        role: "tool",
        tool_call_id: message.toolCallId,
        content: message.content
          .map((part) => (part.type === "text" ? part.text : ""))
          .join(""),
      };
    }
    const parts = message.content.map((part) => {
      if (part.type === "text") {
        return { type: "text", text: part.text };
      }
      const image: Record<string, string> = {};
      if (part.image.url) {
        image.url = part.image.url;
      }
      if (part.image.base64) {
        image.b64_json = part.image.base64;
      }
      return {
        type: "image_url",
        image_url: image,
      };
    });
    return {
      role: message.role,
      content:
        parts.length === 1 && parts[0].type === "text"
          ? parts[0].text
          : parts,
    };
  });
}

function mapMessagesToResponses(messages: GenerateInput["messages"]) {
  return messages.map((message) => ({
    role: message.role,
    content: message.content.map((part) => {
      if (part.type === "text") {
        return { type: "input_text", text: part.text };
      }
      const payload: Record<string, unknown> = {
        type: "input_image",
        media_type: part.image.mediaType,
      };
      if (part.image.url) {
        payload.image_url = part.image.url;
      }
      if (part.image.base64) {
        payload.image_base64 = part.image.base64;
      }
      return payload;
    }),
  }));
}

function mapTools(tools?: ToolDefinition[]) {
  if (!tools) return undefined;
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
}

function mapToolChoice(choice?: GenerateInput["toolChoice"]) {
  if (!choice) return undefined;
  if (choice === "auto" || choice === "none") {
    return choice;
  }
  return {
    type: "function",
    function: { name: choice.name },
  };
}

function mapResponseFormat(format?: ResponseFormat) {
  if (!format) return undefined;
  if (format.type === "json_schema") {
    return {
      type: "json_schema",
      json_schema: {
        name: format.jsonSchema.name,
        strict: format.jsonSchema.strict ?? true,
        schema: format.jsonSchema.schema,
      },
    };
  }
  return { type: "text" };
}

function extractChatText(choice?: ChatCompletion["choices"][number]) {
  if (!choice?.message?.content) {
    return undefined;
  }
  if (typeof choice.message.content === "string") {
    return choice.message.content;
  }
  return choice.message.content
    .map((part) => part.text ?? "")
    .join("");
}

function mapChatUsage(usage?: ChatCompletion["usage"]) {
  if (!usage) return undefined;
  return {
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
  };
}

function mapResponsesUsage(usage?: ResponsesOutput["usage"]) {
  if (!usage) return undefined;
  return {
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    totalTokens: usage.total_tokens,
  };
}

function deriveFamily(modelId: string) {
  return modelId.split("-").slice(0, 2).join("-");
}

function safeExtractErrorCode(body?: string) {
  if (!body) return undefined;
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
