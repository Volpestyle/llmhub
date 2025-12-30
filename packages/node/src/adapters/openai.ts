import { readFile } from "fs/promises";
import { basename } from "path";
import { ProviderAdapter } from "../core/provider.js";
import {
  AudioInput,
  GenerateInput,
  GenerateOutput,
  ImageGenerateInput,
  ImageGenerateOutput,
  ModelMetadata,
  OpenAIProviderConfig,
  Provider,
  ResponseFormat,
  StreamChunk,
  ToolCall,
  ToolDefinition,
  TranscriptSegment,
  TranscribeInput,
  TranscribeOutput,
  Usage,
  FetchLike,
} from "../core/types.js";
import { AiKitError } from "../core/errors.js";
import { ErrorKind } from "../core/types.js";
import { streamSSE, parseEventData } from "../core/stream.js";

interface OpenAIModelList {
  data: Array<{ id: string }>;
}

interface OpenAIImageResponse {
  data: Array<{
    b64_json?: string;
    url?: string;
  }>;
}

interface OpenAITranscriptionResponse {
  text?: string;
  language?: string;
  duration?: number;
  segments?: Array<{
    start?: number;
    end?: number;
    text?: string;
  }>;
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
  required_action?: ResponsesRequiredAction;
}

interface ResponsesFunctionToolCall {
  id: string;
  type?: string;
  function: {
    name: string;
    arguments: string;
  };
}

interface ResponsesRequiredAction {
  type: string;
  submit_tool_outputs?: {
    tool_calls: ResponsesFunctionToolCall[];
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
  required_action?: ResponsesRequiredAction;
  error?: {
    message?: string;
    code?: string;
    type?: string;
    param?: string;
  };
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

  async generateImage(
    input: ImageGenerateInput,
  ): Promise<ImageGenerateOutput> {
    if (input.inputImages?.length) {
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: "OpenAI image edits are not supported in this adapter",
        provider: this.provider,
      });
    }
    const response = await this.fetchJSON<OpenAIImageResponse>("/v1/images", {
      method: "POST",
      body: {
        model: input.model,
        prompt: input.prompt,
        size: input.size ?? "1024x1024",
        response_format: "b64_json",
        n: 1,
      },
    });
    const image = response.data?.[0];
    if (!image?.b64_json) {
      throw new AiKitError({
        kind: ErrorKind.Unknown,
        message: "OpenAI image response missing base64 data",
        provider: this.provider,
      });
    }
    return {
      mime: "image/png",
      data: image.b64_json,
      raw: response,
    };
  }

  async transcribe(input: TranscribeInput): Promise<TranscribeOutput> {
    const audio = await this.loadAudioInput(input.audio);
    const form = new FormData();
    form.append("model", input.model);
    form.append("response_format", "verbose_json");
    if (input.language) {
      form.append("language", input.language);
    }
    if (input.prompt) {
      form.append("prompt", input.prompt);
    }
    if (typeof input.temperature === "number") {
      form.append("temperature", String(input.temperature));
    }
    form.append("file", new Blob([audio.data], { type: audio.mediaType }), audio.fileName);

    const response = await this.fetchForm<OpenAITranscriptionResponse>(
      "/v1/audio/transcriptions",
      form,
    );
    return {
      text: response.text,
      language: response.language,
      duration: response.duration,
      segments: this.normalizeSegments(response.segments),
      raw: response,
    };
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
        case "response.refusal.delta":
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
        case "response.required_action": {
          const action =
            payload.required_action ?? payload.response?.required_action;
          for (const chunk of this.mapRequiredAction(toolStates, action)) {
            yield chunk;
          }
          break;
        }
        case "response.completed":
          yield {
            type: "message_end",
            usage: mapResponsesUsage(payload.response?.usage),
            finishReason: payload.response?.status ?? "completed",
          };
          break;
        case "response.error":
          yield {
            type: "error",
            error: {
              kind: "upstream_error",
              message:
                payload.error?.message ?? "OpenAI streaming error",
              upstreamCode: payload.error?.code,
            },
          };
          return;
        case "response.failed":
        case "response.canceled":
          yield {
            type: "message_end",
            finishReason:
              payload.response?.status ??
              event.event.replace("response.", ""),
          };
          return;
        case "response.output_audio.delta":
          yield {
            type: "error",
            error: {
              kind: "unsupported",
              message: "Audio streaming is not supported by this adapter",
            },
          };
          return;
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
    const appendToolCall = (call: {
      id?: string;
      name?: string;
      arguments?: string;
    }) => {
      if (!call) return;
      const id = call.id ?? `tool_${toolCalls.length}`;
      toolCalls.push({
        id,
        name: call.name ?? "",
        argumentsJson: call.arguments ?? "",
      });
    };
    for (const output of data.output ?? []) {
      for (const content of output.content ?? []) {
        if (content.type === "output_text" && content.text) {
          textParts.push(content.text);
        } else if (content.type === "tool_call") {
          appendToolCall({
            id: content.id,
            name: content.name,
            arguments: stringifyArguments(content.arguments),
          });
        }
      }
    }
    const requiredToolCalls =
      data.required_action?.submit_tool_outputs?.tool_calls ?? [];
    for (const call of requiredToolCalls) {
      appendToolCall({
        id: call.id,
        name: call.function?.name,
        arguments: call.function?.arguments ?? "",
      });
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
    init: RequestInit & { body?: Record<string, unknown> },
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
      throw new AiKitError({
        kind: ErrorKind.Unsupported,
        message: "global fetch is not available",
        provider: this.provider,
      });
    }
    const url = `${this.baseURL}${path}`;
    const headers: Record<string, string> = {
      ...(init.headers ?? {}),
    };
    const hasContentType = Object.keys(headers).some(
      (key) => key.toLowerCase() === "content-type",
    );
    const isFormData =
      typeof FormData !== "undefined" && init.body instanceof FormData;
    if (!hasContentType && !isFormData) {
      headers["content-type"] = "application/json";
    }
    if (this.config.apiKey) {
      headers.Authorization = `Bearer ${this.config.apiKey}`;
    }
    if (this.config.organization) {
      headers["OpenAI-Organization"] = this.config.organization;
    }
    const response = await fetchImpl(url, {
      ...init,
      headers: {
        ...headers,
      },
      signal: (init as any).signal,
    });
    if (!response.ok) {
      const errorBody = await response.text().catch(() => "");
      throw new AiKitError({
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

  private async fetchForm<T>(path: string, body: FormData): Promise<T> {
    const response = await this.fetchRaw(path, {
      method: "POST",
      body,
    });
    const json = await response.json();
    return json as T;
  }

  private normalizeSegments(
    segments?: OpenAITranscriptionResponse["segments"],
  ): TranscriptSegment[] | undefined {
    if (!segments?.length) {
      return undefined;
    }
    return segments
      .filter((segment) => typeof segment.text === "string")
      .map((segment) => ({
        start: segment.start ?? 0,
        end: segment.end ?? 0,
        text: segment.text ?? "",
      }));
  }

  private async loadAudioInput(input: AudioInput): Promise<{
    data: Uint8Array;
    fileName: string;
    mediaType: string;
  }> {
    if (input.path) {
      const data = await readFile(input.path);
      return {
        data,
        fileName: input.fileName ?? basename(input.path),
        mediaType: input.mediaType ?? "application/octet-stream",
      };
    }
    if (input.base64) {
      const { data, mediaType } = decodeBase64Input(input.base64, input.mediaType);
      return {
        data,
        fileName: input.fileName ?? "audio",
        mediaType,
      };
    }
    if (input.url) {
      const fetchImpl = this.fetchImpl ?? globalThis.fetch;
      if (!fetchImpl) {
        throw new AiKitError({
          kind: ErrorKind.Unsupported,
          message: "global fetch is not available",
          provider: this.provider,
        });
      }
      const response = await fetchImpl(input.url);
      if (!response.ok) {
        throw new AiKitError({
          kind: ErrorKind.Validation,
          message: `Unable to fetch audio URL (${response.status})`,
          provider: this.provider,
        });
      }
      const data = new Uint8Array(await response.arrayBuffer());
      return {
        data,
        fileName: input.fileName ?? "audio",
        mediaType:
          input.mediaType ??
          response.headers.get("content-type") ??
          "application/octet-stream",
      };
    }
    throw new AiKitError({
      kind: ErrorKind.Validation,
      message: "Transcribe input requires audio.url, audio.base64, or audio.path",
      provider: this.provider,
    });
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

  private mapRequiredAction(
    toolStates: Map<string, ToolState>,
    action?: ResponsesRequiredAction,
  ): StreamChunk[] {
    const calls =
      action?.submit_tool_outputs?.tool_calls ?? [];
    if (!calls.length) {
      return [];
    }
    const chunks: StreamChunk[] = [];
    for (const call of calls) {
      const id = call.id ?? `tool_${toolStates.size}`;
      const state = {
        id,
        name: call.function?.name,
        arguments: call.function?.arguments ?? "",
      };
      toolStates.set(id, state);
      chunks.push({
        type: "tool_call",
        call: {
          id: state.id,
          name: state.name ?? "",
          argumentsJson: state.arguments,
        },
      });
    }
    return chunks;
  }
}

function decodeBase64Input(
  raw: string,
  explicitType?: string,
): { data: Uint8Array; mediaType: string } {
  if (raw.startsWith("data:")) {
    const [header, payload] = raw.split(",", 2);
    const mediaMatch = header.match(/^data:([^;]+);base64$/);
    const mediaType =
      explicitType ?? mediaMatch?.[1] ?? "application/octet-stream";
    return { data: Buffer.from(payload ?? "", "base64"), mediaType };
  }
  return {
    data: Buffer.from(raw, "base64"),
    mediaType: explicitType ?? "application/octet-stream",
  };
}

function stringifyArguments(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return "";
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
  return messages.map((message) => {
    const content = message.content.map((part) => {
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
    });
    const entry: Record<string, unknown> = {
      role: message.role,
      content,
    };
    if (message.toolCallId) {
      entry.tool_call_id = message.toolCallId;
    }
    if (message.name) {
      entry.name = message.name;
    }
    return entry;
  });
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
