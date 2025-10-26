export enum Provider {
  OpenAI = "openai",
  Anthropic = "anthropic",
  XAI = "xai",
  Google = "google",
}

export type ProviderMap<T> = Partial<Record<Provider, T>>;

export interface ModelCapabilities {
  text: boolean;
  vision: boolean;
  tool_use: boolean;
  structured_output: boolean;
  reasoning: boolean;
}

export interface TokenPrices {
  input?: number;
  output?: number;
}

export interface ModelMetadata {
  id: string;
  displayName: string;
  provider: Provider;
  family?: string;
  capabilities: ModelCapabilities;
  contextWindow?: number;
  tokenPrices?: TokenPrices;
  deprecated?: boolean;
  inPreview?: boolean;
}

export interface TextContentPart {
  type: "text";
  text: string;
}

export interface ImageContentPart {
  type: "image";
  image: {
    /**
     * Remote URL for provider-hosted images.
     */
    url?: string;
    /**
     * Base64 data URI payload (without prefix).
     */
    base64?: string;
    mediaType?: string;
  };
}

export type ContentPart = TextContentPart | ImageContentPart;

export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface Message {
  role: MessageRole;
  content: ContentPart[];
  /**
   * Identifier linking a tool response back to the originating tool call.
   * Required by OpenAI when sending tool role messages.
   */
  toolCallId?: string;
  name?: string;
}

export interface JsonSchema {
  type: string;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  required?: string[];
  enum?: Array<string | number>;
  description?: string;
  additionalProperties?: boolean | JsonSchema;
  format?: string;
  const?: unknown;
  minimum?: number;
  maximum?: number;
  minItems?: number;
  maxItems?: number;
  pattern?: string;
  [additional: string]: unknown;
}

export interface ToolDefinition {
  name: string;
  description?: string;
  parameters: JsonSchema;
}

export type ToolChoice =
  | "auto"
  | "none"
  | { type: "tool"; name: string };

export interface JsonSchemaResponseFormat {
  type: "json_schema";
  jsonSchema: {
    name: string;
    schema: JsonSchema;
    strict?: boolean;
  };
}

export interface TextResponseFormat {
  type: "text";
}

export type ResponseFormat = JsonSchemaResponseFormat | TextResponseFormat;

export interface ToolCall {
  id: string;
  name: string;
  argumentsJson: string;
}

export interface Usage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

export interface GenerateInput {
  provider: Provider;
  model: string;
  messages: Message[];
  tools?: ToolDefinition[];
  toolChoice?: ToolChoice;
  responseFormat?: ResponseFormat;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  stream?: boolean;
  metadata?: Record<string, string>;
  signal?: AbortSignal;
}

export interface GenerateOutput {
  text?: string;
  toolCalls?: ToolCall[];
  usage?: Usage;
  finishReason?: string;
  raw?: unknown;
}

export type StreamChunk =
  | { type: "delta"; textDelta: string }
  | { type: "tool_call"; call: ToolCall; delta?: string }
  | { type: "message_end"; usage?: Usage; finishReason?: string }
  | {
      type: "error";
      error: {
        kind: string;
        message: string;
        upstreamCode?: string;
        requestId?: string;
      };
    };

export interface ListModelsParams {
  providers?: Provider[];
  refresh?: boolean;
}

export enum ErrorKind {
  Unknown = "unknown_error",
  ProviderAuth = "provider_auth_error",
  ProviderRateLimit = "provider_rate_limit",
  ProviderNotFound = "provider_not_found",
  ProviderUnavailable = "provider_unavailable",
  Validation = "validation_error",
  Unsupported = "unsupported",
  Timeout = "timeout",
}

export interface HubErrorPayload {
  kind: ErrorKind;
  message: string;
  provider?: Provider;
  upstreamStatus?: number;
  upstreamCode?: string;
  requestId?: string;
  cause?: unknown;
}

export type FetchLike = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export interface OpenAIProviderConfig {
  apiKey: string;
  baseURL?: string;
  organization?: string;
  defaultUseResponses?: boolean;
}

export interface AnthropicProviderConfig {
  apiKey: string;
  baseURL?: string;
  version?: string;
}

export interface XAIProviderConfig {
  apiKey: string;
  baseURL?: string;
  /**
   * "openai" uses the OpenAI-compatible surfaces; "anthropic" routes to Messages API.
   * Default: "openai".
   */
  compatibilityMode?: "openai" | "anthropic";
}

export interface GoogleProviderConfig {
  apiKey: string;
  baseURL?: string;
}

export type ProviderConfigs = {
  [Provider.OpenAI]: OpenAIProviderConfig;
  [Provider.Anthropic]: AnthropicProviderConfig;
  [Provider.XAI]: XAIProviderConfig;
  [Provider.Google]: GoogleProviderConfig;
};

export type AnyProviderConfig =
  ProviderConfigs[keyof ProviderConfigs];

export interface HubConfig {
  providers: ProviderMap<AnyProviderConfig>;
  httpClient?: FetchLike;
  registry?: {
    ttlMs?: number;
  };
}

export interface Hub {
  listModels(params?: ListModelsParams): Promise<ModelMetadata[]>;
  generate(input: GenerateInput): Promise<GenerateOutput>;
  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk>;
}
