import type { AdapterMap, ProviderAdapter } from "./provider.js";

export enum Provider {
  OpenAI = "openai",
  Anthropic = "anthropic",
  XAI = "xai",
  Google = "google",
  Ollama = "ollama",
  Local = "local",
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

export interface EntitlementContext {
  provider?: Provider;
  apiKey?: string;
  apiKeyFingerprint?: string;
  accountId?: string;
  region?: string;
  environment?: "dev" | "staging" | "prod";
  tenantId?: string;
  userId?: string;
}

export interface ModelModalities {
  text: boolean;
  vision?: boolean;
  audioIn?: boolean;
  audioOut?: boolean;
  imageOut?: boolean;
}

export interface ModelFeatures {
  tools?: boolean;
  jsonMode?: boolean;
  jsonSchema?: boolean;
  streaming?: boolean;
  batch?: boolean;
}

export interface ModelLimits {
  contextTokens?: number;
  maxOutputTokens?: number;
}

export interface ModelPricing {
  currency: "USD";
  inputPer1M?: number;
  cachedInputPer1M?: number;
  outputPer1M?: number;
  extras?: Record<string, number>;
  effectiveAsOf?: string;
  source?: "config";
}

export type AvailabilityConfidence = "listed" | "inferred" | "learned";

export interface ModelAvailability {
  entitled: boolean;
  lastVerifiedAt?: string;
  confidence?: AvailabilityConfidence;
  reason?: string;
}

export interface ModelRecord {
  id: string;
  provider: Provider;
  providerModelId: string;
  displayName?: string;
  modalities: ModelModalities;
  features: ModelFeatures;
  limits?: ModelLimits;
  tags?: string[];
  pricing?: ModelPricing;
  availability: ModelAvailability;
}

export interface ModelConstraints {
  requireTools?: boolean;
  requireJson?: boolean;
  requireVision?: boolean;
  maxCostUsd?: number;
  latencyClass?: "fast" | "balanced" | "best";
  allowPreview?: boolean;
}

export interface ModelResolutionRequest {
  constraints?: ModelConstraints;
  preferredModels?: string[];
}

export interface ResolvedModel {
  primary: ModelRecord;
  fallback?: ModelRecord[];
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

export interface ToolUseContentPart {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
}

export type ContentPart = TextContentPart | ImageContentPart | ToolUseContentPart;

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

export interface CostBreakdown {
  input_cost_usd?: number;
  output_cost_usd?: number;
  total_cost_usd?: number;
  pricing_per_million?: TokenPrices;
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
  cost?: CostBreakdown;
  raw?: unknown;
}

export interface ImageInput {
  url?: string;
  base64?: string;
  mediaType?: string;
}

export interface AudioInput {
  url?: string;
  base64?: string;
  mediaType?: string;
  fileName?: string;
  path?: string;
}

export interface ImageGenerateInput {
  provider: Provider;
  model: string;
  prompt: string;
  size?: string;
  inputImages?: ImageInput[];
  parameters?: Record<string, unknown>;
}

export interface ImageGenerateOutput {
  mime: string;
  data: string;
  images?: Array<{ mime: string; data: string }>;
  raw?: unknown;
}

export interface MeshGenerateInput {
  provider: Provider;
  model: string;
  prompt: string;
  inputImages?: ImageInput[];
  format?: "glb";
}

export interface MeshGenerateOutput {
  data: string;
  format?: "glb";
  raw?: unknown;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export interface TranscribeInput {
  provider: Provider;
  model: string;
  audio: AudioInput;
  language?: string;
  prompt?: string;
  temperature?: number;
  metadata?: Record<string, string>;
}

export interface TranscribeOutput {
  text?: string;
  language?: string;
  duration?: number;
  segments?: TranscriptSegment[];
  raw?: unknown;
}

export type StreamChunk =
  | { type: "delta"; textDelta: string }
  | { type: "tool_call"; call: ToolCall; delta?: string }
  | { type: "message_end"; usage?: Usage; finishReason?: string; cost?: CostBreakdown }
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
  entitlement?: EntitlementContext;
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

export interface KitErrorPayload {
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
  apiKey?: string;
  apiKeys?: string[];
  baseURL?: string;
  organization?: string;
  defaultUseResponses?: boolean;
}

export interface OllamaProviderConfig {
  apiKey?: string;
  baseURL?: string;
  defaultUseResponses?: boolean;
}

export interface AnthropicProviderConfig {
  apiKey?: string;
  apiKeys?: string[];
  baseURL?: string;
  version?: string;
}

export interface XAIProviderConfig {
  apiKey?: string;
  apiKeys?: string[];
  baseURL?: string;
  /**
   * "openai" uses the OpenAI-compatible surfaces; "anthropic" routes to Messages API.
   * Default: "openai".
   */
  compatibilityMode?: "openai" | "anthropic";
}

export interface GoogleProviderConfig {
  apiKey?: string;
  apiKeys?: string[];
  baseURL?: string;
}

export type ProviderConfigs = {
  [Provider.OpenAI]: OpenAIProviderConfig;
  [Provider.Anthropic]: AnthropicProviderConfig;
  [Provider.XAI]: XAIProviderConfig;
  [Provider.Google]: GoogleProviderConfig;
  [Provider.Ollama]: OllamaProviderConfig;
};

export type AnyProviderConfig =
  ProviderConfigs[keyof ProviderConfigs];

export interface KitConfig {
  providers: ProviderMap<AnyProviderConfig>;
  httpClient?: FetchLike;
  registry?: {
    ttlMs?: number;
  };
  adapters?: AdapterMap;
  adapterFactory?: AdapterFactory;
}

export type AdapterFactory = (
  provider: Provider,
  entitlement?: EntitlementContext,
) => ProviderAdapter | undefined;

export interface Kit {
  listModels(params?: ListModelsParams): Promise<ModelMetadata[]>;
  listModelRecords(params?: ListModelsParams): Promise<ModelRecord[]>;
  generate(input: GenerateInput): Promise<GenerateOutput>;
  generateWithContext(
    entitlement: EntitlementContext | undefined,
    input: GenerateInput,
  ): Promise<GenerateOutput>;
  generateImage(input: ImageGenerateInput): Promise<ImageGenerateOutput>;
  generateMesh(input: MeshGenerateInput): Promise<MeshGenerateOutput>;
  transcribe(input: TranscribeInput): Promise<TranscribeOutput>;
  streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk>;
  streamGenerateWithContext(
    entitlement: EntitlementContext | undefined,
    input: GenerateInput,
  ): AsyncIterable<StreamChunk>;
}
