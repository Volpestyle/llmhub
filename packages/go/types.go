package llmhub

type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderXAI       Provider = "xai"
	ProviderGoogle    Provider = "google"
)

type ModelCapabilities struct {
	Text             bool `json:"text"`
	Vision           bool `json:"vision"`
	ToolUse          bool `json:"tool_use"`
	StructuredOutput bool `json:"structured_output"`
	Reasoning        bool `json:"reasoning"`
}

type TokenPrices struct {
	Input  float64 `json:"input"`
	Output float64 `json:"output"`
}

type ModelMetadata struct {
	ID            string            `json:"id"`
	DisplayName   string            `json:"displayName"`
	Provider      Provider          `json:"provider"`
	Family        string            `json:"family,omitempty"`
	Capabilities  ModelCapabilities `json:"capabilities"`
	ContextWindow int               `json:"contextWindow,omitempty"`
	TokenPrices   *TokenPrices      `json:"tokenPrices,omitempty"`
	Deprecated    bool              `json:"deprecated,omitempty"`
	InPreview     bool              `json:"inPreview,omitempty"`
}

type EntitlementContext struct {
	Provider         Provider `json:"provider,omitempty"`
	APIKey           string   `json:"apiKey,omitempty"`
	APIKeyFingerprint string  `json:"apiKeyFingerprint,omitempty"`
	AccountID        string   `json:"accountId,omitempty"`
	Region           string   `json:"region,omitempty"`
	Environment      string   `json:"environment,omitempty"`
	TenantID         string   `json:"tenantId,omitempty"`
	UserID           string   `json:"userId,omitempty"`
}

type ModelModalities struct {
	Text     bool `json:"text"`
	Vision   bool `json:"vision,omitempty"`
	AudioIn  bool `json:"audioIn,omitempty"`
	AudioOut bool `json:"audioOut,omitempty"`
	ImageOut bool `json:"imageOut,omitempty"`
}

type ModelFeatures struct {
	Tools      bool `json:"tools,omitempty"`
	JSONMode   bool `json:"jsonMode,omitempty"`
	JSONSchema bool `json:"jsonSchema,omitempty"`
	Streaming  bool `json:"streaming,omitempty"`
	Batch      bool `json:"batch,omitempty"`
}

type ModelLimits struct {
	ContextTokens   int `json:"contextTokens,omitempty"`
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
}

type ModelPricing struct {
	Currency       string             `json:"currency"`
	InputPer1M     float64            `json:"inputPer1M,omitempty"`
	CachedInputPer1M float64           `json:"cachedInputPer1M,omitempty"`
	OutputPer1M    float64            `json:"outputPer1M,omitempty"`
	Extras         map[string]float64 `json:"extras,omitempty"`
	EffectiveAsOf  string             `json:"effectiveAsOf,omitempty"`
	Source         string             `json:"source,omitempty"`
}

type AvailabilityConfidence string

const (
	AvailabilityListed  AvailabilityConfidence = "listed"
	AvailabilityInferred AvailabilityConfidence = "inferred"
	AvailabilityLearned AvailabilityConfidence = "learned"
)

type ModelAvailability struct {
	Entitled      bool                  `json:"entitled"`
	LastVerifiedAt string               `json:"lastVerifiedAt,omitempty"`
	Confidence    AvailabilityConfidence `json:"confidence,omitempty"`
	Reason        string                `json:"reason,omitempty"`
}

type ModelRecord struct {
	ID              string           `json:"id"`
	Provider         Provider         `json:"provider"`
	ProviderModelID  string           `json:"providerModelId"`
	DisplayName      string           `json:"displayName,omitempty"`
	Modalities       ModelModalities  `json:"modalities"`
	Features         ModelFeatures    `json:"features"`
	Limits           *ModelLimits     `json:"limits,omitempty"`
	Tags             []string         `json:"tags,omitempty"`
	Pricing          *ModelPricing    `json:"pricing,omitempty"`
	Availability     ModelAvailability `json:"availability"`
}

type ContentPart struct {
	Type  string        `json:"type"`
	Text  string        `json:"text,omitempty"`
	Image *ImageContent `json:"image,omitempty"`
}

type ImageContent struct {
	URL       string `json:"url,omitempty"`
	Base64    string `json:"base64,omitempty"`
	MediaType string `json:"mediaType,omitempty"`
}

type Message struct {
	Role       string        `json:"role"`
	Content    []ContentPart `json:"content"`
	ToolCallID string        `json:"toolCallId,omitempty"`
	Name       string        `json:"name,omitempty"`
}

type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type ToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type JsonSchemaFormat struct {
	Name   string                 `json:"name"`
	Schema map[string]interface{} `json:"schema"`
	Strict bool                   `json:"strict"`
}

type ResponseFormat struct {
	Type       string            `json:"type"`
	JsonSchema *JsonSchemaFormat `json:"jsonSchema,omitempty"`
}

type GenerateInput struct {
	Provider       Provider          `json:"provider"`
	Model          string            `json:"model"`
	Messages       []Message         `json:"messages"`
	Tools          []ToolDefinition  `json:"tools,omitempty"`
	ToolChoice     *ToolChoice       `json:"toolChoice,omitempty"`
	ResponseFormat *ResponseFormat   `json:"responseFormat,omitempty"`
	Temperature    *float64          `json:"temperature,omitempty"`
	TopP           *float64          `json:"topP,omitempty"`
	MaxTokens      *int              `json:"maxTokens,omitempty"`
	Stream         bool              `json:"stream,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

type ToolCall struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	ArgumentsJSON string `json:"argumentsJson"`
}

type Usage struct {
	InputTokens  int `json:"inputTokens,omitempty"`
	OutputTokens int `json:"outputTokens,omitempty"`
	TotalTokens  int `json:"totalTokens,omitempty"`
}

type GenerateOutput struct {
	Text         string      `json:"text,omitempty"`
	ToolCalls    []ToolCall  `json:"toolCalls,omitempty"`
	Usage        *Usage      `json:"usage,omitempty"`
	FinishReason string      `json:"finishReason,omitempty"`
	Raw          interface{} `json:"raw,omitempty"`
}

type StreamChunkType string

const (
	StreamChunkDelta      StreamChunkType = "delta"
	StreamChunkToolCall   StreamChunkType = "tool_call"
	StreamChunkMessageEnd StreamChunkType = "message_end"
	StreamChunkError      StreamChunkType = "error"
)

type ChunkError struct {
	Kind         string `json:"kind"`
	Message      string `json:"message"`
	UpstreamCode string `json:"upstreamCode,omitempty"`
	RequestID    string `json:"requestId,omitempty"`
}

type StreamChunk struct {
	Type         StreamChunkType `json:"type"`
	TextDelta    string          `json:"textDelta,omitempty"`
	Call         *ToolCall       `json:"call,omitempty"`
	Delta        string          `json:"delta,omitempty"`
	Usage        *Usage          `json:"usage,omitempty"`
	FinishReason string          `json:"finishReason,omitempty"`
	Error        *ChunkError     `json:"error,omitempty"`
}

type ListModelsOptions struct {
	Providers []Provider
	Refresh   bool
	Entitlement *EntitlementContext
}
