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
}
