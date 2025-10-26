package llmhub

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

type Config struct {
	OpenAI      *OpenAIConfig
	Anthropic   *AnthropicConfig
	XAI         *XAIConfig
	Google      *GoogleConfig
	HTTPClient  *http.Client
	RegistryTTL time.Duration
}

type OpenAIConfig struct {
	APIKey              string
	BaseURL             string
	Organization        string
	DefaultUseResponses bool
}

type AnthropicConfig struct {
	APIKey  string
	BaseURL string
	Version string
}

type XAIConfig struct {
	APIKey            string
	BaseURL           string
	CompatibilityMode string
}

type GoogleConfig struct {
	APIKey  string
	BaseURL string
}

type adapter interface {
	ListModels(ctx context.Context) ([]ModelMetadata, error)
	Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error)
	Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error)
}

type Hub struct {
	adapters map[Provider]adapter
	registry *modelRegistry
}

func New(config Config) (*Hub, error) {
	adapters := make(map[Provider]adapter)
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	if config.OpenAI != nil {
		if config.OpenAI.APIKey == "" {
			return nil, fmt.Errorf("openai api key is required")
		}
		adapters[ProviderOpenAI] = newOpenAIAdapter(config.OpenAI, client, ProviderOpenAI)
	}
	if config.Anthropic != nil {
		if config.Anthropic.APIKey == "" {
			return nil, fmt.Errorf("anthropic api key is required")
		}
		adapters[ProviderAnthropic] = newAnthropicAdapter(config.Anthropic, client, ProviderAnthropic)
	}
	if config.XAI != nil {
		if config.XAI.APIKey == "" {
			return nil, fmt.Errorf("xai api key is required")
		}
		adapters[ProviderXAI] = newXAIAdapter(config.XAI, client)
	}
	if config.Google != nil {
		if config.Google.APIKey == "" {
			return nil, fmt.Errorf("google api key is required")
		}
		adapters[ProviderGoogle] = newGoogleAdapter(config.Google, client)
	}
	if len(adapters) == 0 {
		return nil, fmt.Errorf("at least one provider config is required")
	}
	ttl := config.RegistryTTL
	if ttl == 0 {
		ttl = 6 * time.Hour
	}
	registry := newModelRegistry(adapters, ttl)
	return &Hub{
		adapters: adapters,
		registry: registry,
	}, nil
}

func (h *Hub) ListModels(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error) {
	return h.registry.List(ctx, opts)
}

func (h *Hub) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	adapter, ok := h.adapters[in.Provider]
	if !ok {
		return GenerateOutput{}, fmt.Errorf("provider %s is not configured", in.Provider)
	}
	return adapter.Generate(ctx, in)
}

func (h *Hub) StreamGenerate(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	adapter, ok := h.adapters[in.Provider]
	if !ok {
		return nil, fmt.Errorf("provider %s is not configured", in.Provider)
	}
	return adapter.Stream(ctx, in)
}
