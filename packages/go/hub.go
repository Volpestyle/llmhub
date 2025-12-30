package aikit

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type Config struct {
	OpenAI         *OpenAIConfig
	Anthropic      *AnthropicConfig
	XAI            *XAIConfig
	Google         *GoogleConfig
	Ollama         *OllamaConfig
	HTTPClient     *http.Client
	RegistryTTL    time.Duration
	Adapters       map[Provider]ProviderAdapter
	AdapterFactory AdapterFactory
}

type OpenAIConfig struct {
	APIKey              string
	APIKeys             []string
	BaseURL             string
	Organization        string
	DefaultUseResponses bool
}

type AnthropicConfig struct {
	APIKey  string
	APIKeys []string
	BaseURL string
	Version string
}

type XAIConfig struct {
	APIKey            string
	APIKeys           []string
	BaseURL           string
	CompatibilityMode string
}

type GoogleConfig struct {
	APIKey  string
	APIKeys []string
	BaseURL string
}

type OllamaConfig struct {
	APIKey              string
	APIKeys             []string
	BaseURL             string
	DefaultUseResponses bool
}

type ProviderAdapter interface {
	ListModels(ctx context.Context) ([]ModelMetadata, error)
	Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error)
	GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error)
	GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error)
	Transcribe(ctx context.Context, in TranscribeInput) (TranscribeOutput, error)
	Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error)
}

type AdapterFactory func(provider Provider, entitlement *EntitlementContext) (ProviderAdapter, error)

type Kit struct {
	adapters map[Provider]ProviderAdapter
	registry *modelRegistry
	factory  AdapterFactory
	keyPools map[Provider]*keyPool
}

func New(config Config) (*Kit, error) {
	adapters := make(map[Provider]ProviderAdapter)
	keyPools := make(map[Provider]*keyPool)
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	for provider, adapter := range config.Adapters {
		if adapter != nil {
			adapters[provider] = adapter
		}
	}

	if config.OpenAI != nil && adapters[ProviderOpenAI] == nil {
		keys := normalizeKeys(config.OpenAI.APIKey, config.OpenAI.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("openai api key is required")
		}
		cfg := *config.OpenAI
		cfg.APIKey = keys[0]
		adapters[ProviderOpenAI] = newOpenAIAdapter(&cfg, client, ProviderOpenAI)
		keyPools[ProviderOpenAI] = newKeyPool(keys)
	}
	if config.Anthropic != nil && adapters[ProviderAnthropic] == nil {
		keys := normalizeKeys(config.Anthropic.APIKey, config.Anthropic.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("anthropic api key is required")
		}
		cfg := *config.Anthropic
		cfg.APIKey = keys[0]
		adapters[ProviderAnthropic] = newAnthropicAdapter(&cfg, client, ProviderAnthropic)
		keyPools[ProviderAnthropic] = newKeyPool(keys)
	}
	if config.XAI != nil && adapters[ProviderXAI] == nil {
		keys := normalizeKeys(config.XAI.APIKey, config.XAI.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("xai api key is required")
		}
		cfg := *config.XAI
		cfg.APIKey = keys[0]
		adapters[ProviderXAI] = newXAIAdapter(&cfg, client)
		keyPools[ProviderXAI] = newKeyPool(keys)
	}
	if config.Google != nil && adapters[ProviderGoogle] == nil {
		keys := normalizeKeys(config.Google.APIKey, config.Google.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("google api key is required")
		}
		cfg := *config.Google
		cfg.APIKey = keys[0]
		adapters[ProviderGoogle] = newGoogleAdapter(&cfg, client)
		keyPools[ProviderGoogle] = newKeyPool(keys)
	}
	if config.Ollama != nil && adapters[ProviderOllama] == nil {
		cfg := *config.Ollama
		keys := normalizeKeys(cfg.APIKey, cfg.APIKeys)
		if len(keys) > 0 {
			cfg.APIKey = keys[0]
			keyPools[ProviderOllama] = newKeyPool(keys)
		}
		adapters[ProviderOllama] = newOllamaAdapter(&cfg, client)
	}
	if len(adapters) == 0 && config.AdapterFactory == nil {
		return nil, fmt.Errorf("at least one provider config or adapter is required")
	}
	ttl := config.RegistryTTL
	if ttl == 0 {
		ttl = 30 * time.Minute
	}
	factory := config.AdapterFactory
	if factory == nil {
		factory = newAdapterFactory(config, client, adapters)
	}
	registry := newModelRegistry(adapters, ttl, factory)
	return &Kit{
		adapters: adapters,
		registry: registry,
		factory:  factory,
		keyPools: keyPools,
	}, nil
}

func (h *Kit) ListModels(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error) {
	models, err := h.registry.List(ctx, opts)
	if err != nil {
		return nil, err
	}
	annotateModelAvailability(models, h.adapters)
	return models, nil
}

func (h *Kit) ListModelRecords(ctx context.Context, opts *ListModelsOptions) ([]ModelRecord, error) {
	return h.registry.ListRecords(ctx, opts)
}

func annotateModelAvailability(models []ModelMetadata, adapters map[Provider]ProviderAdapter) {
	if len(models) == 0 {
		return
	}
	availableProviders := make(map[Provider]struct{}, len(adapters)+1)
	for provider := range adapters {
		availableProviders[provider] = struct{}{}
	}
	for idx, model := range models {
		_, ok := availableProviders[model.Provider]
		models[idx].Available = ok
	}
}

func (h *Kit) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	if entitlement := h.entitlementForProvider(in.Provider); entitlement != nil {
		return h.GenerateWithContext(ctx, entitlement, in)
	}
	adapter, ok := h.adapters[in.Provider]
	if !ok {
		return GenerateOutput{}, fmt.Errorf("provider %s is not configured", in.Provider)
	}
	output, err := adapter.Generate(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(nil, in.Provider, in.Model, err)
		return GenerateOutput{}, err
	}
	return attachCost(in, output), nil
}

func (h *Kit) GenerateWithContext(ctx context.Context, entitlement *EntitlementContext, in GenerateInput) (GenerateOutput, error) {
	adapter, err := h.factory(in.Provider, entitlement)
	if err != nil {
		return GenerateOutput{}, err
	}
	output, err := adapter.Generate(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(entitlement, in.Provider, in.Model, err)
	}
	return attachCost(in, output), err
}

func (h *Kit) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
	if entitlement := h.entitlementForProvider(in.Provider); entitlement != nil {
		return h.GenerateImageWithContext(ctx, entitlement, in)
	}
	adapter, ok := h.adapters[in.Provider]
	if !ok {
		return ImageGenerateOutput{}, fmt.Errorf("provider %s is not configured", in.Provider)
	}
	output, err := adapter.GenerateImage(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(nil, in.Provider, in.Model, err)
		return ImageGenerateOutput{}, err
	}
	return output, nil
}

func (h *Kit) GenerateImageWithContext(ctx context.Context, entitlement *EntitlementContext, in ImageGenerateInput) (ImageGenerateOutput, error) {
	adapter, err := h.factory(in.Provider, entitlement)
	if err != nil {
		return ImageGenerateOutput{}, err
	}
	output, err := adapter.GenerateImage(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(entitlement, in.Provider, in.Model, err)
	}
	return output, err
}

func (h *Kit) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
	if entitlement := h.entitlementForProvider(in.Provider); entitlement != nil {
		return h.GenerateMeshWithContext(ctx, entitlement, in)
	}
	adapter, ok := h.adapters[in.Provider]
	if !ok {
		return MeshGenerateOutput{}, fmt.Errorf("provider %s is not configured", in.Provider)
	}
	output, err := adapter.GenerateMesh(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(nil, in.Provider, in.Model, err)
		return MeshGenerateOutput{}, err
	}
	return output, nil
}

func (h *Kit) Transcribe(ctx context.Context, in TranscribeInput) (TranscribeOutput, error) {
	if entitlement := h.entitlementForProvider(in.Provider); entitlement != nil {
		return h.TranscribeWithContext(ctx, entitlement, in)
	}
	adapter, ok := h.adapters[in.Provider]
	if !ok {
		return TranscribeOutput{}, fmt.Errorf("provider %s is not configured", in.Provider)
	}
	output, err := adapter.Transcribe(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(nil, in.Provider, in.Model, err)
		return TranscribeOutput{}, err
	}
	return output, nil
}

func (h *Kit) GenerateMeshWithContext(ctx context.Context, entitlement *EntitlementContext, in MeshGenerateInput) (MeshGenerateOutput, error) {
	adapter, err := h.factory(in.Provider, entitlement)
	if err != nil {
		return MeshGenerateOutput{}, err
	}
	output, err := adapter.GenerateMesh(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(entitlement, in.Provider, in.Model, err)
	}
	return output, err
}

func (h *Kit) TranscribeWithContext(ctx context.Context, entitlement *EntitlementContext, in TranscribeInput) (TranscribeOutput, error) {
	adapter, err := h.factory(in.Provider, entitlement)
	if err != nil {
		return TranscribeOutput{}, err
	}
	output, err := adapter.Transcribe(ctx, in)
	if err != nil {
		h.registry.LearnModelUnavailable(entitlement, in.Provider, in.Model, err)
	}
	return output, err
}

func (h *Kit) StreamGenerate(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	if entitlement := h.entitlementForProvider(in.Provider); entitlement != nil {
		return h.StreamGenerateWithContext(ctx, entitlement, in)
	}
	adapter, ok := h.adapters[in.Provider]
	if !ok {
		return nil, fmt.Errorf("provider %s is not configured", in.Provider)
	}
	stream, err := adapter.Stream(ctx, in)
	if err != nil {
		return nil, err
	}
	return attachCostToStream(in.Provider, in.Model, stream), nil
}

func (h *Kit) StreamGenerateWithContext(ctx context.Context, entitlement *EntitlementContext, in GenerateInput) (<-chan StreamChunk, error) {
	adapter, err := h.factory(in.Provider, entitlement)
	if err != nil {
		return nil, err
	}
	stream, err := adapter.Stream(ctx, in)
	if err != nil {
		return nil, err
	}
	return attachCostToStream(in.Provider, in.Model, stream), nil
}

func (h *Kit) entitlementForProvider(provider Provider) *EntitlementContext {
	pool := h.keyPools[provider]
	if pool == nil {
		return nil
	}
	key := pool.Next()
	if strings.TrimSpace(key) == "" {
		return nil
	}
	return &EntitlementContext{
		Provider:          provider,
		APIKey:            key,
		APIKeyFingerprint: FingerprintAPIKey(key),
	}
}

func attachCost(in GenerateInput, output GenerateOutput) GenerateOutput {
	cost := estimateCost(in.Provider, in.Model, output.Usage)
	if cost != nil {
		output.Cost = cost
	}
	return output
}

func attachCostToStream(provider Provider, model string, stream <-chan StreamChunk) <-chan StreamChunk {
	out := make(chan StreamChunk)
	go func() {
		defer close(out)
		for chunk := range stream {
			if chunk.Type == StreamChunkMessageEnd {
				cost := estimateCost(provider, model, chunk.Usage)
				if cost != nil {
					chunk.Cost = cost
				}
			}
			out <- chunk
		}
	}()
	return out
}

func newAdapterFactory(config Config, client *http.Client, adapters map[Provider]ProviderAdapter) AdapterFactory {
	return func(provider Provider, entitlement *EntitlementContext) (ProviderAdapter, error) {
		if entitlement == nil || strings.TrimSpace(entitlement.APIKey) == "" {
			if adapter, ok := adapters[provider]; ok {
				return adapter, nil
			}
			return nil, fmt.Errorf("provider %s is not configured", provider)
		}
		apiKey := strings.TrimSpace(entitlement.APIKey)
		switch provider {
		case ProviderOpenAI:
			if config.OpenAI == nil {
				return nil, fmt.Errorf("openai config is not available")
			}
			cfg := *config.OpenAI
			cfg.APIKey = apiKey
			return newOpenAIAdapter(&cfg, client, ProviderOpenAI), nil
		case ProviderAnthropic:
			if config.Anthropic == nil {
				return nil, fmt.Errorf("anthropic config is not available")
			}
			cfg := *config.Anthropic
			cfg.APIKey = apiKey
			return newAnthropicAdapter(&cfg, client, ProviderAnthropic), nil
		case ProviderXAI:
			if config.XAI == nil {
				return nil, fmt.Errorf("xai config is not available")
			}
			cfg := *config.XAI
			cfg.APIKey = apiKey
			return newXAIAdapter(&cfg, client), nil
		case ProviderGoogle:
			if config.Google == nil {
				return nil, fmt.Errorf("google config is not available")
			}
			cfg := *config.Google
			cfg.APIKey = apiKey
			return newGoogleAdapter(&cfg, client), nil
		case ProviderOllama:
			if config.Ollama == nil {
				return nil, fmt.Errorf("ollama config is not available")
			}
			cfg := *config.Ollama
			cfg.APIKey = apiKey
			return newOllamaAdapter(&cfg, client), nil
		default:
			return nil, fmt.Errorf("provider %s is not configured", provider)
		}
	}
}
