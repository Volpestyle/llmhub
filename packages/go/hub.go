package aikit

import (
	"context"
	"fmt"
	"net/http"
	"strings"
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

type adapter interface {
	ListModels(ctx context.Context) ([]ModelMetadata, error)
	Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error)
	GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error)
	GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error)
	Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error)
}

type adapterFactory func(provider Provider, entitlement *EntitlementContext) (adapter, error)

type Hub struct {
	adapters map[Provider]adapter
	registry *modelRegistry
	factory  adapterFactory
	keyPools map[Provider]*keyPool
}

func New(config Config) (*Hub, error) {
	adapters := make(map[Provider]adapter)
	keyPools := make(map[Provider]*keyPool)
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	if config.OpenAI != nil {
		keys := normalizeKeys(config.OpenAI.APIKey, config.OpenAI.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("openai api key is required")
		}
		cfg := *config.OpenAI
		cfg.APIKey = keys[0]
		adapters[ProviderOpenAI] = newOpenAIAdapter(&cfg, client, ProviderOpenAI)
		keyPools[ProviderOpenAI] = newKeyPool(keys)
	}
	if config.Anthropic != nil {
		keys := normalizeKeys(config.Anthropic.APIKey, config.Anthropic.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("anthropic api key is required")
		}
		cfg := *config.Anthropic
		cfg.APIKey = keys[0]
		adapters[ProviderAnthropic] = newAnthropicAdapter(&cfg, client, ProviderAnthropic)
		keyPools[ProviderAnthropic] = newKeyPool(keys)
	}
	if config.XAI != nil {
		keys := normalizeKeys(config.XAI.APIKey, config.XAI.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("xai api key is required")
		}
		cfg := *config.XAI
		cfg.APIKey = keys[0]
		adapters[ProviderXAI] = newXAIAdapter(&cfg, client)
		keyPools[ProviderXAI] = newKeyPool(keys)
	}
	if config.Google != nil {
		keys := normalizeKeys(config.Google.APIKey, config.Google.APIKeys)
		if len(keys) == 0 {
			return nil, fmt.Errorf("google api key is required")
		}
		cfg := *config.Google
		cfg.APIKey = keys[0]
		adapters[ProviderGoogle] = newGoogleAdapter(&cfg, client)
		keyPools[ProviderGoogle] = newKeyPool(keys)
	}
	if catalog := newCatalogAdapter(); catalog != nil {
		adapters[ProviderCatalog] = catalog
	}
	if len(adapters) == 0 {
		return nil, fmt.Errorf("at least one provider config is required")
	}
	ttl := config.RegistryTTL
	if ttl == 0 {
		ttl = 30 * time.Minute
	}
	factory := newAdapterFactory(config, client, adapters)
	registry := newModelRegistry(adapters, ttl, factory)
	return &Hub{
		adapters: adapters,
		registry: registry,
		factory:  factory,
		keyPools: keyPools,
	}, nil
}

func (h *Hub) ListModels(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error) {
	return h.registry.List(ctx, opts)
}

func (h *Hub) ListModelRecords(ctx context.Context, opts *ListModelsOptions) ([]ModelRecord, error) {
	return h.registry.ListRecords(ctx, opts)
}

func (h *Hub) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
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

func (h *Hub) GenerateWithContext(ctx context.Context, entitlement *EntitlementContext, in GenerateInput) (GenerateOutput, error) {
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

func (h *Hub) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
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

func (h *Hub) GenerateImageWithContext(ctx context.Context, entitlement *EntitlementContext, in ImageGenerateInput) (ImageGenerateOutput, error) {
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

func (h *Hub) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
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

func (h *Hub) GenerateMeshWithContext(ctx context.Context, entitlement *EntitlementContext, in MeshGenerateInput) (MeshGenerateOutput, error) {
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

func (h *Hub) StreamGenerate(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
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

func (h *Hub) StreamGenerateWithContext(ctx context.Context, entitlement *EntitlementContext, in GenerateInput) (<-chan StreamChunk, error) {
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

func (h *Hub) entitlementForProvider(provider Provider) *EntitlementContext {
	pool := h.keyPools[provider]
	if pool == nil {
		return nil
	}
	key := pool.Next()
	if strings.TrimSpace(key) == "" {
		return nil
	}
	return &EntitlementContext{
		Provider:         provider,
		APIKey:           key,
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

func newAdapterFactory(config Config, client *http.Client, adapters map[Provider]adapter) adapterFactory {
	return func(provider Provider, entitlement *EntitlementContext) (adapter, error) {
		if provider == ProviderCatalog {
			if adapter, ok := adapters[provider]; ok {
				return adapter, nil
			}
			return nil, fmt.Errorf("provider %s is not configured", provider)
		}
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
		default:
			return nil, fmt.Errorf("provider %s is not configured", provider)
		}
	}
}
