package aikit

import (
	"context"
	"sort"
	"strings"
	"sync"
	"time"
)

type registryEntry struct {
	expires   time.Time
	fetchedAt time.Time
	data      []ModelMetadata
}

type learnedEntry struct {
	expires time.Time
	reason  string
}

type registryKey struct {
	Provider    Provider
	Fingerprint string
	AccountID   string
	Region      string
	Environment string
	TenantID    string
	UserID      string
}

type learnedKey struct {
	registryKey
	ModelID string
}

type modelRegistry struct {
	adapters    map[Provider]ProviderAdapter
	factory     AdapterFactory
	ttl         time.Duration
	learnedTTL  time.Duration
	cache       map[registryKey]registryEntry
	learned     map[learnedKey]learnedEntry
	mu          sync.RWMutex
}

func newModelRegistry(adapters map[Provider]ProviderAdapter, ttl time.Duration, factory AdapterFactory) *modelRegistry {
	return &modelRegistry{
		adapters:   adapters,
		factory:    factory,
		ttl:       ttl,
		learnedTTL: 20 * time.Minute,
		cache:      make(map[registryKey]registryEntry),
		learned:    make(map[learnedKey]learnedEntry),
	}
}

func (r *modelRegistry) List(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error) {
	entries, err := r.entriesForProviders(ctx, opts)
	if err != nil {
		return nil, err
	}
	results := make([]ModelMetadata, 0)
	for _, entry := range entries {
		results = append(results, entry.data...)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Provider == results[j].Provider {
			return results[i].DisplayName < results[j].DisplayName
		}
		return results[i].Provider < results[j].Provider
	})
	return results, nil
}

func (r *modelRegistry) ListRecords(ctx context.Context, opts *ListModelsOptions) ([]ModelRecord, error) {
	entries, err := r.entriesForProviders(ctx, opts)
	if err != nil {
		return nil, err
	}
	var entitlement *EntitlementContext
	if opts != nil {
		entitlement = opts.Entitlement
	}
	results := make([]ModelRecord, 0)
	for provider, entry := range entries {
		for _, model := range entry.data {
			results = append(results, r.modelRecordFromMetadata(model, provider, entry.fetchedAt, entitlement))
		}
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Provider == results[j].Provider {
			return results[i].DisplayName < results[j].DisplayName
		}
		return results[i].Provider < results[j].Provider
	})
	return results, nil
}

func (r *modelRegistry) LearnModelUnavailable(entitlement *EntitlementContext, provider Provider, modelID string, err error) {
	reason, ok := learnReason(err)
	if !ok {
		return
	}
	key := r.learnedKey(provider, entitlement, modelID)
	r.mu.Lock()
	r.learned[key] = learnedEntry{
		expires: time.Now().Add(r.learnedTTL),
		reason:  reason,
	}
	r.mu.Unlock()
}

func (r *modelRegistry) entriesForProviders(ctx context.Context, opts *ListModelsOptions) (map[Provider]registryEntry, error) {
	providers := r.resolveProviders(opts)
	results := make(map[Provider]registryEntry, len(providers))
	for _, provider := range providers {
		entry, err := r.modelsForProvider(ctx, provider, opts)
		if err != nil {
			return nil, err
		}
		results[provider] = entry
	}
	return results, nil
}

func (r *modelRegistry) resolveProviders(opts *ListModelsOptions) []Provider {
	if opts != nil && len(opts.Providers) > 0 {
		return opts.Providers
	}
	if opts != nil && opts.Entitlement != nil && opts.Entitlement.Provider != "" {
		return []Provider{opts.Entitlement.Provider}
	}
	providers := make([]Provider, 0, len(r.adapters))
	for provider := range r.adapters {
		providers = append(providers, provider)
	}
	return providers
}

func (r *modelRegistry) modelsForProvider(ctx context.Context, provider Provider, opts *ListModelsOptions) (registryEntry, error) {
	refresh := opts != nil && opts.Refresh
	var entitlement *EntitlementContext
	if opts != nil {
		entitlement = opts.Entitlement
	}
	key := r.registryKey(provider, entitlement)
	if !refresh {
		if cached, ok := r.cached(key); ok {
			return cached, nil
		}
	}
	entry, err := r.fetchAndCache(ctx, provider, entitlement, key)
	if err != nil {
		if cached, ok := r.cached(key); ok {
			return cached, nil
		}
		return registryEntry{}, err
	}
	return entry, nil
}

func (r *modelRegistry) cached(key registryKey) (registryEntry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.cache[key]
	if !ok || time.Now().After(entry.expires) {
		return registryEntry{}, false
	}
	return entry, true
}

func (r *modelRegistry) fetchAndCache(ctx context.Context, provider Provider, entitlement *EntitlementContext, key registryKey) (registryEntry, error) {
	adapter, err := r.adapterFor(provider, entitlement)
	if err != nil {
		return registryEntry{}, err
	}
	models, err := adapter.ListModels(ctx)
	if err != nil {
		return registryEntry{}, err
	}
	for idx, model := range models {
		models[idx] = applyCuratedMetadata(model)
	}
	now := time.Now()
	entry := registryEntry{
		data:      models,
		expires:   now.Add(r.ttl),
		fetchedAt: now,
	}
	r.mu.Lock()
	r.cache[key] = entry
	r.mu.Unlock()
	return entry, nil
}

func (r *modelRegistry) adapterFor(provider Provider, entitlement *EntitlementContext) (ProviderAdapter, error) {
	if r.factory != nil {
		return r.factory(provider, entitlement)
	}
	if adapter, ok := r.adapters[provider]; ok {
		return adapter, nil
	}
	return nil, &KitError{
		Kind:     ErrorValidation,
		Message:  "provider not configured",
		Provider: provider,
	}
}

func (r *modelRegistry) registryKey(provider Provider, entitlement *EntitlementContext) registryKey {
	key := registryKey{
		Provider:    provider,
		Fingerprint: "default",
	}
	if entitlement == nil {
		return key
	}
	key.AccountID = strings.TrimSpace(entitlement.AccountID)
	key.Region = strings.TrimSpace(entitlement.Region)
	key.Environment = strings.TrimSpace(entitlement.Environment)
	key.TenantID = strings.TrimSpace(entitlement.TenantID)
	key.UserID = strings.TrimSpace(entitlement.UserID)
	fingerprint := strings.TrimSpace(entitlement.APIKeyFingerprint)
	if fingerprint == "" && strings.TrimSpace(entitlement.APIKey) != "" {
		fingerprint = FingerprintAPIKey(entitlement.APIKey)
	}
	if fingerprint != "" {
		key.Fingerprint = fingerprint
	}
	return key
}

func (r *modelRegistry) learnedKey(provider Provider, entitlement *EntitlementContext, modelID string) learnedKey {
	return learnedKey{
		registryKey: r.registryKey(provider, entitlement),
		ModelID:     modelID,
	}
}

func (r *modelRegistry) learnedStatus(provider Provider, entitlement *EntitlementContext, modelID string) (learnedEntry, bool) {
	key := r.learnedKey(provider, entitlement, modelID)
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.learned[key]
	if !ok {
		return learnedEntry{}, false
	}
	if time.Now().After(entry.expires) {
		delete(r.learned, key)
		return learnedEntry{}, false
	}
	return entry, true
}

func (r *modelRegistry) modelRecordFromMetadata(model ModelMetadata, provider Provider, verifiedAt time.Time, entitlement *EntitlementContext) ModelRecord {
	modalities := ModelModalities{
		Text:   model.Capabilities.Text,
		Vision: model.Capabilities.Vision,
		AudioIn: model.Capabilities.AudioIn,
		AudioOut: model.Capabilities.AudioOut,
		ImageOut: model.Capabilities.Image,
		VideoIn: model.Capabilities.VideoIn,
		VideoOut: model.Capabilities.Video,
	}
	features := ModelFeatures{
		Tools:      model.Capabilities.ToolUse,
		JSONMode:   model.Capabilities.StructuredOutput,
		JSONSchema: model.Capabilities.StructuredOutput,
		Streaming:  true,
	}
	var limits *ModelLimits
	if model.ContextWindow > 0 {
		limits = &ModelLimits{ContextTokens: model.ContextWindow}
	}
	var pricing *ModelPricing
	if model.TokenPrices != nil {
		pricing = &ModelPricing{
			Currency:    "USD",
			InputPer1M:  model.TokenPrices.Input,
			OutputPer1M: model.TokenPrices.Output,
			Source:      "config",
		}
	}
	tags := []string{}
	if model.InPreview {
		tags = append(tags, "preview")
	}
	if model.Deprecated {
		tags = append(tags, "deprecated")
	}
	availability := ModelAvailability{
		Entitled:   true,
		Confidence: AvailabilityListed,
	}
	if !verifiedAt.IsZero() {
		availability.LastVerifiedAt = verifiedAt.UTC().Format(time.RFC3339)
	}
	if learned, ok := r.learnedStatus(provider, entitlement, model.ID); ok {
		availability.Entitled = false
		availability.Confidence = AvailabilityLearned
		availability.Reason = learned.reason
	}
	recordID := string(provider) + ":" + model.ID
	return ModelRecord{
		ID:             recordID,
		Provider:       provider,
		ProviderModelID: model.ID,
		DisplayName:    model.DisplayName,
		Modalities:     modalities,
		Features:       features,
		Limits:         limits,
		Tags:           tags,
		Pricing:        pricing,
		Availability:   availability,
	}
}

func learnReason(err error) (string, bool) {
	if err == nil {
		return "", false
	}
	if kitErr, ok := err.(*KitError); ok {
		if kitErr.Kind == ErrorProviderNotFound || kitErr.Kind == ErrorValidation {
			return kitErr.Message, true
		}
		if kitErr.UpstreamStatus == 400 || kitErr.UpstreamStatus == 404 || kitErr.UpstreamStatus == 403 {
			return kitErr.Message, true
		}
	}
	return "", false
}
