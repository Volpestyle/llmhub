package llmhub

import (
	"context"
	"sort"
	"sync"
	"time"
)

type registryEntry struct {
	expires time.Time
	data    []ModelMetadata
}

type modelRegistry struct {
	adapters map[Provider]adapter
	ttl      time.Duration
	cache    map[Provider]registryEntry
	mu       sync.RWMutex
}

func newModelRegistry(adapters map[Provider]adapter, ttl time.Duration) *modelRegistry {
	return &modelRegistry{
		adapters: adapters,
		ttl:      ttl,
		cache:    make(map[Provider]registryEntry),
	}
}

func (r *modelRegistry) List(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error) {
	providers := make([]Provider, 0, len(r.adapters))
	if opts != nil && len(opts.Providers) > 0 {
		providers = append(providers, opts.Providers...)
	} else {
		for provider := range r.adapters {
			providers = append(providers, provider)
		}
	}
	results := make([]ModelMetadata, 0)
	for _, provider := range providers {
		models, err := r.modelsForProvider(ctx, provider, opts != nil && opts.Refresh)
		if err != nil {
			return nil, err
		}
		results = append(results, models...)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Provider == results[j].Provider {
			return results[i].DisplayName < results[j].DisplayName
		}
		return results[i].Provider < results[j].Provider
	})
	return results, nil
}

func (r *modelRegistry) modelsForProvider(ctx context.Context, provider Provider, refresh bool) ([]ModelMetadata, error) {
	r.mu.RLock()
	entry, ok := r.cache[provider]
	if ok && !refresh && time.Now().Before(entry.expires) {
		r.mu.RUnlock()
		return entry.data, nil
	}
	r.mu.RUnlock()

	adapter, ok := r.adapters[provider]
	if !ok {
		return nil, &HubError{
			Kind:     ErrorValidation,
			Message:  "provider not configured",
			Provider: provider,
		}
	}
	models, err := adapter.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	r.cache[provider] = registryEntry{
		data:    models,
		expires: time.Now().Add(r.ttl),
	}
	r.mu.Unlock()
	return models, nil
}
