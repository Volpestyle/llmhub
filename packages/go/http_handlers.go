package llmhub

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

type HubAPI interface {
	ListModels(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error)
	Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error)
	StreamGenerate(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error)
}

type ModelsHandlerOptions struct {
	Refresh bool
}

func ModelsHandler(h HubAPI, opts *ModelsHandlerOptions) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()
		providers := parseProviders(r.URL.Query().Get("providers"))
		models, err := h.ListModels(ctx, &ListModelsOptions{
			Providers: providers,
			Refresh:   opts != nil && opts.Refresh,
		})
		if err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, models)
	}
}

func GenerateHandler(h HubAPI) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var input GenerateInput
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		output, err := h.Generate(r.Context(), input)
		if err != nil {
			writeError(w, err)
			return
		}
		writeJSON(w, output)
	}
}

func GenerateSSEHandler(h HubAPI) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var input GenerateInput
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		ch, err := h.StreamGenerate(r.Context(), input)
		if err != nil {
			writeError(w, err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		encoder := json.NewEncoder(w)
		for chunk := range ch {
			w.Write([]byte("event: chunk\n"))
			w.Write([]byte("data: "))
			if err := encoder.Encode(chunk); err != nil {
				break
			}
			w.Write([]byte("\n"))
			flusher.Flush()
		}
	}
}

func parseProviders(value string) []Provider {
	if value == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	var providers []Provider
	for _, part := range parts {
		switch strings.TrimSpace(strings.ToLower(part)) {
		case string(ProviderOpenAI):
			providers = append(providers, ProviderOpenAI)
		case string(ProviderAnthropic):
			providers = append(providers, ProviderAnthropic)
		case string(ProviderGoogle):
			providers = append(providers, ProviderGoogle)
		case string(ProviderXAI):
			providers = append(providers, ProviderXAI)
		}
	}
	return providers
}

func writeJSON(w http.ResponseWriter, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, err error) {
	if hubErr, ok := err.(*HubError); ok {
		status := http.StatusInternalServerError
		switch hubErr.Kind {
		case ErrorValidation:
			status = http.StatusBadRequest
		case ErrorProviderAuth:
			status = http.StatusUnauthorized
		case ErrorProviderRateLimit:
			status = http.StatusTooManyRequests
		case ErrorProviderUnavailable:
			status = http.StatusServiceUnavailable
		}
		http.Error(w, hubErr.Message, status)
		return
	}
	http.Error(w, err.Error(), http.StatusInternalServerError)
}
