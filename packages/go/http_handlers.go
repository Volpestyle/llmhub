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
		query := r.URL.Query()
		providers := parseProviders(query.Get("providers"))
		refresh := shouldRefresh(query["refresh"])
		if opts != nil && opts.Refresh {
			refresh = true
		}
		models, err := h.ListModels(ctx, &ListModelsOptions{
			Providers: providers,
			Refresh:   refresh,
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
		w.Write([]byte("event: done\n"))
		w.Write([]byte("data: {\"ok\":true}\n\n"))
		flusher.Flush()
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

func shouldRefresh(values []string) bool {
	if len(values) == 0 {
		return false
	}
	v := strings.ToLower(values[0])
	return v == "" || v == "1" || v == "true" || v == "yes" || v == "on"
}

func writeJSON(w http.ResponseWriter, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(payload)
}

type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

func writeError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	resp := errorResponse{
		Error: errorDetail{
			Kind:    string(ErrorUnknown),
			Message: err.Error(),
		},
	}
	if hubErr, ok := err.(*HubError); ok {
		resp.Error.Kind = string(hubErr.Kind)
		resp.Error.Message = hubErr.Message
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
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(resp)
}
