package aikit

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestHubGenerateContract(t *testing.T) {
	client := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			switch {
			case strings.Contains(req.URL.Host, "api.openai.com") && strings.Contains(req.URL.Path, "/responses"):
				return jsonHTTPResponse(`{"status":"completed","output":[{"content":[{"type":"output_text","text":"Unified response"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`), nil
			case strings.Contains(req.URL.Host, "api.openai.com"):
				return jsonHTTPResponse(`{"choices":[{"finish_reason":"stop","message":{"content":[{"type":"text","text":"Unified response"}],"tool_calls":[]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`), nil
			case strings.Contains(req.URL.Host, "api.anthropic.com"):
				return jsonHTTPResponse(`{"content":[{"type":"text","text":"Unified response"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`), nil
			case strings.Contains(req.URL.Host, "api.x.ai"):
				return jsonHTTPResponse(`{"choices":[{"finish_reason":"stop","message":{"content":[{"type":"text","text":"Unified response"}],"tool_calls":[]}}]}`), nil
			case strings.Contains(req.URL.Host, "generativelanguage.googleapis.com"):
				return jsonHTTPResponse(`{"candidates":[{"content":{"parts":[{"text":"Unified response"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`), nil
			default:
				t.Fatalf("unexpected request: %s", req.URL.String())
				return nil, nil
			}
		}),
	}

	hub, err := New(Config{
		OpenAI:     &OpenAIConfig{APIKey: "openai", DefaultUseResponses: true},
		Anthropic:  &AnthropicConfig{APIKey: "anthropic"},
		XAI:        &XAIConfig{APIKey: "xai", CompatibilityMode: "openai"},
		Google:     &GoogleConfig{APIKey: "google"},
		HTTPClient: client,
	})
	if err != nil {
		t.Fatalf("new hub: %v", err)
	}

	baseInput := GenerateInput{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: []ContentPart{{Type: "text", Text: "hello"}}}},
	}
	providers := []Provider{ProviderOpenAI, ProviderAnthropic, ProviderGoogle, ProviderXAI}
	for _, provider := range providers {
		input := baseInput
		input.Provider = provider
		if provider == ProviderOpenAI {
			input.ResponseFormat = &ResponseFormat{
				Type: "json_schema",
				JsonSchema: &JsonSchemaFormat{
					Name:   "demo",
					Schema: map[string]interface{}{"type": "object"},
					Strict: true,
				},
			}
		}
		output, err := hub.Generate(context.Background(), input)
		if err != nil {
			t.Fatalf("generate %s: %v", provider, err)
		}
		if output.Text != "Unified response" {
			t.Fatalf("provider %s expected normalized text, got %s", provider, output.Text)
		}
	}
}

func jsonHTTPResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewBufferString(body)),
	}
}
