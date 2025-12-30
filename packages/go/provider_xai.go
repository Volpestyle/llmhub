package aikit

import (
	"context"
	"net/http"
)

const xaiCompatibilityKey = "xai:compatibility"

type xaiAdapter struct {
	openai    ProviderAdapter
	anthropic ProviderAdapter
	mode      string
}

func newXAIAdapter(cfg *XAIConfig, client *http.Client) ProviderAdapter {
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.x.ai"
	}
	openaiCfg := &OpenAIConfig{
		APIKey:  cfg.APIKey,
		BaseURL: base,
	}
	anthropicCfg := &AnthropicConfig{
		APIKey:  cfg.APIKey,
		BaseURL: base,
		Version: "2023-06-01",
	}
	return &xaiAdapter{
		openai:    newOpenAIAdapter(openaiCfg, client, ProviderXAI),
		anthropic: newAnthropicAdapter(anthropicCfg, client, ProviderXAI),
		mode:      cfg.CompatibilityMode,
	}
}

func (x *xaiAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	return x.openai.ListModels(ctx)
}

func (x *xaiAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	adapter := x.selectAdapter(in)
	return adapter.Generate(ctx, in)
}

func (x *xaiAdapter) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
	return ImageGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "xAI image generation is not supported",
		Provider: ProviderXAI,
	}
}

func (x *xaiAdapter) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
	return MeshGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "xAI mesh generation is not supported",
		Provider: ProviderXAI,
	}
}

func (x *xaiAdapter) Transcribe(ctx context.Context, in TranscribeInput) (TranscribeOutput, error) {
	return TranscribeOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "xAI transcription is not supported",
		Provider: ProviderXAI,
	}
}

func (x *xaiAdapter) Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	adapter := x.selectAdapter(in)
	return adapter.Stream(ctx, in)
}

func (x *xaiAdapter) selectAdapter(in GenerateInput) ProviderAdapter {
	mode := x.mode
	if in.Metadata != nil {
		if value, ok := in.Metadata[xaiCompatibilityKey]; ok {
			mode = value
		}
	}
	if mode == "anthropic" {
		return x.anthropic
	}
	return x.openai
}
