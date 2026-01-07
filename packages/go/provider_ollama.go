package aikit

import (
	"context"
	"net/http"
	"strings"
)

type ollamaAdapter struct {
	openai ProviderAdapter
}

func newOllamaAdapter(cfg *OllamaConfig, client *http.Client) ProviderAdapter {
	base := strings.TrimSpace(cfg.BaseURL)
	if base == "" {
		base = "http://localhost:11434"
	}
	openaiConfig := &OpenAIConfig{
		APIKey:              cfg.APIKey,
		APIKeys:             cfg.APIKeys,
		BaseURL:             base,
		DefaultUseResponses: cfg.DefaultUseResponses,
	}
	return &ollamaAdapter{
		openai: newOpenAIAdapter(openaiConfig, client, ProviderOllama),
	}
}

func (a *ollamaAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	return a.openai.ListModels(ctx)
}

func (a *ollamaAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	in.Provider = ProviderOllama
	return a.openai.Generate(ctx, in)
}

func (a *ollamaAdapter) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
	return ImageGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Ollama image generation is not supported",
		Provider: ProviderOllama,
	}
}

func (a *ollamaAdapter) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
	return MeshGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Ollama mesh generation is not supported",
		Provider: ProviderOllama,
	}
}

func (a *ollamaAdapter) GenerateSpeech(ctx context.Context, in SpeechGenerateInput) (SpeechGenerateOutput, error) {
	_ = ctx
	_ = in
	return SpeechGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Ollama speech generation is not supported",
		Provider: ProviderOllama,
	}
}

func (a *ollamaAdapter) GenerateVideo(ctx context.Context, in VideoGenerateInput) (VideoGenerateOutput, error) {
	_ = ctx
	_ = in
	return VideoGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Ollama video generation is not supported",
		Provider: ProviderOllama,
	}
}

func (a *ollamaAdapter) GenerateVoiceAgent(ctx context.Context, in VoiceAgentInput) (VoiceAgentOutput, error) {
	_ = ctx
	_ = in
	return VoiceAgentOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Ollama voice agent is not supported",
		Provider: ProviderOllama,
	}
}

func (a *ollamaAdapter) Transcribe(ctx context.Context, in TranscribeInput) (TranscribeOutput, error) {
	return TranscribeOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Ollama transcription is not supported",
		Provider: ProviderOllama,
	}
}

func (a *ollamaAdapter) Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	in.Provider = ProviderOllama
	return a.openai.Stream(ctx, in)
}
