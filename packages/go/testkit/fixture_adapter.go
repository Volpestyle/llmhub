package testkit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	aikit "github.com/Volpestyle/ai-kit/packages/go"
)

type FixtureKeyInput struct {
	Method   string
	Provider aikit.Provider
	Model    string
	Input    interface{}
}

type Fixture struct {
	Generate *aikit.GenerateOutput
	Stream   []aikit.StreamChunk
	Image    *aikit.ImageGenerateOutput
	Mesh     *aikit.MeshGenerateOutput
	Transcribe *aikit.TranscribeOutput
}

type FixtureCalls struct {
	Generate []aikit.GenerateInput
	Stream   []aikit.GenerateInput
	Image    []aikit.ImageGenerateInput
	Mesh     []aikit.MeshGenerateInput
	Transcribe []aikit.TranscribeInput
}

type FixtureAdapter struct {
	Provider  aikit.Provider
	Fixtures  map[string]Fixture
	Models    []aikit.ModelMetadata
	ChunkSize int
	KeyFunc   func(input FixtureKeyInput) string
	Calls     FixtureCalls
}

func (f *FixtureAdapter) ListModels(ctx context.Context) ([]aikit.ModelMetadata, error) {
	_ = ctx
	return f.Models, nil
}

func (f *FixtureAdapter) Generate(ctx context.Context, in aikit.GenerateInput) (aikit.GenerateOutput, error) {
	_ = ctx
	f.Calls.Generate = append(f.Calls.Generate, in)
	entry, err := f.lookupFixture("generate", in.Provider, in.Model, in)
	if err != nil {
		return aikit.GenerateOutput{}, err
	}
	if entry.Generate == nil {
		return aikit.GenerateOutput{}, fmt.Errorf("fixture for generate is missing (key: %s)", f.keyFor("generate", in.Provider, in.Model, in))
	}
	return *entry.Generate, nil
}

func (f *FixtureAdapter) GenerateImage(ctx context.Context, in aikit.ImageGenerateInput) (aikit.ImageGenerateOutput, error) {
	_ = ctx
	f.Calls.Image = append(f.Calls.Image, in)
	entry, err := f.lookupFixture("image", in.Provider, in.Model, in)
	if err != nil {
		return aikit.ImageGenerateOutput{}, err
	}
	if entry.Image == nil {
		return aikit.ImageGenerateOutput{}, fmt.Errorf("fixture for image is missing (key: %s)", f.keyFor("image", in.Provider, in.Model, in))
	}
	return *entry.Image, nil
}

func (f *FixtureAdapter) GenerateMesh(ctx context.Context, in aikit.MeshGenerateInput) (aikit.MeshGenerateOutput, error) {
	_ = ctx
	f.Calls.Mesh = append(f.Calls.Mesh, in)
	entry, err := f.lookupFixture("mesh", in.Provider, in.Model, in)
	if err != nil {
		return aikit.MeshGenerateOutput{}, err
	}
	if entry.Mesh == nil {
		return aikit.MeshGenerateOutput{}, fmt.Errorf("fixture for mesh is missing (key: %s)", f.keyFor("mesh", in.Provider, in.Model, in))
	}
	return *entry.Mesh, nil
}

func (f *FixtureAdapter) Transcribe(ctx context.Context, in aikit.TranscribeInput) (aikit.TranscribeOutput, error) {
	_ = ctx
	f.Calls.Transcribe = append(f.Calls.Transcribe, in)
	entry, err := f.lookupFixture("transcribe", in.Provider, in.Model, in)
	if err != nil {
		return aikit.TranscribeOutput{}, err
	}
	if entry.Transcribe == nil {
		return aikit.TranscribeOutput{}, fmt.Errorf("fixture for transcribe is missing (key: %s)", f.keyFor("transcribe", in.Provider, in.Model, in))
	}
	return *entry.Transcribe, nil
}

func (f *FixtureAdapter) Stream(ctx context.Context, in aikit.GenerateInput) (<-chan aikit.StreamChunk, error) {
	_ = ctx
	f.Calls.Stream = append(f.Calls.Stream, in)
	entry, err := f.lookupFixture("stream", in.Provider, in.Model, in)
	if err != nil {
		return nil, err
	}
	if entry.Stream != nil {
		return streamFromChunks(entry.Stream), nil
	}
	if entry.Generate == nil {
		return nil, fmt.Errorf("fixture for stream is missing (key: %s)", f.keyFor("stream", in.Provider, in.Model, in))
	}
	return streamFromChunks(buildStreamChunks(entry.Generate, f.chunkSize())), nil
}

func DefaultFixtureKey(input FixtureKeyInput) string {
	payload := struct {
		Method   string          `json:"method"`
		Provider aikit.Provider  `json:"provider"`
		Model    string          `json:"model"`
		Input    interface{}     `json:"input"`
	}{
		Method:   input.Method,
		Provider: input.Provider,
		Model:    input.Model,
		Input:    input.Input,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Sprintf("%s:%s:%s:marshal_error", input.Method, input.Provider, input.Model)
	}
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%s:%s:%s:%s", input.Method, input.Provider, input.Model, hex.EncodeToString(hash[:])[:12])
}

func GenerateKey(input aikit.GenerateInput) string {
	return DefaultFixtureKey(FixtureKeyInput{
		Method:   "generate",
		Provider: input.Provider,
		Model:    input.Model,
		Input:    input,
	})
}

func StreamKey(input aikit.GenerateInput) string {
	return DefaultFixtureKey(FixtureKeyInput{
		Method:   "stream",
		Provider: input.Provider,
		Model:    input.Model,
		Input:    input,
	})
}

func ImageKey(input aikit.ImageGenerateInput) string {
	return DefaultFixtureKey(FixtureKeyInput{
		Method:   "image",
		Provider: input.Provider,
		Model:    input.Model,
		Input:    input,
	})
}

func MeshKey(input aikit.MeshGenerateInput) string {
	return DefaultFixtureKey(FixtureKeyInput{
		Method:   "mesh",
		Provider: input.Provider,
		Model:    input.Model,
		Input:    input,
	})
}

func TranscribeKey(input aikit.TranscribeInput) string {
	return DefaultFixtureKey(FixtureKeyInput{
		Method:   "transcribe",
		Provider: input.Provider,
		Model:    input.Model,
		Input:    input,
	})
}

func buildStreamChunks(output *aikit.GenerateOutput, chunkSize int) []aikit.StreamChunk {
	if output == nil {
		return nil
	}
	chunks := make([]aikit.StreamChunk, 0)
	if output.Text != "" {
		for _, part := range chunkText(output.Text, chunkSize) {
			chunks = append(chunks, aikit.StreamChunk{Type: aikit.StreamChunkDelta, TextDelta: part})
		}
	}
	for idx := range output.ToolCalls {
		call := output.ToolCalls[idx]
		chunks = append(chunks, aikit.StreamChunk{Type: aikit.StreamChunkToolCall, Call: &call})
	}
	chunks = append(chunks, aikit.StreamChunk{
		Type:         aikit.StreamChunkMessageEnd,
		Usage:        output.Usage,
		FinishReason: output.FinishReason,
	})
	return chunks
}

func streamFromChunks(chunks []aikit.StreamChunk) <-chan aikit.StreamChunk {
	out := make(chan aikit.StreamChunk, len(chunks))
	go func() {
		defer close(out)
		for _, chunk := range chunks {
			out <- chunk
		}
	}()
	return out
}

func chunkText(text string, size int) []string {
	if text == "" {
		return nil
	}
	if size <= 0 {
		return []string{text}
	}
	chunks := make([]string, 0, (len(text)/size)+1)
	for start := 0; start < len(text); start += size {
		end := start + size
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[start:end])
	}
	return chunks
}

func (f *FixtureAdapter) keyFor(method string, provider aikit.Provider, model string, input interface{}) string {
	keyFn := f.KeyFunc
	if keyFn == nil {
		keyFn = DefaultFixtureKey
	}
	return keyFn(FixtureKeyInput{
		Method:   strings.TrimSpace(method),
		Provider: provider,
		Model:    model,
		Input:    input,
	})
}

func (f *FixtureAdapter) lookupFixture(method string, provider aikit.Provider, model string, input interface{}) (Fixture, error) {
	key := f.keyFor(method, provider, model, input)
	entry, ok := f.Fixtures[key]
	if !ok {
		return Fixture{}, fmt.Errorf("fixture not found (key: %s)", key)
	}
	return entry, nil
}

func (f *FixtureAdapter) chunkSize() int {
	if f.ChunkSize > 0 {
		return f.ChunkSize
	}
	return 24
}
