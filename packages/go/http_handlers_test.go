package aikit

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

type mockKit struct {
	modelsResp   []ModelMetadata
	generateResp GenerateOutput
	imageResp    ImageGenerateOutput
	meshResp     MeshGenerateOutput
	streamChunks []StreamChunk
	lastListOpts *ListModelsOptions
}

func (m *mockKit) ListModels(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error) {
	m.lastListOpts = opts
	return m.modelsResp, nil
}

func (m *mockKit) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	return m.generateResp, nil
}

func (m *mockKit) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
	return m.imageResp, nil
}

func (m *mockKit) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
	return m.meshResp, nil
}

func (m *mockKit) StreamGenerate(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk)
	go func() {
		defer close(ch)
		for _, chunk := range m.streamChunks {
			ch <- chunk
		}
	}()
	return ch, nil
}

func TestModelsHandlerReturnsJSON(t *testing.T) {
	kit := &mockKit{
		modelsResp: []ModelMetadata{{ID: "test", Provider: ProviderOpenAI}},
	}
	req := httptest.NewRequest(http.MethodGet, "/provider-models?providers=openai", nil)
	rec := httptest.NewRecorder()
	ModelsHandler(kit, nil)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var payload []ModelMetadata
	readBody(t, rec.Result().Body, &payload)
	if len(payload) != 1 || payload[0].ID != "test" {
		t.Fatalf("unexpected payload: %+v", payload)
	}
}

func TestModelsHandlerParsesRefreshQuery(t *testing.T) {
	kit := &mockKit{}
	req := httptest.NewRequest(http.MethodGet, "/provider-models?refresh=true", nil)
	rec := httptest.NewRecorder()
	ModelsHandler(kit, nil)(rec, req)
	if kit.lastListOpts == nil || !kit.lastListOpts.Refresh {
		t.Fatalf("expected refresh flag to be forwarded: %+v", kit.lastListOpts)
	}
}

func TestGenerateHandlerReturnsOutput(t *testing.T) {
	kit := &mockKit{
		generateResp: GenerateOutput{Text: "ok"},
	}
	body := bytes.NewBufferString(`{"provider":"openai","model":"gpt","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/generate", body)
	rec := httptest.NewRecorder()
	GenerateHandler(kit)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var payload GenerateOutput
	readBody(t, rec.Result().Body, &payload)
	if payload.Text != "ok" {
		t.Fatalf("unexpected payload: %+v", payload)
	}
}

func TestGenerateSSEHandlerStreams(t *testing.T) {
	kit := &mockKit{
		streamChunks: []StreamChunk{
			{Type: StreamChunkDelta, TextDelta: "hel"},
			{Type: StreamChunkDelta, TextDelta: "lo"},
		},
	}
	body := bytes.NewBufferString(`{"provider":"openai","model":"gpt","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/generate/stream", body)
	rec := httptest.NewRecorder()
	GenerateSSEHandler(kit)(rec, req)
	result := rec.Body.String()
	if status := rec.Code; status != http.StatusOK {
		t.Fatalf("expected 200, got %d", status)
	}
	if !bytes.Contains([]byte(result), []byte("event: chunk")) {
		t.Fatalf("expected SSE chunk events, got %s", result)
	}
}

func readBody[T any](t *testing.T, r io.ReadCloser, target *T) {
	t.Helper()
	defer r.Close()
	data, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		t.Fatalf("decode: %v", err)
	}
}
