package llmhub

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

type mockHub struct {
	modelsResp   []ModelMetadata
	generateResp GenerateOutput
	streamChunks []StreamChunk
	lastListOpts *ListModelsOptions
}

func (m *mockHub) ListModels(ctx context.Context, opts *ListModelsOptions) ([]ModelMetadata, error) {
	m.lastListOpts = opts
	return m.modelsResp, nil
}

func (m *mockHub) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	return m.generateResp, nil
}

func (m *mockHub) StreamGenerate(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
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
	hub := &mockHub{
		modelsResp: []ModelMetadata{{ID: "test", Provider: ProviderOpenAI}},
	}
	req := httptest.NewRequest(http.MethodGet, "/provider-models?providers=openai", nil)
	rec := httptest.NewRecorder()
	ModelsHandler(hub, nil)(rec, req)
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
	hub := &mockHub{}
	req := httptest.NewRequest(http.MethodGet, "/provider-models?refresh=true", nil)
	rec := httptest.NewRecorder()
	ModelsHandler(hub, nil)(rec, req)
	if hub.lastListOpts == nil || !hub.lastListOpts.Refresh {
		t.Fatalf("expected refresh flag to be forwarded: %+v", hub.lastListOpts)
	}
}

func TestGenerateHandlerReturnsOutput(t *testing.T) {
	hub := &mockHub{
		generateResp: GenerateOutput{Text: "ok"},
	}
	body := bytes.NewBufferString(`{"provider":"openai","model":"gpt","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/generate", body)
	rec := httptest.NewRecorder()
	GenerateHandler(hub)(rec, req)
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
	hub := &mockHub{
		streamChunks: []StreamChunk{
			{Type: StreamChunkDelta, TextDelta: "hel"},
			{Type: StreamChunkDelta, TextDelta: "lo"},
		},
	}
	body := bytes.NewBufferString(`{"provider":"openai","model":"gpt","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/generate/stream", body)
	rec := httptest.NewRecorder()
	GenerateSSEHandler(hub)(rec, req)
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
