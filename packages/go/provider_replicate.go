package aikit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type replicateAdapter struct {
	apiKey  string
	baseURL string
	client  *http.Client
	timeout time.Duration
}

func newReplicateAdapter(cfg *ReplicateConfig, client *http.Client) *replicateAdapter {
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://api.replicate.com/v1"
	}
	timeout := time.Duration(cfg.TimeoutS * float64(time.Second))
	if timeout <= 0 {
		timeout = 6 * time.Minute
	}
	return &replicateAdapter{
		apiKey:  cfg.APIKey,
		baseURL: baseURL,
		client:  client,
		timeout: timeout,
	}
}

type replicateCatalogEntry struct {
	ID           string             `json:"id"`
	DisplayName  string             `json:"displayName"`
	Provider     Provider           `json:"provider"`
	Family       string             `json:"family"`
	Capabilities ModelCapabilities  `json:"capabilities"`
	Inputs       []map[string]interface{} `json:"inputs"`
	ContextWindow int               `json:"contextWindow"`
	TokenPrices  *TokenPrices       `json:"tokenPrices"`
	VideoPrices  map[string]float64 `json:"videoPrices"`
	Deprecated   bool               `json:"deprecated"`
	InPreview    bool               `json:"inPreview"`
}

var replicateCatalogOnce sync.Once
var replicateCatalog []ModelMetadata

func loadReplicateCatalog() []ModelMetadata {
	replicateCatalogOnce.Do(func() {
		root := modelsRoot()
		if root == "" {
			replicateCatalog = []ModelMetadata{}
			return
		}
		path := filepath.Join(root, "replicate_models.json")
		raw, err := os.ReadFile(path)
		if err != nil {
			replicateCatalog = []ModelMetadata{}
			return
		}
		var entries []replicateCatalogEntry
		if err := json.Unmarshal(raw, &entries); err != nil {
			replicateCatalog = []ModelMetadata{}
			return
		}
		models := make([]ModelMetadata, 0, len(entries))
		for _, entry := range entries {
			provider := entry.Provider
			if provider == "" {
				provider = ProviderReplicate
			}
			models = append(models, ModelMetadata{
				ID:            entry.ID,
				DisplayName:   entry.DisplayName,
				Provider:      provider,
				Family:        entry.Family,
				Capabilities:  entry.Capabilities,
				Inputs:        entry.Inputs,
				ContextWindow: entry.ContextWindow,
				TokenPrices:   entry.TokenPrices,
				VideoPrices:   entry.VideoPrices,
				Deprecated:    entry.Deprecated,
				InPreview:     entry.InPreview,
			})
		}
		replicateCatalog = models
	})
	return replicateCatalog
}

func (a *replicateAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	_ = ctx
	models := loadReplicateCatalog()
	if len(models) == 0 {
		return []ModelMetadata{
			{
				ID:          "replicate",
				DisplayName: "Replicate",
				Provider:    ProviderReplicate,
				Family:      "video",
				Capabilities: ModelCapabilities{
					Text:             false,
					Vision:           true,
					Image:            false,
					Video:            true,
					ToolUse:          false,
					StructuredOutput: false,
					Reasoning:        false,
				},
			},
		}, nil
	}
	return models, nil
}

func (a *replicateAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	_ = ctx
	_ = in
	return GenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Replicate text generation is not supported",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
	_ = ctx
	_ = in
	return ImageGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Replicate image generation is not supported",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
	_ = ctx
	_ = in
	return MeshGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Replicate mesh generation is not supported",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) GenerateSpeech(ctx context.Context, in SpeechGenerateInput) (SpeechGenerateOutput, error) {
	_ = ctx
	_ = in
	return SpeechGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Replicate speech generation is not supported",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) GenerateVideo(ctx context.Context, in VideoGenerateInput) (VideoGenerateOutput, error) {
	ctx, cancel := a.withTimeout(ctx)
	if cancel != nil {
		defer cancel()
	}
	payload := map[string]interface{}{
		"prompt": in.Prompt,
	}
	startImage := in.StartImage
	if startImage == "" && len(in.InputImages) > 0 {
		startImage = imageInputToPayload(in.InputImages[0])
	}
	if startImage != "" {
		payload["start_image"] = startImage
	}
	if in.Duration != nil {
		payload["duration"] = *in.Duration
	}
	if in.AspectRatio != "" {
		payload["aspect_ratio"] = in.AspectRatio
	}
	if in.NegativePrompt != "" {
		payload["negative_prompt"] = in.NegativePrompt
	}
	if in.GenerateAudio != nil {
		payload["generate_audio"] = *in.GenerateAudio
	}
	for key, value := range in.Parameters {
		payload[key] = value
	}
	output, err := a.runPrediction(ctx, in.Model, payload)
	if err != nil {
		return VideoGenerateOutput{}, err
	}
	data, err := a.coerceVideoBytes(ctx, output)
	if err != nil {
		return VideoGenerateOutput{}, err
	}
	return VideoGenerateOutput{
		Mime: "video/mp4",
		Data: base64.StdEncoding.EncodeToString(data),
		Raw:  output,
	}, nil
}

func (a *replicateAdapter) GenerateVoiceAgent(ctx context.Context, in VoiceAgentInput) (VoiceAgentOutput, error) {
	_ = ctx
	_ = in
	return VoiceAgentOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Replicate voice agent is not supported",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) Transcribe(ctx context.Context, in TranscribeInput) (TranscribeOutput, error) {
	_ = ctx
	_ = in
	return TranscribeOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Replicate transcription is not supported",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	_ = ctx
	_ = in
	return nil, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Replicate streaming is not supported",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if a.timeout <= 0 {
		return ctx, nil
	}
	if _, ok := ctx.Deadline(); ok {
		return ctx, nil
	}
	return context.WithTimeout(ctx, a.timeout)
}

type replicatePrediction struct {
	ID     string      `json:"id"`
	Status string      `json:"status"`
	Output interface{} `json:"output"`
	Error  interface{} `json:"error"`
}

func (a *replicateAdapter) runPrediction(ctx context.Context, model string, input map[string]interface{}) (interface{}, error) {
	modelID, version := splitModelVersion(model)
	body := map[string]interface{}{"input": input}
	path := fmt.Sprintf("/models/%s/predictions", modelID)
	if version != "" {
		body["version"] = version
		path = "/predictions"
	}
	req, err := a.jsonRequest(ctx, http.MethodPost, path, body)
	if err != nil {
		return nil, err
	}
	var prediction replicatePrediction
	if err := doJSON(ctx, a.client, req, ProviderReplicate, &prediction); err != nil {
		return nil, err
	}
	return a.pollPrediction(ctx, prediction.ID)
}

func (a *replicateAdapter) pollPrediction(ctx context.Context, id string) (interface{}, error) {
	path := fmt.Sprintf("/predictions/%s", id)
	for {
		req, err := a.jsonRequest(ctx, http.MethodGet, path, nil)
		if err != nil {
			return nil, err
		}
		var prediction replicatePrediction
		if err := doJSON(ctx, a.client, req, ProviderReplicate, &prediction); err != nil {
			return nil, err
		}
		switch prediction.Status {
		case "succeeded":
			return prediction.Output, nil
		case "failed", "canceled":
			return nil, &KitError{
				Kind:     ErrorProviderUnavailable,
				Message:  fmt.Sprintf("replicate prediction %s", prediction.Status),
				Provider: ProviderReplicate,
			}
		default:
			if err := sleepWithContext(ctx, 2*time.Second); err != nil {
				return nil, err
			}
		}
	}
}

func (a *replicateAdapter) jsonRequest(ctx context.Context, method, path string, body interface{}) (*http.Request, error) {
	url := a.baseURL + path
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewBuffer(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Token "+a.apiKey)
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

func (a *replicateAdapter) coerceVideoBytes(ctx context.Context, output interface{}) ([]byte, error) {
	switch value := output.(type) {
	case nil:
		return nil, &KitError{
			Kind:     ErrorUnknown,
			Message:  "Replicate video output missing",
			Provider: ProviderReplicate,
		}
	case string:
		return a.downloadOrDecode(ctx, value)
	case []interface{}:
		if len(value) == 0 {
			return nil, &KitError{
				Kind:     ErrorUnknown,
				Message:  "Replicate video output empty",
				Provider: ProviderReplicate,
			}
		}
		return a.coerceVideoBytes(ctx, value[0])
	case map[string]interface{}:
		if url, ok := value["url"].(string); ok {
			return a.downloadOrDecode(ctx, url)
		}
		if video, ok := value["video"].(map[string]interface{}); ok {
			if url, ok := video["url"].(string); ok {
				return a.downloadOrDecode(ctx, url)
			}
		}
	}
	return nil, &KitError{
		Kind:     ErrorUnknown,
		Message:  "Unsupported Replicate video output",
		Provider: ProviderReplicate,
	}
}

func (a *replicateAdapter) downloadOrDecode(ctx context.Context, value string) ([]byte, error) {
	if strings.HasPrefix(value, "http") {
		return a.downloadURL(ctx, value)
	}
	data, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		return nil, &KitError{
			Kind:     ErrorUnknown,
			Message:  "Replicate video output decode failed",
			Provider: ProviderReplicate,
		}
	}
	return data, nil
}

func (a *replicateAdapter) downloadURL(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := doRequest(ctx, a.client, req, ProviderReplicate)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func splitModelVersion(model string) (string, string) {
	parts := strings.SplitN(model, ":", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return model, ""
}

func imageInputToPayload(input ImageInput) string {
	if input.URL != "" {
		return input.URL
	}
	if input.Base64 == "" {
		return ""
	}
	prefix := "data:image/png;base64,"
	if input.MediaType != "" {
		prefix = "data:" + input.MediaType + ";base64,"
	}
	return prefix + input.Base64
}

func sleepWithContext(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
