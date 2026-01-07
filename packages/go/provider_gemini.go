package aikit

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type googleAdapter struct {
	config  *GoogleConfig
	client  *http.Client
	baseURL string
}

type geminiModelList struct {
	Models []struct {
		Name             string `json:"name"`
		DisplayName      string `json:"displayName"`
		InputTokenLimit  int    `json:"inputTokenLimit"`
		OutputTokenLimit int    `json:"outputTokenLimit"`
	} `json:"models"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []map[string]interface{} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

func newGoogleAdapter(cfg *GoogleConfig, client *http.Client) ProviderAdapter {
	base := cfg.BaseURL
	if base == "" {
		base = "https://generativelanguage.googleapis.com"
	}
	return &googleAdapter{
		config:  cfg,
		client:  client,
		baseURL: base,
	}
}

func (g *googleAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, g.baseURL+"/v1beta/models?key="+g.config.APIKey, nil)
	if err != nil {
		return nil, err
	}
	var payload geminiModelList
	if err := doJSON(ctx, g.client, req, ProviderGoogle, &payload); err != nil {
		return nil, err
	}
	var models []ModelMetadata
	for _, model := range payload.Models {
		id := strings.TrimPrefix(model.Name, "models/")
		models = append(models, ModelMetadata{
			ID:          id,
			DisplayName: model.DisplayName,
			Provider:    ProviderGoogle,
			Family:      deriveFamily(id),
			ContextWindow: model.InputTokenLimit,
		})
	}
	return models, nil
}

func (g *googleAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	path := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s", g.baseURL, ensureModelsPrefix(in.Model), g.config.APIKey)
	payload := g.buildPayload(in)
	req, err := jsonRequestWithContext(ctx, http.MethodPost, path, payload)
	if err != nil {
		return GenerateOutput{}, err
	}
	var resp geminiResponse
	if err := doJSON(ctx, g.client, req, ProviderGoogle, &resp); err != nil {
		return GenerateOutput{}, err
	}
	return convertGeminiResponse(resp), nil
}

func (g *googleAdapter) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
	path := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s", g.baseURL, ensureModelsPrefix(in.Model), g.config.APIKey)
	payload := buildGeminiImagePayload(in)
	req, err := jsonRequestWithContext(ctx, http.MethodPost, path, payload)
	if err != nil {
		return ImageGenerateOutput{}, err
	}
	var resp geminiResponse
	if err := doJSON(ctx, g.client, req, ProviderGoogle, &resp); err != nil {
		return ImageGenerateOutput{}, err
	}
	mime, data := extractGeminiInlineImage(resp)
	if data == "" {
		return ImageGenerateOutput{}, &KitError{
			Kind:     ErrorUnknown,
			Message:  "Gemini image response missing inline data",
			Provider: ProviderGoogle,
		}
	}
	return ImageGenerateOutput{
		Mime: mime,
		Data: data,
		Raw:  resp,
	}, nil
}

func (g *googleAdapter) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
	return MeshGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Gemini mesh generation is not supported",
		Provider: ProviderGoogle,
	}
}

func (g *googleAdapter) GenerateSpeech(ctx context.Context, in SpeechGenerateInput) (SpeechGenerateOutput, error) {
	_ = ctx
	_ = in
	return SpeechGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Gemini speech generation is not supported",
		Provider: ProviderGoogle,
	}
}

func (g *googleAdapter) GenerateVideo(ctx context.Context, in VideoGenerateInput) (VideoGenerateOutput, error) {
	_ = ctx
	_ = in
	return VideoGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Gemini video generation is not supported",
		Provider: ProviderGoogle,
	}
}

func (g *googleAdapter) GenerateVoiceAgent(ctx context.Context, in VoiceAgentInput) (VoiceAgentOutput, error) {
	_ = ctx
	_ = in
	return VoiceAgentOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Gemini voice agent is not supported",
		Provider: ProviderGoogle,
	}
}

func (g *googleAdapter) Transcribe(ctx context.Context, in TranscribeInput) (TranscribeOutput, error) {
	return TranscribeOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Gemini transcription is not supported",
		Provider: ProviderGoogle,
	}
}

func (g *googleAdapter) Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	path := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", g.baseURL, ensureModelsPrefix(in.Model), g.config.APIKey)
	payload := g.buildPayload(in)
	req, err := jsonRequestWithContext(ctx, http.MethodPost, path, payload)
	if err != nil {
		return nil, err
	}
	resp, err := doRequest(ctx, g.client, req, ProviderGoogle)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamChunk)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		events := streamSSE(ctx, resp.Body)
		for event := range events {
			if event.Data == "" || event.Data == "[DONE]" {
				continue
			}
			var payload geminiResponse
			if err := json.Unmarshal([]byte(event.Data), &payload); err != nil {
				continue
			}
			output := convertGeminiResponse(payload)
			if output.Text != "" {
				ch <- StreamChunk{Type: StreamChunkDelta, TextDelta: output.Text}
			}
			for _, call := range output.ToolCalls {
				callCopy := call
				ch <- StreamChunk{Type: StreamChunkToolCall, Call: &callCopy}
			}
		}
		ch <- StreamChunk{
			Type:         StreamChunkMessageEnd,
			FinishReason: "stop",
		}
	}()
	return ch, nil
}

func (g *googleAdapter) buildPayload(in GenerateInput) map[string]interface{} {
	system, contents := buildGeminiMessages(in.Messages)
	config := map[string]interface{}{}
	if in.Temperature != nil {
		config["temperature"] = in.Temperature
	}
	if in.TopP != nil {
		config["topP"] = in.TopP
	}
	if in.MaxTokens != nil {
		config["maxOutputTokens"] = in.MaxTokens
	}
	if in.ResponseFormat != nil && in.ResponseFormat.Type == "json_schema" && in.ResponseFormat.JsonSchema != nil {
		config["responseMimeType"] = "application/json"
		config["responseSchema"] = in.ResponseFormat.JsonSchema.Schema
	}
	payload := map[string]interface{}{
		"contents":          contents,
		"systemInstruction": system,
		"generationConfig":  config,
		"tools":             buildGeminiTools(in.Tools),
		"toolConfig":        buildGeminiToolConfig(in.ToolChoice),
	}
	return payload
}

func buildGeminiImagePayload(in ImageGenerateInput) map[string]interface{} {
	parts := []map[string]interface{}{
		{"text": in.Prompt},
	}
	for _, image := range in.InputImages {
		if image.Base64 != "" {
			mimeType := image.MediaType
			if strings.TrimSpace(mimeType) == "" {
				mimeType = "image/png"
			}
			parts = append(parts, map[string]interface{}{
				"inlineData": map[string]string{
					"mimeType": mimeType,
					"data":     image.Base64,
				},
			})
		} else if image.URL != "" {
			parts = append(parts, map[string]interface{}{
				"fileData": map[string]string{
					"fileUri": image.URL,
				},
			})
		}
	}
	return map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"role":  "user",
				"parts": parts,
			},
		},
	}
}

func buildGeminiMessages(messages []Message) (map[string]interface{}, []map[string]interface{}) {
	var systemParts []string
	var contents []map[string]interface{}
	for _, message := range messages {
		if message.Role == "system" {
			systemParts = append(systemParts, joinTextContent(message.Content))
			continue
		}
		if message.Role == "tool" {
			contents = append(contents, map[string]interface{}{
				"role": "user",
				"parts": []map[string]interface{}{
					{
						"functionResponse": map[string]interface{}{
							"name":     message.Name,
							"response": joinTextContent(message.Content),
						},
					},
				},
			})
			continue
		}
		parts := make([]map[string]interface{}, 0, len(message.Content))
		for _, part := range message.Content {
			if part.Type == "text" {
				parts = append(parts, map[string]interface{}{"text": part.Text})
			} else if part.Image != nil {
				if part.Image.Base64 != "" {
					parts = append(parts, map[string]interface{}{
						"inlineData": map[string]string{
							"mimeType": part.Image.MediaType,
							"data":     part.Image.Base64,
						},
					})
				} else if part.Image.URL != "" {
					parts = append(parts, map[string]interface{}{
						"fileData": map[string]string{
							"fileUri": part.Image.URL,
						},
					})
				}
			}
		}
		role := "user"
		if message.Role == "assistant" {
			role = "model"
		}
		contents = append(contents, map[string]interface{}{
			"role":  role,
			"parts": parts,
		})
	}
	var system map[string]interface{}
	if len(systemParts) > 0 {
		system = map[string]interface{}{
			"role": "system",
			"parts": []map[string]string{
				{"text": strings.Join(systemParts, "\n")},
			},
		}
	}
	return system, contents
}

func buildGeminiTools(tools []ToolDefinition) interface{} {
	if len(tools) == 0 {
		return nil
	}
	funcs := make([]map[string]interface{}, 0, len(tools))
	for _, tool := range tools {
		funcs = append(funcs, map[string]interface{}{
			"name":        tool.Name,
			"description": tool.Description,
			"parameters":  tool.Parameters,
		})
	}
	return []map[string]interface{}{
		{
			"functionDeclarations": funcs,
		},
	}
}

func buildGeminiToolConfig(choice *ToolChoice) interface{} {
	if choice == nil {
		return nil
	}
	if choice.Type == "auto" {
		return map[string]interface{}{
			"functionCallConfig": map[string]string{"mode": "AUTO"},
		}
	}
	if choice.Type == "none" {
		return map[string]interface{}{
			"functionCallConfig": map[string]string{"mode": "NONE"},
		}
	}
	return map[string]interface{}{
		"functionCallConfig": map[string]interface{}{
			"mode":                 "ANY",
			"allowedFunctionNames": []string{choice.Name},
		},
	}
}

func convertGeminiResponse(resp geminiResponse) GenerateOutput {
	text := ""
	var calls []ToolCall
	var finish string
	if len(resp.Candidates) > 0 {
		finish = resp.Candidates[0].FinishReason
		for _, part := range resp.Candidates[0].Content.Parts {
			if value, ok := part["text"].(string); ok {
				text += value
			}
			if fn, ok := part["functionCall"].(map[string]interface{}); ok {
				args, _ := json.Marshal(fn["args"])
				name, _ := fn["name"].(string)
				calls = append(calls, ToolCall{
					ID:            name,
					Name:          name,
					ArgumentsJSON: string(args),
				})
			}
		}
	}
	usage := &Usage{}
	usage.InputTokens = resp.UsageMetadata.PromptTokenCount
	usage.OutputTokens = resp.UsageMetadata.CandidatesTokenCount
	usage.TotalTokens = resp.UsageMetadata.TotalTokenCount
	return GenerateOutput{
		Text:         text,
		ToolCalls:    calls,
		Usage:        usage,
		FinishReason: finish,
		Raw:          resp,
	}
}

func extractGeminiInlineImage(resp geminiResponse) (string, string) {
	for _, candidate := range resp.Candidates {
		for _, part := range candidate.Content.Parts {
			if inline, ok := part["inlineData"].(map[string]interface{}); ok {
				data, _ := inline["data"].(string)
				mime, _ := inline["mimeType"].(string)
				if data != "" {
					if mime == "" {
						mime = "image/png"
					}
					return mime, data
				}
			}
		}
	}
	return "", ""
}

func ensureModelsPrefix(id string) string {
	if strings.HasPrefix(id, "models/") {
		return id
	}
	return "models/" + id
}

func jsonRequestWithContext(ctx context.Context, method, url string, payload map[string]interface{}) (*http.Request, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("content-type", "application/json")
	return req, nil
}
