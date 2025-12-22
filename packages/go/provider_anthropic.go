package llmhub

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

type anthropicAdapter struct {
	provider Provider
	config   *AnthropicConfig
	client   *http.Client
	baseURL  string
	version  string
}

type anthropicModelList struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type anthropicMessageResponse struct {
	Content    []anthropicContentBlock `json:"content"`
	StopReason string                  `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type anthropicContentBlock struct {
	Type  string                 `json:"type"`
	Text  string                 `json:"text,omitempty"`
	ID    string                 `json:"id,omitempty"`
	Name  string                 `json:"name,omitempty"`
	Input map[string]interface{} `json:"input,omitempty"`
}

type anthropicStreamEvent struct {
	Type         string                `json:"type"`
	Index        int                   `json:"index"`
	ContentBlock anthropicContentBlock `json:"content_block"`
	Delta        struct {
		Type        string `json:"type"`
		Text        string `json:"text,omitempty"`
		PartialJSON string `json:"partial_json,omitempty"`
	} `json:"delta"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	Message struct {
		StopReason string `json:"stop_reason"`
	} `json:"message"`
}

func newAnthropicAdapter(cfg *AnthropicConfig, client *http.Client, provider Provider) adapter {
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.anthropic.com"
	}
	version := cfg.Version
	if version == "" {
		version = "2023-06-01"
	}
	return &anthropicAdapter{
		provider: provider,
		config:   cfg,
		client:   client,
		baseURL:  base,
		version:  version,
	}
}

func (a *anthropicAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.baseURL+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	a.applyHeaders(req, false)
	var payload anthropicModelList
	if err := doJSON(ctx, a.client, req, a.provider, &payload); err != nil {
		return nil, err
	}
	models := make([]ModelMetadata, 0, len(payload.Data))
	for _, model := range payload.Data {
		models = append(models, ModelMetadata{
			ID:          model.ID,
			DisplayName: model.ID,
			Provider:    a.provider,
			Family:      deriveFamily(model.ID),
			Capabilities: ModelCapabilities{
				Text:             true,
				Vision:           true,
				ToolUse:          true,
				StructuredOutput: true,
				Reasoning:        false,
			},
		})
	}
	return models, nil
}

func (a *anthropicAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	payload := a.buildPayload(in, false)
	usesStructuredOutput := in.ResponseFormat != nil && in.ResponseFormat.Type == "json_schema"
	req, err := a.jsonRequest(ctx, payload, usesStructuredOutput)
	if err != nil {
		return GenerateOutput{}, err
	}
	var resp anthropicMessageResponse
	if err := doJSON(ctx, a.client, req, a.provider, &resp); err != nil {
		return GenerateOutput{}, err
	}
	return convertAnthropicResponse(resp), nil
}

func (a *anthropicAdapter) Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	payload := a.buildPayload(in, true)
	usesStructuredOutput := in.ResponseFormat != nil && in.ResponseFormat.Type == "json_schema"
	req, err := a.jsonRequest(ctx, payload, usesStructuredOutput)
	if err != nil {
		return nil, err
	}
	resp, err := doRequest(ctx, a.client, req, a.provider)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamChunk)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		events := streamSSE(ctx, resp.Body)
		toolStates := map[int]*ToolCall{}
		var usage *Usage
		var finishReason string
		for event := range events {
			if event.Data == "" || event.Data == "[DONE]" {
				continue
			}
			var payload anthropicStreamEvent
			if err := json.Unmarshal([]byte(event.Data), &payload); err != nil {
				continue
			}
			if payload.Usage.InputTokens != 0 || payload.Usage.OutputTokens != 0 {
				usage = &Usage{
					InputTokens:  payload.Usage.InputTokens,
					OutputTokens: payload.Usage.OutputTokens,
					TotalTokens:  payload.Usage.InputTokens + payload.Usage.OutputTokens,
				}
			}
			switch payload.Type {
			case "content_block_delta":
				if payload.Delta.Type == "text_delta" && payload.Delta.Text != "" {
					ch <- StreamChunk{Type: StreamChunkDelta, TextDelta: payload.Delta.Text}
				}
				if payload.Delta.Type == "input_json_delta" && payload.Index >= 0 {
					state := toolStates[payload.Index]
					if state != nil {
						state.ArgumentsJSON += payload.Delta.PartialJSON
						ch <- StreamChunk{
							Type:  StreamChunkToolCall,
							Call:  state,
							Delta: payload.Delta.PartialJSON,
						}
					}
				}
			case "content_block_start":
				if payload.ContentBlock.Type == "tool_use" {
					toolStates[payload.Index] = &ToolCall{
						ID:            payload.ContentBlock.ID,
						Name:          payload.ContentBlock.Name,
						ArgumentsJSON: toJSONString(payload.ContentBlock.Input),
					}
				}
			case "content_block_stop":
				if state, ok := toolStates[payload.Index]; ok {
					ch <- StreamChunk{
						Type: StreamChunkToolCall,
						Call: state,
					}
				}
			case "message_stop":
				finishReason = payload.Message.StopReason
			}
		}
		ch <- StreamChunk{
			Type:         StreamChunkMessageEnd,
			FinishReason: finishReason,
			Usage:        usage,
		}
	}()
	return ch, nil
}

func (a *anthropicAdapter) buildPayload(in GenerateInput, stream bool) map[string]interface{} {
	system, messages := splitAnthropicMessages(in.Messages)
	payload := map[string]interface{}{
		"model":       in.Model,
		"system":      system,
		"messages":    messages,
		"max_tokens":  defaultMaxTokens(in.MaxTokens),
		"temperature": in.Temperature,
		"top_p":       in.TopP,
		"metadata":    in.Metadata,
		"tools":       convertAnthropicTools(in.Tools),
		"tool_choice": convertAnthropicToolChoice(in.ToolChoice),
		"stream":      stream,
	}
	if in.ResponseFormat != nil && in.ResponseFormat.Type == "json_schema" && in.ResponseFormat.JsonSchema != nil {
		payload["output_format"] = map[string]interface{}{
			"type":   "json_schema",
			"schema": in.ResponseFormat.JsonSchema.Schema,
		}
	}
	return payload
}

func (a *anthropicAdapter) jsonRequest(ctx context.Context, payload map[string]interface{}, useStructuredOutputsBeta bool) (*http.Request, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/v1/messages", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	a.applyHeaders(req, useStructuredOutputsBeta)
	return req, nil
}

func (a *anthropicAdapter) applyHeaders(req *http.Request, useStructuredOutputsBeta bool) {
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-api-key", a.config.APIKey)
	req.Header.Set("anthropic-version", a.version)
	if useStructuredOutputsBeta {
		req.Header.Set("anthropic-beta", "structured-outputs-2025-11-13")
	}
}

func convertAnthropicResponse(resp anthropicMessageResponse) GenerateOutput {
	text := ""
	var calls []ToolCall
	for _, block := range resp.Content {
		if block.Type == "text" {
			text += block.Text
		}
		if block.Type == "tool_use" {
			calls = append(calls, ToolCall{
				ID:            block.ID,
				Name:          block.Name,
				ArgumentsJSON: toJSONString(block.Input),
			})
		}
	}
	usage := &Usage{
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
	}
	return GenerateOutput{
		Text:         text,
		ToolCalls:    calls,
		Usage:        usage,
		FinishReason: resp.StopReason,
		Raw:          resp,
	}
}

func splitAnthropicMessages(messages []Message) (string, []map[string]interface{}) {
	var systemParts []string
	var result []map[string]interface{}
	for _, message := range messages {
		if message.Role == "system" {
			systemParts = append(systemParts, joinTextContent(message.Content))
			continue
		}
		if message.Role == "tool" {
			result = append(result, map[string]interface{}{
				"role": "user",
				"content": []map[string]interface{}{
					{
						"type":        "tool_result",
						"tool_use_id": message.ToolCallID,
						"content":     joinTextContent(message.Content),
					},
				},
			})
			continue
		}
		parts := make([]map[string]interface{}, 0, len(message.Content))
		for _, part := range message.Content {
			if part.Type == "text" {
				parts = append(parts, map[string]interface{}{
					"type": "text",
					"text": part.Text,
				})
			} else if part.Image != nil {
				entry := map[string]interface{}{
					"type": "image",
					"source": map[string]string{
						"type":       "base64",
						"media_type": part.Image.MediaType,
						"data":       part.Image.Base64,
					},
				}
				if part.Image.URL != "" {
					entry["source"] = map[string]string{
						"type": "url",
						"url":  part.Image.URL,
					}
				}
				parts = append(parts, entry)
			}
		}
		result = append(result, map[string]interface{}{
			"role":    message.Role,
			"content": parts,
		})
	}
	system := ""
	if len(systemParts) > 0 {
		system = strings.Join(systemParts, "\n")
	}
	return system, result
}

func convertAnthropicTools(tools []ToolDefinition) []map[string]interface{} {
	if len(tools) == 0 {
		return nil
	}
	list := make([]map[string]interface{}, 0, len(tools))
	for _, tool := range tools {
		list = append(list, map[string]interface{}{
			"name":         tool.Name,
			"description":  tool.Description,
			"input_schema": tool.Parameters,
		})
	}
	return list
}

func convertAnthropicToolChoice(choice *ToolChoice) interface{} {
	if choice == nil {
		return nil
	}
	if choice.Type == "auto" || choice.Type == "none" {
		return choice.Type
	}
	return map[string]string{
		"type": "tool",
		"name": choice.Name,
	}
}

func defaultMaxTokens(value *int) int {
	if value != nil {
		return *value
	}
	return 1024
}

func toJSONString(value interface{}) string {
	data, err := json.Marshal(value)
	if err != nil {
		return "{}"
	}
	return string(data)
}
