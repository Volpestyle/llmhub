package llmhub

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type openAIAdapter struct {
	provider Provider
	config   *OpenAIConfig
	client   *http.Client
	baseURL  string
}

type openAIModelList struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type openAIFunctionToolCall struct {
	ID string `json:"id"`
	// Type is expected to be "function" but we keep it for forward compatibility.
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIResponsesRequiredAction struct {
	Type              string `json:"type"`
	SubmitToolOutputs *struct {
		ToolCalls []openAIFunctionToolCall `json:"tool_calls"`
	} `json:"submit_tool_outputs"`
}

type openAIChatResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content   interface{} `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type openAIChatChunk struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Delta        struct {
			Content   interface{} `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type openAIResponsesResponse struct {
	Status string `json:"status"`
	Output []struct {
		Content []struct {
			Type      string      `json:"type"`
			Text      string      `json:"text"`
			ID        string      `json:"id"`
			Name      string      `json:"name"`
			Arguments interface{} `json:"arguments"`
		} `json:"content"`
	} `json:"output"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	RequiredAction *openAIResponsesRequiredAction `json:"required_action"`
}

type openAIResponsesStreamPayload struct {
	Delta struct {
		Text      string `json:"text"`
		Arguments string `json:"arguments"`
		Name      string `json:"name"`
	} `json:"delta"`
	ToolCallID     string                         `json:"tool_call_id"`
	Response       *openAIResponsesResponse       `json:"response"`
	RequiredAction *openAIResponsesRequiredAction `json:"required_action"`
	Error          *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
		Type    string `json:"type"`
		Param   string `json:"param"`
	} `json:"error"`
}

func (p *openAIResponsesStreamPayload) ErrorCode() string {
	if p == nil || p.Error == nil {
		return ""
	}
	return p.Error.Code
}

type toolState struct {
	ID            string
	Name          string
	ArgumentsJSON string
}

func newOpenAIAdapter(cfg *OpenAIConfig, client *http.Client, provider Provider) adapter {
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.openai.com"
	}
	return &openAIAdapter{
		provider: provider,
		config:   cfg,
		client:   client,
		baseURL:  base,
	}
}

func (a *openAIAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.baseURL+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	a.applyHeaders(req)
	var payload openAIModelList
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
				Vision:           false,
				ToolUse:          true,
				StructuredOutput: false,
				Reasoning:        false,
			},
		})
	}
	return models, nil
}

func (a *openAIAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	if a.shouldUseResponses(in) {
		body := a.buildResponsesPayload(in, false)
		req, err := a.jsonRequest(ctx, http.MethodPost, "/v1/responses", body)
		if err != nil {
			return GenerateOutput{}, err
		}
		var payload openAIResponsesResponse
		if err := doJSON(ctx, a.client, req, a.provider, &payload); err != nil {
			return GenerateOutput{}, err
		}
		return convertResponsesOutput(payload), nil
	}
	body := a.buildChatPayload(in, false)
	req, err := a.jsonRequest(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		return GenerateOutput{}, err
	}
	var payload openAIChatResponse
	if err := doJSON(ctx, a.client, req, a.provider, &payload); err != nil {
		return GenerateOutput{}, err
	}
	return convertOpenAIChatResponse(payload), nil
}

func (a *openAIAdapter) Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	if a.shouldUseResponses(in) {
		body := a.buildResponsesPayload(in, true)
		req, err := a.jsonRequest(ctx, http.MethodPost, "/v1/responses", body)
		if err != nil {
			return nil, err
		}
		resp, err := doRequest(ctx, a.client, req, a.provider)
		if err != nil {
			return nil, err
		}
		return a.streamResponses(ctx, resp), nil
	}
	body := a.buildChatPayload(in, true)
	req, err := a.jsonRequest(ctx, http.MethodPost, "/v1/chat/completions", body)
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
		var finishReason string
		var usage *Usage
		for event := range events {
			if event.Data == "" || event.Data == "[DONE]" {
				continue
			}
			var chunk openAIChatChunk
			if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
				continue
			}
			if chunk.Usage != nil {
				usage = &Usage{
					InputTokens:  chunk.Usage.PromptTokens,
					OutputTokens: chunk.Usage.CompletionTokens,
					TotalTokens:  chunk.Usage.TotalTokens,
				}
			}
			for _, choice := range chunk.Choices {
				if choice.FinishReason != "" {
					finishReason = choice.FinishReason
				}
				if choice.Delta.Content != nil {
					switch val := choice.Delta.Content.(type) {
					case string:
						if val != "" {
							ch <- StreamChunk{Type: StreamChunkDelta, TextDelta: val}
						}
					case []interface{}:
						for _, item := range val {
							if part, ok := item.(map[string]interface{}); ok {
								if text, ok := part["text"].(string); ok && text != "" {
									ch <- StreamChunk{Type: StreamChunkDelta, TextDelta: text}
								}
							}
						}
					}
				}
				for _, tool := range choice.Delta.ToolCalls {
					state := toolStates[tool.Index]
					if state == nil {
						state = &ToolCall{
							ID:   tool.ID,
							Name: tool.Function.Name,
						}
						toolStates[tool.Index] = state
					}
					state.ArgumentsJSON += tool.Function.Arguments
					ch <- StreamChunk{
						Type:  StreamChunkToolCall,
						Call:  state,
						Delta: tool.Function.Arguments,
					}
				}
			}
		}
		if finishReason != "" {
			ch <- StreamChunk{
				Type:         StreamChunkMessageEnd,
				FinishReason: finishReason,
				Usage:        usage,
			}
		}
	}()
	return ch, nil
}

func (a *openAIAdapter) jsonRequest(ctx context.Context, method, path string, payload map[string]interface{}) (*http.Request, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, a.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	a.applyHeaders(req)
	return req, nil
}

func (a *openAIAdapter) applyHeaders(req *http.Request) {
	req.Header.Set("content-type", "application/json")
	req.Header.Set("authorization", fmt.Sprintf("Bearer %s", a.config.APIKey))
	if a.config.Organization != "" {
		req.Header.Set("OpenAI-Organization", a.config.Organization)
	}
}

func (a *openAIAdapter) buildChatPayload(in GenerateInput, stream bool) map[string]interface{} {
	return map[string]interface{}{
		"model":           in.Model,
		"messages":        mapMessagesToChat(in.Messages),
		"temperature":     in.Temperature,
		"top_p":           in.TopP,
		"max_tokens":      in.MaxTokens,
		"stream":          stream,
		"tools":           mapToolsToOpenAI(in.Tools),
		"tool_choice":     mapToolChoiceToOpenAI(in.ToolChoice),
		"response_format": mapResponseFormatToOpenAI(in.ResponseFormat),
		"metadata":        in.Metadata,
	}
}

func (a *openAIAdapter) buildResponsesPayload(in GenerateInput, stream bool) map[string]interface{} {
	return map[string]interface{}{
		"model":             in.Model,
		"input":             mapMessagesToResponses(in.Messages),
		"temperature":       in.Temperature,
		"top_p":             in.TopP,
		"max_output_tokens": in.MaxTokens,
		"stream":            stream,
		"tools":             mapToolsToOpenAI(in.Tools),
		"tool_choice":       mapToolChoiceToOpenAI(in.ToolChoice),
		"response_format":   mapResponseFormatToOpenAI(in.ResponseFormat),
		"metadata":          in.Metadata,
	}
}

func (a *openAIAdapter) shouldUseResponses(in GenerateInput) bool {
	if in.ResponseFormat != nil && in.ResponseFormat.Type == "json_schema" {
		return true
	}
	return a.config.DefaultUseResponses
}

func (a *openAIAdapter) streamResponses(ctx context.Context, resp *http.Response) <-chan StreamChunk {
	ch := make(chan StreamChunk)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		events := streamSSE(ctx, resp.Body)
		toolStates := map[string]*toolState{}
		for event := range events {
			if event.Data == "" || event.Data == "[DONE]" {
				continue
			}
			var payload openAIResponsesStreamPayload
			if err := json.Unmarshal([]byte(event.Data), &payload); err != nil {
				continue
			}
			switch event.Event {
			case "response.output_text.delta", "response.refusal.delta":
				if payload.Delta.Text != "" {
					ch <- StreamChunk{Type: StreamChunkDelta, TextDelta: payload.Delta.Text}
				}
			case "response.tool_call.delta", "response.output_tool_call.delta":
				if payload.ToolCallID == "" {
					continue
				}
				state := toolStates[payload.ToolCallID]
				if state == nil {
					state = &toolState{ID: payload.ToolCallID}
					toolStates[payload.ToolCallID] = state
				}
				if payload.Delta.Name != "" {
					state.Name = payload.Delta.Name
				}
				if payload.Delta.Arguments != "" {
					state.ArgumentsJSON += payload.Delta.Arguments
				}
				ch <- StreamChunk{
					Type: StreamChunkToolCall,
					Call: &ToolCall{
						ID:            state.ID,
						Name:          state.Name,
						ArgumentsJSON: state.ArgumentsJSON,
					},
					Delta: payload.Delta.Arguments,
				}
			case "response.required_action":
				action := payload.RequiredAction
				if action == nil && payload.Response != nil {
					action = payload.Response.RequiredAction
				}
				if action != nil && action.SubmitToolOutputs != nil {
					for _, call := range action.SubmitToolOutputs.ToolCalls {
						state := toolStates[call.ID]
						if state == nil {
							state = &toolState{ID: call.ID, Name: call.Function.Name}
							toolStates[call.ID] = state
						}
						if call.Function.Name != "" {
							state.Name = call.Function.Name
						}
						if call.Function.Arguments != "" {
							state.ArgumentsJSON = call.Function.Arguments
						}
						ch <- StreamChunk{
							Type: StreamChunkToolCall,
							Call: &ToolCall{
								ID:            state.ID,
								Name:          state.Name,
								ArgumentsJSON: state.ArgumentsJSON,
							},
						}
					}
				}
			case "response.completed":
				var usage *Usage
				var status string
				if payload.Response != nil {
					usage = mapResponsesUsage(payload.Response.Usage)
					status = payload.Response.Status
				}
				ch <- StreamChunk{
					Type:         StreamChunkMessageEnd,
					FinishReason: status,
					Usage:        usage,
				}
			case "response.failed", "response.canceled":
				var status string
				if payload.Response != nil && payload.Response.Status != "" {
					status = payload.Response.Status
				} else {
					status = strings.TrimPrefix(event.Event, "response.")
				}
				ch <- StreamChunk{
					Type:         StreamChunkMessageEnd,
					FinishReason: status,
				}
			case "response.error":
				message := "openai streaming error"
				if payload.Error != nil && payload.Error.Message != "" {
					message = payload.Error.Message
				}
				ch <- StreamChunk{
					Type: StreamChunkError,
					Error: &ChunkError{
						Kind:         "upstream_error",
						Message:      message,
						UpstreamCode: payload.ErrorCode(),
					},
				}
			case "response.output_audio.delta":
				ch <- StreamChunk{
					Type: StreamChunkError,
					Error: &ChunkError{
						Kind:    "unsupported",
						Message: "audio streaming is not supported by this adapter",
					},
				}
			}
		}
	}()
	return ch
}

func convertOpenAIChatResponse(payload openAIChatResponse) GenerateOutput {
	if len(payload.Choices) == 0 {
		return GenerateOutput{}
	}
	choice := payload.Choices[0]
	text := extractOpenAIText(choice.Message.Content)
	toolCalls := make([]ToolCall, 0, len(choice.Message.ToolCalls))
	for _, call := range choice.Message.ToolCalls {
		toolCalls = append(toolCalls, ToolCall{
			ID:            call.ID,
			Name:          call.Function.Name,
			ArgumentsJSON: call.Function.Arguments,
		})
	}
	var usage *Usage
	if payload.Usage != nil {
		usage = &Usage{
			InputTokens:  payload.Usage.PromptTokens,
			OutputTokens: payload.Usage.CompletionTokens,
			TotalTokens:  payload.Usage.TotalTokens,
		}
	}
	return GenerateOutput{
		Text:         text,
		ToolCalls:    toolCalls,
		Usage:        usage,
		FinishReason: choice.FinishReason,
		Raw:          payload,
	}
}

func convertResponsesOutput(payload openAIResponsesResponse) GenerateOutput {
	text := ""
	var calls []ToolCall
	appendToolCall := func(id, name, args string) {
		if id == "" {
			id = fmt.Sprintf("tool_%d", len(calls))
		}
		calls = append(calls, ToolCall{
			ID:            id,
			Name:          name,
			ArgumentsJSON: args,
		})
	}
	for _, output := range payload.Output {
		for _, content := range output.Content {
			switch content.Type {
			case "output_text":
				text += content.Text
			case "tool_call":
				args := stringifyArguments(content.Arguments)
				appendToolCall(content.ID, content.Name, args)
			}
		}
	}
	if payload.RequiredAction != nil && payload.RequiredAction.SubmitToolOutputs != nil {
		for _, call := range payload.RequiredAction.SubmitToolOutputs.ToolCalls {
			args := call.Function.Arguments
			appendToolCall(call.ID, call.Function.Name, args)
		}
	}
	return GenerateOutput{
		Text:         text,
		ToolCalls:    calls,
		Usage:        mapResponsesUsage(payload.Usage),
		FinishReason: payload.Status,
		Raw:          payload,
	}
}

func stringifyArguments(value interface{}) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case json.RawMessage:
		return string(v)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		return string(data)
	}
}

func extractOpenAIText(content interface{}) string {
	switch val := content.(type) {
	case string:
		return val
	case []interface{}:
		var buf bytes.Buffer
		for _, item := range val {
			if part, ok := item.(map[string]interface{}); ok {
				if text, ok := part["text"].(string); ok {
					buf.WriteString(text)
				}
			}
		}
		return buf.String()
	default:
		return ""
	}
}

func mapMessagesToChat(messages []Message) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(messages))
	for _, message := range messages {
		if message.Role == "tool" {
			result = append(result, map[string]interface{}{
				"role":         "tool",
				"tool_call_id": message.ToolCallID,
				"content":      joinTextContent(message.Content),
			})
			continue
		}
		parts := make([]interface{}, 0, len(message.Content))
		for _, part := range message.Content {
			if part.Type == "text" {
				parts = append(parts, map[string]string{
					"type": "text",
					"text": part.Text,
				})
			} else if part.Image != nil {
				image := map[string]string{}
				if part.Image.URL != "" {
					image["url"] = part.Image.URL
				}
				if part.Image.Base64 != "" {
					image["b64_json"] = part.Image.Base64
				}
				parts = append(parts, map[string]interface{}{
					"type":      "image_url",
					"image_url": image,
				})
			}
		}
		content := interface{}(parts)
		if len(parts) == 1 {
			if text, ok := parts[0].(map[string]string); ok && text["type"] == "text" {
				content = text["text"]
			}
		}
		result = append(result, map[string]interface{}{
			"role":    message.Role,
			"content": content,
		})
	}
	return result
}

func mapMessagesToResponses(messages []Message) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(messages))
	for _, message := range messages {
		content := make([]map[string]interface{}, 0, len(message.Content))
		for _, part := range message.Content {
			if part.Type == "text" {
				content = append(content, map[string]interface{}{
					"type": "input_text",
					"text": part.Text,
				})
			} else if part.Image != nil {
				entry := map[string]interface{}{
					"type": "input_image",
				}
				if part.Image.URL != "" {
					entry["image_url"] = part.Image.URL
				}
				if part.Image.Base64 != "" {
					entry["image_base64"] = part.Image.Base64
				}
				if part.Image.MediaType != "" {
					entry["media_type"] = part.Image.MediaType
				}
				content = append(content, entry)
			}
		}
		entry := map[string]interface{}{
			"role":    message.Role,
			"content": content,
		}
		if message.ToolCallID != "" {
			entry["tool_call_id"] = message.ToolCallID
		}
		if message.Name != "" {
			entry["name"] = message.Name
		}
		result = append(result, entry)
	}
	return result
}

func joinTextContent(parts []ContentPart) string {
	var buf bytes.Buffer
	for _, part := range parts {
		if part.Type == "text" {
			buf.WriteString(part.Text)
		}
	}
	return buf.String()
}

func mapToolsToOpenAI(tools []ToolDefinition) []map[string]interface{} {
	if len(tools) == 0 {
		return nil
	}
	result := make([]map[string]interface{}, 0, len(tools))
	for _, tool := range tools {
		result = append(result, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  tool.Parameters,
			},
		})
	}
	return result
}

func mapToolChoiceToOpenAI(choice *ToolChoice) interface{} {
	if choice == nil {
		return nil
	}
	if choice.Type == "auto" || choice.Type == "none" {
		return choice.Type
	}
	return map[string]interface{}{
		"type": "function",
		"function": map[string]string{
			"name": choice.Name,
		},
	}
}

func mapResponseFormatToOpenAI(format *ResponseFormat) interface{} {
	if format == nil {
		return nil
	}
	if format.Type == "json_schema" && format.JsonSchema != nil {
		return map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name":   format.JsonSchema.Name,
				"strict": format.JsonSchema.Strict,
				"schema": format.JsonSchema.Schema,
			},
		}
	}
	return map[string]string{"type": "text"}
}

func mapResponsesUsage(usage *struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}) *Usage {
	if usage == nil {
		return nil
	}
	return &Usage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
		TotalTokens:  usage.TotalTokens,
	}
}

func deriveFamily(id string) string {
	parts := strings.Split(id, "-")
	if len(parts) >= 2 {
		return strings.Join(parts[:2], "-")
	}
	return id
}
