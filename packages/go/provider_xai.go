package aikit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"nhooyr.io/websocket"
)

const (
	xaiCompatibilityKey  = "xai:compatibility"
	xaiSpeechModeKey     = "xai:speech-mode"
	xaiDefaultVoice      = "Ara"
	xaiDefaultSampleRate = 24000
	xaiVoiceAgentModelID = "grok-voice"
)

var xaiVoiceSampleRates = []int{8000, 16000, 21050, 24000, 32000, 44100, 48000}

type xaiAdapter struct {
	openai     ProviderAdapter
	anthropic  ProviderAdapter
	mode       string
	speechMode string
	apiKey     string
	baseURL    string
}

func newXAIAdapter(cfg *XAIConfig, client *http.Client) ProviderAdapter {
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.x.ai"
	}
	speechMode := strings.TrimSpace(cfg.SpeechMode)
	if speechMode == "" {
		speechMode = "realtime"
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
		openai:     newOpenAIAdapter(openaiCfg, client, ProviderXAI),
		anthropic:  newAnthropicAdapter(anthropicCfg, client, ProviderXAI),
		mode:       cfg.CompatibilityMode,
		speechMode: speechMode,
		apiKey:     cfg.APIKey,
		baseURL:    base,
	}
}

func (x *xaiAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	models, err := x.openai.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	for _, model := range models {
		if model.ID == xaiVoiceAgentModelID {
			return models, nil
		}
	}
	return append(models, buildXaiVoiceAgentModel()), nil
}

func (x *xaiAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	adapter := x.selectAdapter(in.Metadata)
	return adapter.Generate(ctx, in)
}

func buildXaiVoiceAgentModel() ModelMetadata {
	voiceOptions := []map[string]interface{}{
		{"label": "Ara", "value": "Ara"},
		{"label": "Rex", "value": "Rex"},
		{"label": "Sal", "value": "Sal"},
		{"label": "Eve", "value": "Eve"},
		{"label": "Leo", "value": "Leo"},
	}
	rateOptions := make([]map[string]interface{}, 0, len(xaiVoiceSampleRates))
	for _, rate := range xaiVoiceSampleRates {
		rateOptions = append(rateOptions, map[string]interface{}{
			"label": fmt.Sprintf("%d", rate),
			"value": rate,
		})
	}
	return ModelMetadata{
		ID:          xaiVoiceAgentModelID,
		DisplayName: "Grok Voice Agent",
		Provider:    ProviderXAI,
		Family:      "voice-agent",
		Capabilities: ModelCapabilities{
			Text:             true,
			Vision:           false,
			Image:            false,
			Video:            false,
			ToolUse:          true,
			StructuredOutput: false,
			Reasoning:        false,
			AudioIn:          true,
			AudioOut:         true,
		},
		Inputs: []map[string]interface{}{
			{
				"name":     "voice",
				"label":    "Voice",
				"type":     "select",
				"required": false,
				"options":  voiceOptions,
			},
			{
				"name":     "sample_rate_hz",
				"label":    "Sample rate (Hz)",
				"type":     "select",
				"required": false,
				"default":  xaiDefaultSampleRate,
				"options":  rateOptions,
			},
		},
	}
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

func (x *xaiAdapter) GenerateSpeech(ctx context.Context, in SpeechGenerateInput) (SpeechGenerateOutput, error) {
	mode := x.resolveSpeechMode(in.Metadata)
	if mode == "realtime" {
		return x.generateSpeechRealtime(ctx, in)
	}
	return x.openai.GenerateSpeech(ctx, in)
}

func (x *xaiAdapter) GenerateVideo(ctx context.Context, in VideoGenerateInput) (VideoGenerateOutput, error) {
	_ = ctx
	_ = in
	return VideoGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "xAI video generation is not supported",
		Provider: ProviderXAI,
	}
}

func (x *xaiAdapter) GenerateVoiceAgent(ctx context.Context, in VoiceAgentInput) (VoiceAgentOutput, error) {
	if strings.TrimSpace(x.apiKey) == "" {
		return VoiceAgentOutput{}, &KitError{
			Kind:     ErrorProviderAuth,
			Message:  "xAI api key is required for realtime voice agent",
			Provider: ProviderXAI,
		}
	}
	if strings.TrimSpace(in.UserText) == "" {
		return VoiceAgentOutput{}, &KitError{
			Kind:     ErrorValidation,
			Message:  "Voice agent requires userText",
			Provider: ProviderXAI,
		}
	}

	ctx, cancel := withTimeout(ctx, in.TimeoutMs)
	if cancel != nil {
		defer cancel()
	}

	audio, mime := resolveXaiVoiceAgentAudio(in)
	session := map[string]interface{}{
		"voice": xaiDefaultVoice,
		"turn_detection": map[string]interface{}{
			"type": nil,
		},
		"audio": audio,
	}
	if strings.TrimSpace(in.Voice) != "" {
		session["voice"] = in.Voice
	}
	if strings.TrimSpace(in.Instructions) != "" {
		session["instructions"] = in.Instructions
	}
	if strings.TrimSpace(in.TurnDetection) != "" {
		session["turn_detection"] = map[string]interface{}{
			"type": in.TurnDetection,
		}
	}
	if len(in.Tools) > 0 {
		tools := make([]map[string]interface{}, 0, len(in.Tools))
		for _, tool := range in.Tools {
			tools = append(tools, map[string]interface{}{
				"type":        "function",
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  tool.Parameters,
			})
		}
		session["tools"] = tools
	}
	if in.Parameters != nil {
		if raw, ok := in.Parameters["session"]; ok {
			if overrides, ok := raw.(map[string]interface{}); ok {
				for key, value := range overrides {
					session[key] = value
				}
			}
		}
	}
	response := map[string]interface{}{
		"modalities": []string{"audio", "text"},
	}
	if len(in.ResponseModalities) > 0 {
		response["modalities"] = in.ResponseModalities
	}
	if in.Parameters != nil {
		if raw, ok := in.Parameters["response"]; ok {
			if overrides, ok := raw.(map[string]interface{}); ok {
				for key, value := range overrides {
					response[key] = value
				}
			}
		}
	}

	url, err := resolveRealtimeURL(x.baseURL)
	if err != nil {
		return VoiceAgentOutput{}, err
	}
	header := http.Header{}
	header.Set("Authorization", fmt.Sprintf("Bearer %s", x.apiKey))
	conn, _, err := websocket.Dial(ctx, url, &websocket.DialOptions{
		HTTPHeader: header,
		Subprotocols: []string{
			"realtime",
			"openai-beta.realtime-v1",
		},
	})
	if err != nil {
		return VoiceAgentOutput{}, err
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	sessionSent := false
	responseSent := false
	var audioBuffer bytes.Buffer
	var transcript strings.Builder
	toolCalls := make([]ToolCall, 0)

	for {
		_, data, readErr := conn.Read(ctx)
		if readErr != nil {
			return VoiceAgentOutput{}, readErr
		}
		var event map[string]interface{}
		if err := json.Unmarshal(data, &event); err != nil {
			continue
		}
		eventType, _ := event["type"].(string)
		switch eventType {
		case "conversation.created":
			if !sessionSent {
				sessionSent = true
				if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
					"type":    "session.update",
					"session": session,
				}); err != nil {
					return VoiceAgentOutput{}, err
				}
			}
		case "session.updated":
			if !responseSent {
				responseSent = true
				if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
					"type": "conversation.item.create",
					"item": map[string]interface{}{
						"type": "message",
						"role": "user",
						"content": []map[string]interface{}{
							{
								"type": "input_text",
								"text": in.UserText,
							},
						},
					},
				}); err != nil {
					return VoiceAgentOutput{}, err
				}
				if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
					"type":     "response.create",
					"response": response,
				}); err != nil {
					return VoiceAgentOutput{}, err
				}
			}
		case "response.function_call_arguments.done":
			name, _ := event["name"].(string)
			callID, _ := event["call_id"].(string)
			if callID == "" {
				callID = fmt.Sprintf("call_%d", len(toolCalls)+1)
			}
			var argsJSON string
			if rawArgs, ok := event["arguments"].(string); ok {
				argsJSON = rawArgs
			} else if rawArgs, ok := event["arguments"]; ok && rawArgs != nil {
				encoded, err := json.Marshal(rawArgs)
				if err == nil {
					argsJSON = string(encoded)
				}
			}
			if argsJSON == "" {
				argsJSON = "{}"
			}
			call := ToolCall{
				ID:            callID,
				Name:          name,
				ArgumentsJSON: argsJSON,
			}
			toolCalls = append(toolCalls, call)

			var outputPayload interface{} = map[string]interface{}{"ok": true}
			if in.ToolHandler != nil {
				result, err := in.ToolHandler(ctx, call)
				if err != nil {
					outputPayload = map[string]interface{}{
						"ok":    false,
						"error": err.Error(),
					}
				} else if result != nil {
					outputPayload = result
				} else {
					outputPayload = map[string]interface{}{"ok": true}
				}
			}
			output := ""
			switch value := outputPayload.(type) {
			case string:
				output = value
			default:
				encoded, err := json.Marshal(value)
				if err != nil {
					encoded = []byte(`{"ok":false,"error":"tool_output_encode_failed"}`)
				}
				output = string(encoded)
			}
			if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
				"type": "conversation.item.create",
				"item": map[string]interface{}{
					"type":   "function_call_output",
					"call_id": callID,
					"output": output,
				},
			}); err != nil {
				return VoiceAgentOutput{}, err
			}
			if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
				"type":     "response.create",
				"response": response,
			}); err != nil {
				return VoiceAgentOutput{}, err
			}
		case "response.output_audio_transcript.delta":
			if delta, ok := event["delta"].(string); ok && delta != "" {
				transcript.WriteString(delta)
			}
		case "response.output_audio.delta":
			delta, _ := event["delta"].(string)
			if delta == "" {
				continue
			}
			decoded, err := base64.StdEncoding.DecodeString(delta)
			if err != nil {
				return VoiceAgentOutput{}, err
			}
			audioBuffer.Write(decoded)
		case "response.output_audio.done", "response.done":
			out := VoiceAgentOutput{
				Transcript: strings.TrimSpace(transcript.String()),
			}
			if audioBuffer.Len() > 0 {
				out.Audio = &SpeechGenerateOutput{
					Mime: mime,
					Data: base64.StdEncoding.EncodeToString(audioBuffer.Bytes()),
				}
			}
			if len(toolCalls) > 0 {
				out.ToolCalls = toolCalls
			}
			return out, nil
		case "error", "response.error":
			message := "xAI realtime error"
			if errObj, ok := event["error"].(map[string]interface{}); ok {
				if value, ok := errObj["message"].(string); ok && value != "" {
					message = value
				}
			}
			return VoiceAgentOutput{}, &KitError{
				Kind:     ErrorUnknown,
				Message:  message,
				Provider: ProviderXAI,
			}
		}
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
	adapter := x.selectAdapter(in.Metadata)
	return adapter.Stream(ctx, in)
}

func (x *xaiAdapter) selectAdapter(metadata map[string]string) ProviderAdapter {
	mode := x.mode
	if metadata != nil {
		if value, ok := metadata[xaiCompatibilityKey]; ok {
			mode = value
		}
	}
	if mode == "anthropic" {
		return x.anthropic
	}
	return x.openai
}

func (x *xaiAdapter) resolveSpeechMode(metadata map[string]string) string {
	mode := x.speechMode
	if metadata != nil {
		if value, ok := metadata[xaiSpeechModeKey]; ok {
			mode = value
		}
	}
	if strings.EqualFold(mode, "openai") {
		return "openai"
	}
	return "realtime"
}

func (x *xaiAdapter) generateSpeechRealtime(ctx context.Context, in SpeechGenerateInput) (SpeechGenerateOutput, error) {
	if strings.TrimSpace(x.apiKey) == "" {
		return SpeechGenerateOutput{}, &KitError{
			Kind:     ErrorProviderAuth,
			Message:  "xAI api key is required for realtime speech",
			Provider: ProviderXAI,
		}
	}
	formatType, sampleRate, mime, sessionOverrides, responseOverrides, err := resolveXaiSpeechOptions(in)
	if err != nil {
		return SpeechGenerateOutput{}, err
	}
	session := map[string]interface{}{
		"voice": xaiDefaultVoice,
		"turn_detection": map[string]interface{}{
			"type": nil,
		},
		"audio": map[string]interface{}{
			"output": map[string]interface{}{
				"format": map[string]interface{}{
					"type": formatType,
					"rate": sampleRate,
				},
			},
		},
	}
	if strings.TrimSpace(in.Voice) != "" {
		session["voice"] = in.Voice
	}
	for key, value := range sessionOverrides {
		session[key] = value
	}
	response := map[string]interface{}{
		"modalities": []string{"audio"},
	}
	for key, value := range responseOverrides {
		response[key] = value
	}
	url, err := resolveRealtimeURL(x.baseURL)
	if err != nil {
		return SpeechGenerateOutput{}, err
	}
	header := http.Header{}
	header.Set("Authorization", fmt.Sprintf("Bearer %s", x.apiKey))
	conn, _, err := websocket.Dial(ctx, url, &websocket.DialOptions{
		HTTPHeader: header,
		Subprotocols: []string{
			"realtime",
			"openai-beta.realtime-v1",
		},
	})
	if err != nil {
		return SpeechGenerateOutput{}, err
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	sessionSent := false
	responseSent := false
	var audio bytes.Buffer

	for {
		_, data, readErr := conn.Read(ctx)
		if readErr != nil {
			return SpeechGenerateOutput{}, readErr
		}
		var event map[string]interface{}
		if err := json.Unmarshal(data, &event); err != nil {
			continue
		}
		eventType, _ := event["type"].(string)
		switch eventType {
		case "conversation.created":
			if !sessionSent {
				sessionSent = true
				if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
					"type":    "session.update",
					"session": session,
				}); err != nil {
					return SpeechGenerateOutput{}, err
				}
			}
		case "session.updated":
			if !responseSent {
				responseSent = true
				if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
					"type": "conversation.item.create",
					"item": map[string]interface{}{
						"type": "message",
						"role": "user",
						"content": []map[string]interface{}{
							{
								"type": "input_text",
								"text": in.Text,
							},
						},
					},
				}); err != nil {
					return SpeechGenerateOutput{}, err
				}
				if err := writeRealtimeEvent(ctx, conn, map[string]interface{}{
					"type":     "response.create",
					"response": response,
				}); err != nil {
					return SpeechGenerateOutput{}, err
				}
			}
		case "response.output_audio.delta":
			delta, _ := event["delta"].(string)
			if delta == "" {
				continue
			}
			decoded, err := base64.StdEncoding.DecodeString(delta)
			if err != nil {
				return SpeechGenerateOutput{}, err
			}
			audio.Write(decoded)
		case "response.output_audio.done", "response.done":
			return SpeechGenerateOutput{
				Mime: mime,
				Data: base64.StdEncoding.EncodeToString(audio.Bytes()),
			}, nil
		case "error", "response.error":
			message := "xAI realtime error"
			if errObj, ok := event["error"].(map[string]interface{}); ok {
				if value, ok := errObj["message"].(string); ok && value != "" {
					message = value
				}
			}
			return SpeechGenerateOutput{}, &KitError{
				Kind:     ErrorUnknown,
				Message:  message,
				Provider: ProviderXAI,
			}
		}
	}
}

func writeRealtimeEvent(ctx context.Context, conn *websocket.Conn, payload map[string]interface{}) error {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, encoded)
}

func resolveRealtimeURL(base string) (string, error) {
	base = strings.TrimSpace(base)
	if base == "" {
		base = "https://api.x.ai"
	}
	parsed, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	if parsed.Scheme == "http" {
		parsed.Scheme = "ws"
	} else {
		parsed.Scheme = "wss"
	}
	parsed.Path = "/v1/realtime"
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func resolveXaiSpeechOptions(in SpeechGenerateInput) (string, int, string, map[string]interface{}, map[string]interface{}, error) {
	format := strings.TrimSpace(in.ResponseFormat)
	if format == "" {
		format = strings.TrimSpace(in.Format)
	}
	format = strings.ToLower(format)
	formatType := "audio/pcm"
	mime := "audio/pcm"
	if format != "" && format != "pcm" {
		if format == "pcmu" {
			formatType = "audio/pcmu"
			mime = "audio/pcmu"
		} else if format == "pcma" {
			formatType = "audio/pcma"
			mime = "audio/pcma"
		} else {
			return "", 0, "", nil, nil, &KitError{
				Kind:     ErrorUnsupported,
				Message:  fmt.Sprintf("xAI realtime speech only supports pcm/pcmu/pcma output (received %s)", format),
				Provider: ProviderXAI,
			}
		}
	}
	sampleRate := xaiDefaultSampleRate
	if formatType != "audio/pcm" {
		sampleRate = 8000
	} else if in.Parameters != nil {
		if rate, ok := coerceNumber(in.Parameters["sampleRate"]); ok {
			sampleRate = rate
		}
	}
	sessionOverrides := map[string]interface{}{}
	responseOverrides := map[string]interface{}{}
	if in.Parameters != nil {
		if raw, ok := in.Parameters["session"]; ok {
			if value, ok := raw.(map[string]interface{}); ok {
				sessionOverrides = value
			}
		}
		if raw, ok := in.Parameters["response"]; ok {
			if value, ok := raw.(map[string]interface{}); ok {
				responseOverrides = value
			}
		}
	}
	return formatType, sampleRate, mime, sessionOverrides, responseOverrides, nil
}

func resolveXaiVoiceAgentAudio(in VoiceAgentInput) (map[string]interface{}, string) {
	formatType := "audio/pcm"
	outputRate := 0
	if in.Audio != nil && in.Audio.Output != nil && in.Audio.Output.Format != nil {
		if value := strings.TrimSpace(in.Audio.Output.Format.Type); value != "" {
			formatType = value
		}
		if in.Audio.Output.Format.Rate > 0 {
			outputRate = in.Audio.Output.Format.Rate
		}
	}
	paramsRate := 0
	if in.Parameters != nil {
		if rate, ok := coerceNumber(in.Parameters["sampleRate"]); ok {
			paramsRate = rate
		}
	}
	sampleRate := xaiDefaultSampleRate
	if formatType != "audio/pcm" {
		sampleRate = 8000
	} else if outputRate > 0 {
		sampleRate = outputRate
	} else if paramsRate > 0 {
		sampleRate = paramsRate
	}
	if outputRate <= 0 {
		outputRate = sampleRate
	}
	inputType := formatType
	inputRate := outputRate
	if in.Audio != nil && in.Audio.Input != nil && in.Audio.Input.Format != nil {
		if value := strings.TrimSpace(in.Audio.Input.Format.Type); value != "" {
			inputType = value
		}
		if in.Audio.Input.Format.Rate > 0 {
			inputRate = in.Audio.Input.Format.Rate
		}
	}
	audio := map[string]interface{}{
		"input": map[string]interface{}{
			"format": map[string]interface{}{
				"type": inputType,
				"rate": inputRate,
			},
		},
		"output": map[string]interface{}{
			"format": map[string]interface{}{
				"type": formatType,
				"rate": outputRate,
			},
		},
	}
	return audio, formatType
}

func coerceNumber(value interface{}) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, true
	case int64:
		return int(v), true
	case float64:
		if v != 0 {
			return int(v), true
		}
	}
	return 0, false
}

func withTimeout(ctx context.Context, timeoutMs int) (context.Context, context.CancelFunc) {
	if timeoutMs <= 0 {
		return ctx, nil
	}
	return context.WithTimeout(ctx, time.Duration(timeoutMs)*time.Millisecond)
}
