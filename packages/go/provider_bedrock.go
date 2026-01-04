package aikit

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"
)

type bedrockAdapter struct {
	provider Provider
	config   *BedrockConfig
	client   *http.Client
}

type bedrockModelSummary struct {
	ModelID          string   `json:"modelId"`
	ModelName        string   `json:"modelName"`
	ProviderName     string   `json:"providerName"`
	InputModalities  []string `json:"inputModalities"`
	OutputModalities []string `json:"outputModalities"`
}

type bedrockListResponse struct {
	ModelSummaries []bedrockModelSummary `json:"modelSummaries"`
	Models         []bedrockModelSummary `json:"models"`
}

type bedrockContentBlock struct {
	Text    string `json:"text"`
	ToolUse *struct {
		ToolUseID string      `json:"toolUseId"`
		Name      string      `json:"name"`
		Input     interface{} `json:"input"`
	} `json:"toolUse"`
}

type bedrockConverseResponse struct {
	Output *struct {
		Message *struct {
			Content []bedrockContentBlock `json:"content"`
		} `json:"message"`
	} `json:"output"`
	StopReason string `json:"stopReason"`
	Usage      *struct {
		InputTokens  int `json:"inputTokens"`
		OutputTokens int `json:"outputTokens"`
		TotalTokens  int `json:"totalTokens"`
	} `json:"usage"`
}

type awsCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

func newBedrockAdapter(cfg *BedrockConfig, client *http.Client) ProviderAdapter {
	return &bedrockAdapter{
		provider: ProviderBedrock,
		config:   cfg,
		client:   client,
	}
}

func resolveBedrockConfig(cfg *BedrockConfig) (*BedrockConfig, error) {
	if cfg == nil {
		return nil, fmt.Errorf("bedrock config is required")
	}
	resolved := *cfg
	if strings.TrimSpace(resolved.Region) == "" {
		resolved.Region = strings.TrimSpace(os.Getenv("AWS_REGION"))
		if resolved.Region == "" {
			resolved.Region = strings.TrimSpace(os.Getenv("AWS_DEFAULT_REGION"))
		}
	}
	if resolved.Region == "" {
		return nil, fmt.Errorf("bedrock region is required")
	}
	if strings.TrimSpace(resolved.AccessKeyID) == "" {
		resolved.AccessKeyID = strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID"))
	}
	if strings.TrimSpace(resolved.SecretAccessKey) == "" {
		resolved.SecretAccessKey = strings.TrimSpace(os.Getenv("AWS_SECRET_ACCESS_KEY"))
	}
	if strings.TrimSpace(resolved.SessionToken) == "" {
		resolved.SessionToken = strings.TrimSpace(os.Getenv("AWS_SESSION_TOKEN"))
	}
	if resolved.AccessKeyID == "" || resolved.SecretAccessKey == "" {
		return nil, fmt.Errorf("bedrock AWS credentials are required")
	}
	if strings.TrimSpace(resolved.Endpoint) == "" {
		resolved.Endpoint = fmt.Sprintf("https://bedrock.%s.amazonaws.com", resolved.Region)
	}
	if strings.TrimSpace(resolved.RuntimeEndpoint) == "" {
		resolved.RuntimeEndpoint = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", resolved.Region)
	}
	if strings.TrimSpace(resolved.ControlPlaneService) == "" {
		resolved.ControlPlaneService = "bedrock"
	}
	if strings.TrimSpace(resolved.RuntimeService) == "" {
		resolved.RuntimeService = "bedrock-runtime"
	}
	return &resolved, nil
}

func (a *bedrockAdapter) ListModels(ctx context.Context) ([]ModelMetadata, error) {
	url := strings.TrimRight(a.config.Endpoint, "/") + "/foundation-models"
	var payload bedrockListResponse
	if err := a.doSignedJSON(ctx, http.MethodGet, url, nil, &payload, a.config.ControlPlaneService); err != nil {
		return nil, err
	}
	summaries := payload.ModelSummaries
	if len(summaries) == 0 {
		summaries = payload.Models
	}
	models := make([]ModelMetadata, 0, len(summaries))
	for _, summary := range summaries {
		modelID := strings.TrimSpace(summary.ModelID)
		if modelID == "" {
			continue
		}
		textOut := containsModality(summary.OutputModalities, "TEXT")
		visionIn := containsModality(summary.InputModalities, "IMAGE")
		imageOut := containsModality(summary.OutputModalities, "IMAGE")
		models = append(models, ModelMetadata{
			ID:          modelID,
			DisplayName: firstNonEmpty(summary.ModelName, modelID),
			Provider:    a.provider,
			Family:      deriveBedrockFamily(modelID, summary.ProviderName),
			Capabilities: ModelCapabilities{
				Text:             textOut,
				Vision:           visionIn,
				Image:            imageOut,
				ToolUse:          supportsBedrockToolUse(modelID),
				StructuredOutput: false,
				Reasoning:        false,
			},
		})
	}
	return models, nil
}

func (a *bedrockAdapter) Generate(ctx context.Context, in GenerateInput) (GenerateOutput, error) {
	if in.ResponseFormat != nil && in.ResponseFormat.Type == "json_schema" {
		return GenerateOutput{}, &KitError{
			Kind:     ErrorUnsupported,
			Message:  "Bedrock structured outputs are not supported",
			Provider: a.provider,
		}
	}
	payload, err := a.buildPayload(ctx, in)
	if err != nil {
		return GenerateOutput{}, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return GenerateOutput{}, err
	}
	url := strings.TrimRight(a.config.RuntimeEndpoint, "/") + "/model/" + url.PathEscape(in.Model) + "/converse"
	var response bedrockConverseResponse
	if err := a.doSignedJSON(ctx, http.MethodPost, url, body, &response, a.config.RuntimeService); err != nil {
		return GenerateOutput{}, err
	}
	return normalizeBedrockConverse(response), nil
}

func (a *bedrockAdapter) GenerateImage(ctx context.Context, in ImageGenerateInput) (ImageGenerateOutput, error) {
	return ImageGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Bedrock image generation is not supported",
		Provider: a.provider,
	}
}

func (a *bedrockAdapter) GenerateMesh(ctx context.Context, in MeshGenerateInput) (MeshGenerateOutput, error) {
	return MeshGenerateOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Bedrock mesh generation is not supported",
		Provider: a.provider,
	}
}

func (a *bedrockAdapter) Transcribe(ctx context.Context, in TranscribeInput) (TranscribeOutput, error) {
	return TranscribeOutput{}, &KitError{
		Kind:     ErrorUnsupported,
		Message:  "Bedrock transcription is not supported",
		Provider: a.provider,
	}
}

func (a *bedrockAdapter) Stream(ctx context.Context, in GenerateInput) (<-chan StreamChunk, error) {
	output, err := a.Generate(ctx, in)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamChunk, 1+len(output.ToolCalls)+1)
	if output.Text != "" {
		ch <- StreamChunk{Type: StreamChunkDelta, TextDelta: output.Text}
	}
	for _, call := range output.ToolCalls {
		call := call
		ch <- StreamChunk{Type: StreamChunkToolCall, Call: &call}
	}
	ch <- StreamChunk{
		Type:         StreamChunkMessageEnd,
		Usage:        output.Usage,
		FinishReason: output.FinishReason,
	}
	close(ch)
	return ch, nil
}

func (a *bedrockAdapter) buildPayload(ctx context.Context, in GenerateInput) (map[string]interface{}, error) {
	messages := make([]map[string]interface{}, 0, len(in.Messages))
	system := make([]map[string]string, 0)
	for _, msg := range in.Messages {
		switch msg.Role {
		case "system":
			text := collectText(msg.Content)
			if text != "" {
				system = append(system, map[string]string{"text": text})
			}
		case "tool":
			content := make([]map[string]interface{}, 0)
			toolContent := make([]map[string]string, 0)
			for _, part := range msg.Content {
				if part.Type == "text" {
					toolContent = append(toolContent, map[string]string{"text": part.Text})
				}
			}
			content = append(content, map[string]interface{}{
				"toolResult": map[string]interface{}{
					"toolUseId": msg.ToolCallID,
					"content":   toolContent,
				},
			})
			messages = append(messages, map[string]interface{}{
				"role":    "user",
				"content": content,
			})
		default:
			content := make([]map[string]interface{}, 0, len(msg.Content))
			for _, part := range msg.Content {
				switch part.Type {
				case "text":
					content = append(content, map[string]interface{}{"text": part.Text})
				case "image":
					if part.Image == nil {
						continue
					}
					imageBlock, err := a.resolveImageBlock(ctx, part.Image)
					if err != nil {
						return nil, err
					}
					content = append(content, map[string]interface{}{"image": imageBlock})
				}
			}
			role := "user"
			if msg.Role == "assistant" {
				role = "assistant"
			}
			messages = append(messages, map[string]interface{}{
				"role":    role,
				"content": content,
			})
		}
	}
	payload := map[string]interface{}{
		"messages": messages,
	}
	if len(system) > 0 {
		payload["system"] = system
	}
	inferenceConfig := map[string]interface{}{}
	if in.MaxTokens != nil {
		inferenceConfig["maxTokens"] = *in.MaxTokens
	}
	if in.Temperature != nil {
		inferenceConfig["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		inferenceConfig["topP"] = *in.TopP
	}
	if len(inferenceConfig) > 0 {
		payload["inferenceConfig"] = inferenceConfig
	}
	if len(in.Tools) > 0 {
		if toolConfig := buildBedrockToolConfig(in.Tools, in.ToolChoice); toolConfig != nil {
			payload["toolConfig"] = toolConfig
		}
	}
	return payload, nil
}

func (a *bedrockAdapter) resolveImageBlock(ctx context.Context, image *ImageContent) (map[string]interface{}, error) {
	base64Data := strings.TrimSpace(image.Base64)
	mediaType := strings.TrimSpace(image.MediaType)
	if base64Data == "" && strings.TrimSpace(image.URL) != "" {
		var err error
		mediaType, base64Data, err = fetchURLAsBase64(ctx, a.client, image.URL)
		if err != nil {
			return nil, err
		}
	}
	if base64Data == "" {
		return nil, &KitError{
			Kind:     ErrorValidation,
			Message:  "Bedrock image content requires base64 data or URL",
			Provider: a.provider,
		}
	}
	format := bedrockImageFormat(mediaType, image.URL)
	return map[string]interface{}{
		"format": format,
		"source": map[string]interface{}{
			"bytes": base64Data,
		},
	}, nil
}

func (a *bedrockAdapter) doSignedJSON(ctx context.Context, method, url string, body []byte, out interface{}, service string) error {
	resp, err := a.doSignedRequest(ctx, method, url, body, service)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(out)
}

func (a *bedrockAdapter) doSignedRequest(ctx context.Context, method, url string, body []byte, service string) (*http.Response, error) {
	if body == nil {
		body = []byte{}
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if err := signAWSRequest(req, body, a.config, service); err != nil {
		return nil, err
	}
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		payload, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, &KitError{
			Kind:           classifyStatus(resp.StatusCode),
			Message:        strings.TrimSpace(string(payload)),
			Provider:       a.provider,
			UpstreamStatus: resp.StatusCode,
			RequestID:      resp.Header.Get("x-amzn-requestid"),
		}
	}
	return resp, nil
}

func signAWSRequest(req *http.Request, body []byte, cfg *BedrockConfig, service string) error {
	creds := awsCredentials{
		AccessKeyID:     cfg.AccessKeyID,
		SecretAccessKey: cfg.SecretAccessKey,
		SessionToken:    cfg.SessionToken,
	}
	now := time.Now().UTC()
	amzDate := now.Format("20060102T150405Z")
	dateStamp := now.Format("20060102")
	payloadHash := sha256Hex(body)
	req.Header.Set("x-amz-date", amzDate)
	req.Header.Set("x-amz-content-sha256", payloadHash)
	if creds.SessionToken != "" {
		req.Header.Set("x-amz-security-token", creds.SessionToken)
	}
	req.Host = req.URL.Host
	req.Header.Set("Host", req.URL.Host)
	canonicalHeaders, signedHeaders := canonicalizeHeaders(req.Header)
	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI(req.URL.EscapedPath()),
		canonicalQuery(req.URL.Query()),
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")
	scope := fmt.Sprintf("%s/%s/%s/aws4_request", dateStamp, cfg.Region, service)
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzDate,
		scope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")
	signingKey := deriveSigningKey(creds.SecretAccessKey, dateStamp, cfg.Region, service)
	signature := hmacSHA256Hex(signingKey, stringToSign)
	authHeader := fmt.Sprintf(
		"AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		creds.AccessKeyID,
		scope,
		signedHeaders,
		signature,
	)
	req.Header.Set("Authorization", authHeader)
	return nil
}

func canonicalizeHeaders(headers http.Header) (string, string) {
	normalized := map[string][]string{}
	for key, values := range headers {
		lower := strings.ToLower(key)
		normalized[lower] = append(normalized[lower], values...)
	}
	keys := make([]string, 0, len(normalized))
	for key := range normalized {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var canonical strings.Builder
	signed := make([]string, 0, len(keys))
	for _, key := range keys {
		values := normalized[key]
		for i, value := range values {
			values[i] = normalizeHeaderValue(value)
		}
		canonical.WriteString(key)
		canonical.WriteString(":")
		canonical.WriteString(strings.Join(values, ","))
		canonical.WriteString("\n")
		signed = append(signed, key)
	}
	return canonical.String(), strings.Join(signed, ";")
}

func canonicalURI(path string) string {
	if path == "" {
		return "/"
	}
	segments := strings.Split(path, "/")
	for i, segment := range segments {
		segments[i] = encodeRFC3986(segment)
	}
	return strings.Join(segments, "/")
}

func canonicalQuery(values url.Values) string {
	if len(values) == 0 {
		return ""
	}
	type pair struct {
		key   string
		value string
	}
	pairs := make([]pair, 0)
	for key, vals := range values {
		for _, value := range vals {
			pairs = append(pairs, pair{key: key, value: value})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].key == pairs[j].key {
			return pairs[i].value < pairs[j].value
		}
		return pairs[i].key < pairs[j].key
	})
	out := make([]string, 0, len(pairs))
	for _, entry := range pairs {
		out = append(out, fmt.Sprintf("%s=%s", encodeRFC3986(entry.key), encodeRFC3986(entry.value)))
	}
	return strings.Join(out, "&")
}

func encodeRFC3986(value string) string {
	escaped := url.QueryEscape(value)
	escaped = strings.ReplaceAll(escaped, "+", "%20")
	escaped = strings.ReplaceAll(escaped, "%7E", "~")
	return escaped
}

func normalizeHeaderValue(value string) string {
	return strings.Join(strings.Fields(value), " ")
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func hmacSHA256Hex(key []byte, data string) string {
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(data))
	return hex.EncodeToString(mac.Sum(nil))
}

func deriveSigningKey(secret, dateStamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), dateStamp)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	return hmacSHA256(kService, "aws4_request")
}

func hmacSHA256(key []byte, data string) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(data))
	return mac.Sum(nil)
}

func normalizeBedrockConverse(response bedrockConverseResponse) GenerateOutput {
	textParts := []string{}
	toolCalls := []ToolCall{}
	if response.Output != nil && response.Output.Message != nil {
		for _, block := range response.Output.Message.Content {
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
			if block.ToolUse != nil {
				args, _ := json.Marshal(block.ToolUse.Input)
				toolCalls = append(toolCalls, ToolCall{
					ID:            block.ToolUse.ToolUseID,
					Name:          block.ToolUse.Name,
					ArgumentsJSON: string(args),
				})
			}
		}
	}
	var usage *Usage
	if response.Usage != nil {
		usage = &Usage{
			InputTokens:  response.Usage.InputTokens,
			OutputTokens: response.Usage.OutputTokens,
			TotalTokens:  response.Usage.TotalTokens,
		}
	}
	finishReason := response.StopReason
	if finishReason == "" {
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		} else {
			finishReason = "stop"
		}
	}
	output := GenerateOutput{
		Text:         strings.Join(textParts, ""),
		ToolCalls:    toolCalls,
		Usage:        usage,
		FinishReason: finishReason,
		Raw:          response,
	}
	if output.Text == "" {
		output.Text = ""
	}
	if len(toolCalls) == 0 {
		output.ToolCalls = nil
	}
	return output
}

func buildBedrockToolConfig(tools []ToolDefinition, choice *ToolChoice) map[string]interface{} {
	if len(tools) == 0 {
		return nil
	}
	mapped := make([]map[string]interface{}, 0, len(tools))
	for _, tool := range tools {
		mapped = append(mapped, map[string]interface{}{
			"toolSpec": map[string]interface{}{
				"name":        tool.Name,
				"description": tool.Description,
				"inputSchema": map[string]interface{}{
					"json": tool.Parameters,
				},
			},
		})
	}
	config := map[string]interface{}{
		"tools": mapped,
	}
	if choice != nil {
		config["toolChoice"] = normalizeBedrockToolChoice(choice)
	}
	return config
}

func normalizeBedrockToolChoice(choice *ToolChoice) map[string]interface{} {
	switch choice.Type {
	case "auto":
		return map[string]interface{}{"auto": map[string]interface{}{}}
	case "none":
		return map[string]interface{}{"none": map[string]interface{}{}}
	case "tool":
		if choice.Name == "" {
			return map[string]interface{}{"auto": map[string]interface{}{}}
		}
		return map[string]interface{}{"tool": map[string]interface{}{"name": choice.Name}}
	default:
		return map[string]interface{}{"auto": map[string]interface{}{}}
	}
}

func collectText(parts []ContentPart) string {
	var out strings.Builder
	for _, part := range parts {
		if part.Type == "text" {
			out.WriteString(part.Text)
		}
	}
	return out.String()
}

func bedrockImageFormat(mediaType string, urlValue string) string {
	lower := strings.ToLower(strings.TrimSpace(mediaType))
	switch {
	case strings.Contains(lower, "png"):
		return "png"
	case strings.Contains(lower, "jpeg"), strings.Contains(lower, "jpg"):
		return "jpeg"
	case strings.Contains(lower, "webp"):
		return "webp"
	case strings.Contains(lower, "gif"):
		return "gif"
	}
	ext := strings.ToLower(strings.TrimSpace(urlValue))
	switch {
	case strings.HasSuffix(ext, ".png"):
		return "png"
	case strings.HasSuffix(ext, ".jpg"), strings.HasSuffix(ext, ".jpeg"):
		return "jpeg"
	case strings.HasSuffix(ext, ".webp"):
		return "webp"
	case strings.HasSuffix(ext, ".gif"):
		return "gif"
	}
	return "png"
}

func containsModality(modalities []string, target string) bool {
	for _, item := range modalities {
		if strings.EqualFold(item, target) {
			return true
		}
	}
	return false
}

func supportsBedrockToolUse(modelID string) bool {
	return strings.HasPrefix(modelID, "anthropic.") || strings.HasPrefix(modelID, "cohere.")
}

func deriveBedrockFamily(modelID, providerName string) string {
	if providerName != "" {
		return strings.ToLower(providerName)
	}
	parts := strings.Split(modelID, ".")
	if len(parts) > 0 && parts[0] != "" {
		return parts[0]
	}
	return modelID
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
