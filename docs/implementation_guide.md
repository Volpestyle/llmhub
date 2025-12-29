Below is an “implementation‑details” doc that turns your design into concrete, up‑to‑date API contracts and adapter behaviors for OpenAI, xAI (Grok), Anthropic, and Google Gemini—with request/response shapes, streaming semantics, tool/JSON‑schema support, model‑listing endpoints, and error hints you can code against in Go and TypeScript right now.

Last verified: Oct 25, 2025.
Notes: where providers surface multiple overlapping APIs, this doc recommends one primary surface (e.g., OpenAI Responses API) and explains alternates you may still want to support for compatibility.

---

## 0. Quick capabilities table (what we rely on in v1)

| Provider      | Primary list-models  | Primary generate                                                                                                 | Streaming                                                | Tools / Functions                                                                                | Structured output                                                         | Vision in chat                                        | Notes                                                                    |
| ------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------ |
| OpenAI        | `GET /v1/models`     | Responses API (`POST /v1/responses`), alt: `POST /v1/chat/completions`                                           | SSE (typed events for Responses; deltas for Chat)        | `tools: [{ type: "function", function: { ... } }]`                                               | `response_format: { type: "json_schema", json_schema: { ... } }` (strict) | In Responses via input items; in Chat via `image_url` | OpenAPI spec repo published; SDKs map to both APIs.                      |
| xAI (Grok)    | `GET /v1/models`     | OpenAI‑compatible `POST /v1/chat/completions` and `POST /v1/responses`; Anthropic‑compatible `POST /v1/messages` | SSE                                                      | OpenAI‑style tools; Anthropic‑style tools via `/v1/messages`                                     | In Responses (OpenAI‑style)                                               | Vision variants available per model                   | Explicitly “OpenAI & Anthropic compatible.” Base URL `https://api.x.ai`. |
| Anthropic     | `GET /v1/models`     | `POST /v1/messages` (header `anthropic-version`)                                                                 | SSE event stream (typed events)                          | `[{ name, description, input_schema }]` → model emits `tool_use`; app replies with `tool_result` | JSON schema mode (see Notes)                                              | Image parts in `messages[].content[]`                 | Official docs use `input_schema` for tools; streaming exposed via SDK.   |
| Google Gemini | `GET /v1beta/models` | `POST /v1beta/models/{model}:generateContent` (aka `models.generateContent`)                                     | Streaming variant (`:streamGenerateContent` via SDK/SSE) | Tool calling via `tools.functionDeclarations` and `toolConfig`; app returns `functionResponse`   | `generationConfig.responseMimeType` + `responseSchema`                    | Vision via parts (file/inline data)                   | Models expose `supportedGenerationMethods` and token limits.             |

---

## 1. Model registry: endpoints and fields you can trust

### OpenAI

- List: GET https://api.openai.com/v1/models (auth: Authorization: Bearer)
  The official OpenAPI spec is published in the OpenAI GitHub repo (generated via Stainless). Use it for shapes and pagination behavior. In practice, you’ll still need an allowlist/overlay to filter to chat/Responses‑compatible models. ￼

### xAI (Grok)

- List: GET https://api.x.ai/v1/models (auth: Authorization: Bearer)
  xAI states full API compatibility with OpenAI and also exposes Anthropic‑compatible surfaces. Use this endpoint for IDs; capabilities often inferred from model name and docs. ￼

### Anthropic

- List: GET https://api.anthropic.com/v1/models
  Headers: x-api-key, anthropic-version: 2023-06-01 (current required version header). Response includes data[] with model IDs; enrich with your own capability overlay. ￼

### Google Gemini

- List: GET https://generativelanguage.googleapis.com/v1beta/models (auth: x-goog-api-key)
  Model fields include supportedGenerationMethods[], inputTokenLimit, outputTokenLimit. Filter to those that include generateContent. ￼

### Registry implementation notes

- Cache each provider’s list with a default 6h TTL and background refresh.
- For Gemini, store supportedGenerationMethods to reliably filter text/vision models. For OpenAI/Anthropic/xAI the list responses are more generic; apply a curated allowlist overlay (JSON in repo) to mark capabilities and pricing, then reconcile with live lists. ￼
- To keep the overlay current without hammering `/models`, run `npm run refresh:models` from `packages/node` (set `OPENAI_API_KEY` in `packages/node/.env`/`.env.local`). The script renders provider pricing pages, parses explicit pricing/capability data via an LLM, and rewrites `models/curated_models.json` (no heuristics). Both the Go and Node runtimes serve the same snapshot by default; append `?refresh=true` to `/provider-models` when you explicitly need a live refetch. ￼

---

## 2. Unified Generate adapters (requests, streaming, tools, structured output)

Below, “ours” refers to your domain types in the design doc. For each provider we show how to translate Messages, Tools, Structured output, Streaming, and Vision inputs.

### 2.1 OpenAI

Primary API: Responses API (POST /v1/responses)
Alternates: POST /v1/chat/completions (legacy but supported indefinitely). Use Responses API when you need structured outputs and typed streaming events; keep Chat Completions for widest ecosystem interoperability. ￼

Auth: Authorization: Bearer <OPENAI_API_KEY>

Messages mapping

- Responses API accepts input items (not classic messages). To preserve your {role, content[]} shape, build an input array with user and system content mapped to OpenAI’s items (e.g., type: "input_text"). If you prefer, keep chat.completions for simpler role mapping: { role: "system" | "user" | "assistant" | "tool", content }. ￼

Tool calling

- Request:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "...",
        "parameters": { "type": "object", "properties": {}, "required": [] }
      }
    }
  ],
  "tool_choice": "auto"
}
```

Response (Chat Completions): assistant message includes tool_calls[] with function.name and JSON arguments. You then send a follow‑up tool role message with tool_call_id and the tool result. (Exactly as in the official Cookbook.) ￼

Structured outputs

- Prefer the Responses API response_format: { "type": "json_schema", "json_schema": { "name": "...", "strict": true, "schema": { ... } } }. This performs constrained decoding to match your schema exactly. (Introduced Aug 6, 2024.) ￼

Streaming

- Responses API: SSE stream of typed events like response.output_text.delta, tool events, and completion events. Your Node adapter should iterate the async stream and fold deltas into your StreamChunk (type: 'delta' | 'tool_call' | ...); your Go adapter should parse SSE event names and JSON payloads similarly. ￼
- Chat Completions: SSE “data:” lines carrying chat.completion.chunk objects with .choices[].delta. Handle delta.tool_calls[n].function.arguments chunks by accumulating per tool‑call. (OpenAI Node SDK readme shows streaming usage; same transport.) ￼

Vision inputs

- Chat Completions: message content array entries with { type: "image_url", image_url: { url | b64 } }.
- Responses API: supply image parts via input items (e.g., {"type":"input_image", ...}) then stream/output text. (General capability noted in Responses API/SDK docs.) ￼

Errors

- The official SDK enumerates error classes (AuthenticationError, RateLimitError, NotFoundError, etc.). Map to your ErrorKind with upstreamStatus, x-request-id. ￼

---

### 2.2 xAI (Grok)

Base URL: https://api.x.ai
Auth: Authorization: Bearer <XAI_API_KEY>
Available endpoints (OpenAI‑compatible): POST /v1/chat/completions, POST /v1/responses, GET /v1/models
Anthropic‑compatible: POST /v1/messages
(Plus additional endpoints like :deferred-completion, tokenizer, images.) ￼

Compatibility

- xAI explicitly documents compatibility with both OpenAI and Anthropic SDKs. For us this means the same adapter logic we implement for OpenAI Chat/Responses and Anthropic Messages can be reused for xAI with only the base URL and model IDs swapped. ￼

Tools / Structured output / Streaming

- Use OpenAI‑style tools and structured output when calling chat/completions or responses. SSE semantics match the respective OpenAI surfaces. If targeting messages, reuse your Anthropic tool_use mapping. ￼

---

### 2.3 Anthropic

Primary API: POST https://api.anthropic.com/v1/messages
Headers:

- x-api-key: <ANTHROPIC_API_KEY>
- anthropic-version: 2023-06-01 (required)
- Optional feature flags via anthropic-beta when using certain betas. ￼

Messages mapping

- Request body includes model, messages: [{role: "user" | "assistant", content: Content[]}], optional system, max_tokens, temperature, top_p, etc. Return value is a Message with content[]. ￼

Tool calling

- Define tools under top‑level tools: [{ name, description, input_schema }]. The model may emit content[] blocks with { type: "tool_use", id, name, input }. You then call your tool and respond in the next request by sending a user turn containing { type: "tool_result", tool_use_id, content } blocks (or text). This maps cleanly to your ToolCall + follow‑up. ￼

Structured output

- Anthropic supports schema‑constrained JSON via the Messages API. (Docs present this within “tools for JSON” and JSON‑mode/consistency guidance; you can implement a convenience that emits a “virtual tool” to coerce schema when native response_format isn’t available or is behind a beta header in your account.) ￼

Streaming

- SSE stream with typed events (e.g., message_start, content_block_start, content_block_delta, message_delta, message_stop). The official SDK exposes stream=True returning an iterator of events; your Node/Go adapters should translate events into your StreamChunk (delta/tool_call/message_end) and compute final usage. ￼

Vision inputs

- In messages[].content[], include image content via { type: "image", source: { type: "base64", media_type: "image/png", data: "..." } } (or URL depending on model support). (See Messages docs for content part types.) ￼

List models

- GET /v1/models with same required headers; response includes data[] with minimal fields (id, display_name, timestamps). Maintain a capability overlay similar to OpenAI. ￼

---

### 2.4 Google Gemini (Developer API / Generative Language API)

Base: https://generativelanguage.googleapis.com/v1beta
Auth: API key via x-goog-api-key in header (or ?key= query).

List models

- GET /v1beta/models → models[] includes supportedGenerationMethods, inputTokenLimit, outputTokenLimit, temperatures, etc. Filter where supportedGenerationMethods contains "generateContent". ￼

Generate

- REST: POST /v1beta/models/{model}:generateContent
  Body:
- contents: [ { role, parts: [{text|fileData|inlineData|functionCall|functionResponse}] } ]
- Optional systemInstruction (Content), tools (Function declarations), toolConfig, generationConfig, safetySettings. ￼

Tool calling

- Declare tools under tools.functionDeclarations. The model returns a functionCall in response parts; your app executes the function, then call the model again adding a functionResponse part to supply results. (You can cycle until no further tool calls.) ￼

Structured output

- Configure at generationConfig:
  responseMimeType: "application/json" and responseSchema (OpenAPI‑like subset). This yields guaranteed JSON matching the schema (and SDKs can expose .parsed). ￼

Streaming

- Use the streaming method variant (:streamGenerateContent) exposed in SDKs; REST supports SSE (alt=sse) in some examples. Your adapter should consume iterative candidates/chunks and emit StreamChunk('delta') text pieces and a final 'message_end' with usage if available. (See API ref + streaming mentions.) ￼

Vision inputs

- Provide images via parts: fileData (from uploaded file resource) or inlineData (base64 bytes). Multi‑part prompts (text + image) are standard. ￼

---

## 3. Type mappings (Node & Go)

### 3.1 Core types (shared mental model)

```ts
// Node (TS)
export type Provider = "openai" | "anthropic" | "google" | "xai";

export interface ContentText {
  type: "text";
  text: string;
}
export interface ContentImageURL {
  type: "image_url";
  url: string;
  mimeType?: string;
}
export interface ContentImageB64 {
  type: "image_b64";
  b64: string;
  mimeType: string;
}
export type Content = ContentText | ContentImageURL | ContentImageB64;

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: Content[]; // tool results go in content as text (and metadata carries tool_call_id)
  tool_call_id?: string; // when role='tool'
  name?: string; // optional, for tool messages
}

export interface JsonSchema {
  /* JSON Schema (or Gemini Schema subset) */
}

export interface Tool {
  name: string;
  description?: string;
  parameters: JsonSchema; // Anthropic calls this input_schema
}

export interface GenerateInput {
  provider: Provider;
  model: string;
  messages: Message[];
  tools?: Tool[];
  toolChoice?:
    | "auto"
    | "none"
    | { type: "function"; function: { name: string } };
  responseFormat?:
    | { type: "text" }
    | {
        type: "json_schema";
        schema: JsonSchema;
        name?: string;
        strict?: boolean;
      };
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  stream?: boolean;
  metadata?: Record<string, string>;
}

export interface ToolCall {
  id: string;
  name: string;
  argumentsJson: string;
}
export interface Usage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}
export interface GenerateOutput {
  text?: string;
  toolCalls?: ToolCall[];
  usage?: Usage;
  finishReason?: string;
  raw?: any;
}

export type StreamChunk =
  | { type: "delta"; textDelta: string }
  | { type: "tool_call"; call: ToolCall; delta?: string }
  | { type: "message_end"; usage?: Usage; finishReason?: string }
  | {
      type: "error";
      error: {
        kind: string;
        message: string;
        upstreamCode?: string;
        requestId?: string;
      };
    };
```

Go mirrors these as structs/interfaces with json tags. (You already defined these in your design; above just makes the mapping concrete.)

---

## 4. Adapter specifics

Each adapter implements ListModels, Generate, and StreamGenerate. Below are the field‑level translations you’ll actually code.

### 4.1 OpenAI adapter

ListModels(ctx)

- GET /v1/models → map { id, ... } into your ModelMetadata.
- Enrich with allowlist overlay for capabilities (text, vision, tools, structured_output, reasoning), context window, pricing; don’t trust /models to mark these. ￼

Generate(ctx, in)

- If in.responseFormat.type==='json_schema': prefer Responses API:
- Build input from in.messages:
- system: push an early input_text item with system text (or use instructions if you choose).
- user/assistant: for each content text → input_text; for images → input_image with b64/URL as supported.
- Add tools (OpenAI function tools) if present.
- Set response_format with json_schema { name, strict, schema }. ￼
- Else: use chat.completions:
- Map messages verbatim; text parts to content strings, images to {type:"image_url", image_url:{url|b64}}.
- Map tools to tools: [{type:'function', function:{...}}].
- Extract:
- text: from choices[0].message.content (Chat) or from Responses output.
- tool calls: from choices[0].message.tool_calls with function.name & function.arguments as a raw JSON string. ￼
- usage: usage.prompt_tokens / completion_tokens.

StreamGenerate(ctx, in)

- Responses API: parse event names:
- response.output_text.delta → StreamChunk{ type:'delta', textDelta: delta }
- response.tool_call.delta and creation → accumulate per‑ID and emit tool_call when call completes (or on first name+arguments).
- On response.completed emit message_end with usage if present. ￼
- Chat Completions:
- Handle choices[].delta.content as text deltas.
- Handle choices[].delta.tool_calls[].function.arguments as argument deltas, building up each call, and emit a tool_call when the finish_reason==='tool_calls'.

Errors

- Map SDK/HTTP errors to your ErrorKind (AuthenticationError→ProviderAuthError, RateLimitError→RateLimitError, etc.) and surface x-request-id. ￼

---

### 4.2 xAI adapter

ListModels(ctx)

- GET /v1/models → same shape as OpenAI; enrich from overlay. ￼

Generate / StreamGenerate

- If your caller selected OpenAI semantics: call /v1/chat/completions or /v1/responses with the same payload shapes you’d send to OpenAI; stream the same way.
- If your caller selected Anthropic semantics: call /v1/messages with tools as input_schema and handle tool_use blocks like Anthropic. (xAI documents all three surfaces.) ￼

---

### 4.3 Anthropic adapter

ListModels(ctx)

- GET /v1/models with x-api-key + anthropic-version. Output is minimal; annotate via overlay. ￼

Generate(ctx, in)

- Build:
- model, max_tokens, temperature/top_p (use one or the other), system (concat your system messages), and messages with only user/assistant roles.
- Convert your tools[] into Anthropic tools[] with input_schema JSON Schema. ￼
- If your request was “structured output”: two strategies 1. Native (if your account exposes JSON‑schema response formatting): pass the documented response‑format parameter or beta header for schema‑constrained JSON. 2. Portable fallback: install a “virtual tool” (e.g., produce_schema_conforming_output) whose input_schema is your desired output schema; instruct the model to always call it. This yields a tool call with guaranteed arguments matching the schema (a practical approach Anthropic recommends in docs). ￼
- Parse:
- Response content[] merges text and tool_use. Accumulate text to out.text. Convert each tool_use into your ToolCall.

StreamGenerate(ctx, in)

- SSE events from SDK or raw HTTP:
- content_block_delta with {type:"text_delta"} → delta.
- content_block_start with {type:"tool_use"} → start tracking a call; subsequent deltas fill input.
- message_stop / stream end → message_end with usage if provided. (SDK exposes for event in stream: print(event.type)). ￼

---

### 4.4 Google Gemini adapter

ListModels(ctx)

- GET /v1beta/models → record supportedGenerationMethods, token limits, and suggested temperatures. Mark models with generateContent as usable for chat; store limits into ModelMetadata. ￼

Generate(ctx, in)

- Endpoint: POST /v1beta/models/{model}:generateContent
- Build contents[]:
- Map your roles to Gemini Content:
- system → systemInstruction (preferred) or prepend a system Content.
- user/assistant → contents[] with parts (text → {text}, images → {inlineData | fileData}).
- Tools:
- Convert your tools to tools: [{ functionDeclarations: [{name, description, parameters(schema)}] }].
- When model returns a functionCall, execute the function, then call generateContent again adding a functionResponse part. Repeat until no calls return. ￼
- Structured output:
- Set generationConfig.responseMimeType = "application/json" and responseSchema mirroring your schema. SDKs can return both .text (JSON string) and .parsed (typed). ￼

StreamGenerate(ctx, in)

- Use streaming (:streamGenerateContent) or SDK stream API; emit your delta chunks on each incremental candidate text update, finalize with message_end. (Gemini API ref documents streaming and realtime surfaces.) ￼

---

## 5. HTTP handler shapes you will ship

### 5.1 GET /provider-models

Query: providers=openai,anthropic,google,xai (optional)
Response: ModelMetadata[] with at least:

```json
{
  "id": "gpt-4o",
  "displayName": "GPT-4o",
  "provider": "openai",
  "family": "gpt-4o",
  "capabilities": {
    "text": true,
    "vision": true,
    "tool_use": true,
    "structured_output": true,
    "reasoning": false
  },
  "contextWindow": 128000,
  "tokenPrices": { "input": 5.0, "output": 15.0 },
  "deprecated": false,
  "inPreview": false
}
```

Implementation: call each adapter’s ListModels, merge with overlay YAML, cache for 6h (with circuit‑breaker per provider).

### 5.2 POST /generate (non-stream)

Body: GenerateInput (your domain type)
Response: GenerateOutput with text, toolCalls[], usage, finishReason, optional raw (behind a flag).

### 5.3 GET /generate/stream (SSE)

Query or body: GenerateInput
SSE events: your StreamChunk JSON strings. Always end with a final message_end that carries usage.

---

## 6. Edge‑case differences & how we normalize them

Difference Normalization rule
System prompts: OpenAI Chat has system; Anthropic has system field; Gemini prefers systemInstruction. Keep your Message{role:'system'}. In adapters: OpenAI/Anthropic → set native system. Gemini → move to systemInstruction. ￼
Vision input encoding Normalize to ContentImageURL/ContentImageB64. Adapter maps to provider‑specific parts (image_url, input_image, fileData/inlineData). ￼
Tool schemas Your Tool.parameters is JSON Schema. OpenAI → tools[].function.parameters; Anthropic → tools[].input_schema; Gemini → tools.functionDeclarations[].parameters. ￼
Structured output OpenAI Responses → response_format.json_schema (strict); Gemini → generationConfig.responseSchema; Anthropic → use native JSON‑schema response if available for your account, else “virtual tool” fallback that yields guaranteed JSON in the tool args. ￼
Streaming event shapes Translate OpenAI Responses’ typed events and Anthropic’s event stream into your delta/tool_call/message_end. Gemini streams candidate chunks → delta. ￼
Model listing fidelity Use live lists + an overlay JSON for capabilities, context windows, pricing. Gemini’s supportedGenerationMethods is reliable; others require curation. ￼

---

## 7. Reference adapter skeletons

### 7.1 Node (TypeScript)

```ts
// OpenAI adapter (Responses + Chat)
export class OpenAIAdapter implements ProviderAdapter {
  constructor(private apiKey: string, private baseURL = 'https://api.openai.com/v1') {}

  async listModels(): Promise<ModelMetadata[]> {
    const r = await fetch(`${this.baseURL}/models`, { headers: { Authorization: `Bearer ${this.apiKey}` } });
    const data = await r.json();
    return normalizeOpenAIModels(data); // overlay capabilities here
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    if (input.responseFormat?.type === 'json_schema') {
      return this.responsesCreate(input);
    }
    return this.chatCompletionsCreate(input);
  }

  async _streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    if (input.responseFormat?.type === 'json_schema') {
      yield_ this.responsesStream(input); // map response._ events to StreamChunk
    } else {
      yield_ this.chatStream(input); // map chat.completion.chunk deltas
    }
  }

  // ... implement responsesCreate/responsesStream/chatCompletionsCreate/chatStream
}

// Anthropic adapter (Messages)
export class AnthropicAdapter implements ProviderAdapter {
  constructor(private apiKey: string, private version = '2023-06-01', private baseURL = 'https://api.anthropic.com/v1') {}

  async listModels(): Promise<ModelMetadata[]> {
    const r = await fetch(`${this.baseURL}/models`, {
      headers: { 'x-api-key': this.apiKey, 'anthropic-version': this.version }
    });
    const data = await r.json();
    return normalizeAnthropicModels(data);
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const body = toAnthropicMessagesPayload(input); // map tools->input_schema, messages, system, etc.
    const r = await fetch(`${this.baseURL}/messages`, {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'anthropic-version': this.version,
        'content-type': 'application/json'
      },
      body: JSON.stringify(body),
    });
    return fromAnthropicMessage(await r.json());
  }

  async _streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const body = { ...toAnthropicMessagesPayload(input), stream: true };
    const res = await fetch(`${this.baseURL}/messages`, {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'anthropic-version': this.version,
        'content-type': 'application/json'
      },
      body: JSON.stringify(body),
    });
    yield_ parseAnthropicSSE(res.body); // emit 'delta'/'tool_call'/'message_end'
  }
}

// Gemini adapter (GenerateContent)
export class GeminiAdapter implements ProviderAdapter {
  constructor(private apiKey: string, private baseURL = 'https://generativelanguage.googleapis.com/v1beta') {}

  async listModels(): Promise<ModelMetadata[]> {
    const r = await fetch(`${this.baseURL}/models?key=${this.apiKey}`);
    return normalizeGeminiModels(await r.json()); // filter supportedGenerationMethods includes generateContent
  }

  async generate(input: GenerateInput): Promise<GenerateOutput> {
    const { model } = input;
    const body = toGeminiGenerateContent(input); // contents, systemInstruction, tools.functionDeclarations, generationConfig
    const r = await fetch(`${this.baseURL}/models/${model}:generateContent?key=${this.apiKey}`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body)
    });
    return fromGeminiGenerateContent(await r.json());
  }

  async _streamGenerate(input: GenerateInput): AsyncIterable<StreamChunk> {
    const { model } = input;
    const body = toGeminiGenerateContent(input);
    const r = await fetch(`${this.baseURL}/models/${model}:streamGenerateContent?key=${this.apiKey}`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body)
    });
    yield_ parseGeminiSSE(r.body);
  }
}
```

### 7.2 Go (net/http)

Implement the same three methods on each provider, using http.Client with per‑request context. For SSE, read with bufio.Reader, split on \n\n, parse event: / data: lines, and forward mapped StreamChunk on a chan.

---

## 8. Error normalization (what to key off of)

- OpenAI: map SDK error types to your canonical kinds; include x-request-id. ￼
- Anthropic: HTTP status + JSON error object; include anthropic-version you sent in logs for repro. ￼
- Gemini: Google APIs standard errors (InvalidArgument 400, PermissionDenied 403, ResourceExhausted 429); record error.status/error.message. (See API reference for generateContent.) ￼
- xAI: treat like OpenAI for /chat/completions & /responses, like Anthropic for /messages. ￼

---

## 9. Testing fixtures you should add

1.  Golden I/O
    - Tools: a trivial get_weather(city, unit) schema.
    - Structured output: minimal schema (e.g., {"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}) used across providers.
2.  Streaming
    - Verify token deltas accumulate to the exact full text for each provider.
    - Verify tool call argument accumulation (OpenAI Chat) and event sequencing (Anthropic/Gemini).
3.  Model registry
    - Record fixtures of each provider’s model list. For Gemini, assert filtering by supportedGenerationMethods works. ￼

---

## 10. Security & telemetry hooks (ready‑to‑drop)

    - Config: provider keys via env + code; optional per‑tenant key resolver.
    - Redaction: hook to scrub PII from logs (never log full prompts by default).
    - Headers: capture and expose x-request-id/xai-request-id/Google request info when present.
    - OTel: span per call; attributes: provider/model/streaming/latency/tokens/error_kind.

---

## 11. Implementation gotchas & workarounds

    - OpenAI model list doesn’t label modalities or tool/JSON guarantees—keep a curated overlay and fall back to it when /models hiccups. ￼
    - Anthropic requires the anthropic-version header on every request. Put it in a shared http client wrapper. ￼
    - Gemini structured output uses responseMimeType + responseSchema under generationConfig (not a top‑level response_format). Keep a schema converter to/from JSON Schema where possible. ￼
    - xAI supports multiple compatibility modes; choose one per request based on the selected model/provider in your input; document precedence if the user supplies a Grok model with an Anthropic style payload. ￼

---

## 12. Minimal OpenAPI for your reference endpoints

```yaml
openapi: 3.1.0
info: { title: llmhub reference API, version: 1.0.0 }
paths:
  /provider-models:
    get:
      parameters:
        - in: query
          name: providers
          schema: { type: string, example: openai, anthropic, google, xai }
      responses:
        "200":
          description: OK
          content:
            application/json:
              schema:
                type: array
                items: { $ref: "#/components/schemas/ModelMetadata" }
  /generate:
    post:
      requestBody:
        required: true
        content:
          application/json:
            { schema: { $ref: "#/components/schemas/GenerateInput" } }
      responses:
        "200":
          description: OK
          content:
            application/json:
              { schema: { $ref: "#/components/schemas/GenerateOutput" } }
components:
  schemas:
    ModelMetadata:
      type: object
      properties:
        id: { type: string }
        displayName: { type: string }
        provider: { type: string, enum: [openai, anthropic, google, xai] }
        family: { type: string }
        capabilities:
          type: object
          properties:
            text: { type: boolean }
            vision: { type: boolean }
            tool_use: { type: boolean }
            structured_output: { type: boolean }
            reasoning: { type: boolean }
        contextWindow: { type: integer }
        tokenPrices:
          type: object
          properties:
            input: { type: number }
            output: { type: number }
        deprecated: { type: boolean }
        inPreview: { type: boolean }
    GenerateInput:
      type: object
      required: [provider, model, messages]
      properties:
        provider: { type: string }
        model: { type: string }
        messages: { type: array, items: {} }
        tools: { type: array, items: {} }
        toolChoice: {}
        responseFormat: {}
        temperature: { type: number }
        topP: { type: number }
        maxTokens: { type: integer }
        stream: { type: boolean }
    GenerateOutput:
      type: object
      properties:
        text: { type: string }
        toolCalls: { type: array, items: {} }
        usage:
          {
            type: object,
            properties:
              {
                inputTokens: { type: integer },
                outputTokens: { type: integer },
              },
          }
        finishReason: { type: string }
```

⸻

## 13. What to ship next in the monorepo

    - packages/node:
    - adapters/openai.ts: Responses + Chat with typed streaming. (Event parser for Responses.)
    - adapters/xai.ts: wraps OpenAI & Anthropic shapes based on endpoint selection.
    - adapters/anthropic.ts: Messages + streaming event parser.
    - adapters/gemini.ts: GenerateContent (+ stream) + function call loop helper.
    - httpHandlers.ts: /provider-models, /generate, /generate/stream (SSE).
    - overlays/ curated JSON for models/capabilities/pricing.
    - packages/go: mirror adapters + SSE parsers (bufio), plus llmhubhttp handlers.
    - Tests:
    - Golden tests per adapter (non‑stream + stream).
    - Contract tests: same GenerateInput across providers to ensure consistent GenerateOutput.

⸻

## Citations

- OpenAI: Responses API & streaming events (typed); SDK streaming examples; function/tool calling (Cookbook); Structured Outputs announcement; OpenAPI spec repo. ￼
- xAI (Grok): API reference (lists /v1/chat/completions, /v1/responses, /v1/messages, /v1/models); migration guide (OpenAI & Anthropic compatibility). ￼
- Anthropic: Messages API (headers, POST /v1/messages); Models list; tool use docs (input_schema); SDK streaming (stream=True). ￼
- Google Gemini: generateContent method & request fields; Structured output via responseMimeType + responseSchema; Models list with supportedGenerationMethods. ￼

⸻

## Final note

Provider APIs evolve; you’ve now got the current concrete shapes plus a portable mapping that will keep working even as providers add features. Where an API supports multiple surfaces (OpenAI/xAI), pick Responses for typed streaming and strict schemas; keep Chat for compatibility. For Anthropic and Gemini, you’re set up to do schema‑constrained outputs today with either native options (Gemini) or a robust fallback pattern (Anthropic via a “virtual tool”).
