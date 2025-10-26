### Design Doc: Cross-Provider LLM Model Registry and Unified Inference Adapter (Go + Node)

#### Problem

- **Fragmented APIs**: OpenAI, Anthropic, xAI (Grok), and Google Gemini expose different request/response shapes, streaming semantics, tool/function calling formats, and model listing endpoints.
- **Duplicated glue**: Every backend has to re-implement model discovery, request adaptation, error mapping, and streaming for each provider.
- **Reuse need**: You want a drop-in way to add `/provider-models` and a unified `/generate` to any backend, and to import this capability in Go and Node projects.

#### Goals

- **Unified library (Go + Node)** that:
  - Lists up-to-date models across providers with a consistent `ModelMetadata`.
  - Adapts generate calls (sync + streaming) to provider-specific APIs via a common `GenerateInput`/`GenerateOutput`.
  - Supports tools/function-calling and structured output (JSON mode / schema) where available.
  - Exposes ready-made HTTP handlers to add `/provider-models` and `/generate` to any service in minutes.
- **Extensible provider plugin model** to add new providers easily.
- **Production-grade**: caching, error normalization, rate-limit resilience, telemetry, strong typing.

#### Non-Goals (v1)

- Full Vertex AI and Azure OpenAI enterprise variants (plan for v1.1).
- Multi-turn conversation memory store (out of scope; pass messages in).
- Embeddings/images/audio APIs (v1 focuses on text + image-in for chat where supported).

---

### High-Level Architecture

- **Core (language-specific) library**

  - Provider-agnostic domain types: `Provider`, `ModelMetadata`, `GenerateInput`, `Message`, `Tool`, `ToolCall`, `StreamChunk`, `Usage`, `ErrorKind`.
  - Provider adapters implementing a common interface.
  - Model registry with pluggable caching + background refresh.
  - HTTP handler package exposing reference endpoints.
  - Typed error mapping and observability hooks.

- **Reference microservice (optional)**

  - Thin server that mounts the handlers using the core library.
  - Ships with OpenAPI spec and ready clients.

- **Provider adapters (v1)**
  - OpenAI, Anthropic, xAI (Grok), Google Gemini.
  - Capabilities matrix for each model: text, vision, streaming, tool_use, structured_output, reasoning, context_window, pricing.

---

### Repository Strategy

One mono-repo, multiple packages

- `llmhub/` (root repo)
  - `packages/node/` → `@volpestyle/llmhub-node` (TS)
  - `packages/go/` → `github.com/Volpestyle/llmhub` (Go)
  - `servers/node/` and `servers/go/` reference apps
- Pros: Shared docs/tests, consistent versioning. Cons: Cross-language tooling in one repo.

---

### Public HTTP Endpoints (reference)

- `GET /provider-models`
  - Query: `providers=openai,anthropic,google,xai` (optional)
  - Response: array of `ModelMetadata`
- `POST /generate`
  - Body: `GenerateInput` (provider, model, messages, tools, response_format, stream?)
  - Response:
    - Non-stream: `GenerateOutput` with text, tool calls, usage, finish_reason
    - Stream: SSE or chunked JSON of `StreamChunk` until done

Ship an OpenAPI spec so your frontends/backend clients can generate code.

#### Why?

- It’s the HTTP contract for both:

  1. The optional microservice (if you deploy one), and
  2. Any app that embeds the library and mounts the provided handlers.

- The term “public” here means “published API contract” (documented and stable), not “internet-exposed.”
- If you run a microservice, it should expose exactly those routes.
- If you embed the lib in an existing backend, you can mount the same routes in that service.
- The OpenAPI spec is for client codegen either way (your apps call your service; browsers shouldn’t call it directly).
- You can deploy the endpoints privately (VPC/service mesh) or behind an API gateway with auth; exposure is a deployment choice.

---

### Core Domain Types (Node + Go aligned)

- Model metadata:
  - `id`, `displayName`, `provider`, `family` (e.g., gpt-4o, claude-3-7, gemini-1.5), `capabilities` (text, vision, tool_use, structured_output, reasoning), `contextWindow`, `tokenPrices` (input/output per 1K), `deprecated`, `inPreview`.
- Messages:
  - `{ role: 'system' | 'user' | 'assistant' | 'tool', content: Content[] }`
  - `Content` supports text and images (base64 or URL) with strong typing.
- Tools:
  - JSON-serializable schema with `name`, `description`, `parameters` (JSON Schema).
- Generate:
  - Input: `provider`, `model`, `messages`, `tools?`, `toolChoice?`, `responseFormat?` (json/object schema or "text"), `temperature`, `topP`, `maxTokens`, `metadata?`, `stream?`.
  - Output: `text`, `toolCalls[]`, `usage`, `finishReason`, `raw?` (optional for debugging).

---

### Provider Adapters (API translation)

Each adapter implements:

- `ListModels(ctx) ([]ModelMetadata, error)`
- `Generate(ctx, GenerateInput) (GenerateOutput, error)`
- `StreamGenerate(ctx, GenerateInput) (<-chan StreamChunk, error)`

Key diffs handled:

- **OpenAI**: `chat.completions`, `response_format` (json/object), `tool_calls`, vision in messages, streaming deltas.
- **Anthropic**: `messages` API, tool use segments, JSON output control, streaming tokens vs events, `anthropic-version` header.
- **xAI (Grok)**: OpenAI-compatible-ish `chat/completions` (normalize any deviations).
- **Google Gemini**: `models:list` and `:generateContent`/`:streamGenerateContent`, role names, tool invocation format, JSON schema via `response_mime_type` + `response_schema`.

Adapters map our domain types to provider payloads and back, including:

- Tools/function-calling translation (OpenAI tool_calls ↔ Anthropic tool_use ↔ Gemini functionCall/result).
- Structured output (native where supported; fallback to robust JSON repair when not guaranteed).
- Streaming unification to `StreamChunk { type: 'delta' | 'tool_call' | 'message_end' | 'error', ... }`.

---

### Model Registry and Caching

- Fetch model lists from each provider:
  - OpenAI: `GET /v1/models` then filter chat-capable models; overlay a curated allowlist to avoid noisy fine-tuned/beta models.
  - Anthropic: `GET /v1/models`.
  - xAI: `GET /v1/models`.
  - Google: `GET /v1beta/models` filter `generation_methods` includes `generateContent`.
- Cache: in-memory + pluggable external (Redis) with TTL (default 6h), background refresh, per-provider circuit breaker.
- Fallback: ship a minimal curated static list when upstream fails.
- Capabilities enrichment: annotate from known feature matrix and provider docs.

---

### Error Normalization

Map upstream errors into typed categories:

- `ProviderAuthError`, `RateLimitError`, `QuotaExceededError`, `ModelNotFoundError`, `ValidationError`, `UpstreamServiceError`, `TimeoutError`, `TransientNetworkError`.
  Include `provider`, `upstreamStatus`, `upstreamCode`, `requestId`, `retryAfter`.

---

### Streaming

- Node: `AsyncIterable<StreamChunk>` and an Express/Fastify SSE handler out of the box.
- Go: `chan StreamChunk` + helper for SSE or HTTP chunked transfer.
- Always send final `message_end` with accumulated usage.

---

### Security and Config

- Per-provider API keys via env and code config; optional per-tenant key resolver hook.
- Optional baseURL overrides (Azure OpenAI, proxies, regional endpoints).
- Redaction hooks for logs; never log content by default.
- OpenTelemetry spans, metrics (latency, tokens, errors, rate limits).

---

### Testing

- Unit tests for adapters with golden fixtures.
- Live integration tests gated by env flags/keys.
- Contract tests ensuring Node and Go parity on the same inputs.
- Replay harness for streaming.

---

### Packaging and Versioning

- Node: TypeScript, ESM + CJS builds, `@volpestyle/llmhub-node`.
- Go: `github.com/Volpestyle/llmhub`, semantic import versioning.
- OpenAPI: published under `servers/openapi.yaml`, CI to generate SDKs.

---

### Example Usage

Node (Express):

```ts
import express from "express";
import { createHub, httpHandlers, Provider } from "@volpestyle/llmhub-node";

const app = express();
app.use(express.json());

const hub = createHub({
  providers: {
    [Provider.OpenAI]: { apiKey: process.env.OPENAI_API_KEY! },
    [Provider.Anthropic]: { apiKey: process.env.ANTHROPIC_API_KEY! },
    [Provider.XAI]: { apiKey: process.env.XAI_API_KEY! },
    [Provider.Google]: { apiKey: process.env.GOOGLE_API_KEY! },
  },
});

app.get(
  "/provider-models",
  httpHandlers.models(hub, { ttlMs: 6 * 60 * 60 * 1000 })
);
app.post("/generate", httpHandlers.generate(hub)); // non-stream
app.get("/generate/stream", httpHandlers.generateSSE(hub)); // streaming

app.listen(3000);
```

Go (net/http):

```go
hub := llmhub.New(llmhub.Config{
  Providers: map[llmhub.Provider]llmhub.ProviderConfig{
    llmhub.OpenAI:   {APIKey: os.Getenv("OPENAI_API_KEY")},
    llmhub.Anthropic:{APIKey: os.Getenv("ANTHROPIC_API_KEY")},
    llmhub.XAI:      {APIKey: os.Getenv("XAI_API_KEY")},
    llmhub.Google:   {APIKey: os.Getenv("GOOGLE_API_KEY")},
  },
})

http.Handle("/provider-models", llmhubhttp.ModelsHandler(hub, llmhubhttp.ModelsOpts{TTL: 6 * time.Hour}))
http.Handle("/generate",        llmhubhttp.GenerateHandler(hub))        // POST for non-stream
http.Handle("/generate/stream", llmhubhttp.GenerateSSEHandler(hub))     // GET for SSE

log.Fatal(http.ListenAndServe(":3000", nil))
```

Client call (Node) with streaming:

```ts
const stream = hub.streamGenerate({
  provider: "anthropic",
  model: "claude-3-7-sonnet",
  messages: [
    { role: "user", content: [{ type: "text", text: "Give me 3 ideas" }] },
  ],
});
for await (const chunk of stream) {
  if (chunk.type === "delta") process.stdout.write(chunk.textDelta);
}
```

---

### Alternatives Considered

- **Library-only (no HTTP)**: simplest; but every service must write handlers repeatedly.
- **Service-only**: easiest to adopt; but inline calls (without network hop) not possible.
- **Adopt third-party routers (e.g., OpenRouter, LangChain, LiteLLM)**: faster start but less control, inconsistent type safety across Go/Node, and often opinionated.  
  Recommendation: own lightweight, typed adapters; small surface area; explicit capabilities.

---

### Roadmap

- v1:
  - Node + Go core libraries
  - OpenAI, Anthropic, xAI, Google adapters
  - Model registry with cache and fallback
  - `/provider-models`, `/generate`, `/generate/stream` handlers
  - OpenAPI spec and examples
- v1.1:
  - Vertex AI and Azure OpenAI variants
  - Redis cache adapter; rate-limiters and circuit breakers
  - Structured output with JSON schema across providers
- v1.2:
  - Embeddings/images/audio
  - Per-tenant key resolver and quotas
  - CLI and docs site

---

### Naming and Licensing

- Repo: `llmhub` (mono-repo)
- Packages: `@volpestyle/llmhub-node`, `github.com/Volpestyle/llmhub`
- License: MIT

---

### How You’ll Use It in Improview (example)

- Add `llmhub` dependency (Go), mount `/provider-models` and `/generate` in your existing backend.
- Replace provider-specific code in your `/generate` endpoint with a single `hub.generate(...)`.
- Frontend/API client remains the same; you continue to send `{ provider, model, ... }`.

---

### Risks and Mitigations

- **API churn by providers**: version adapters, pin API dates (Anthropic), add integration tests.
- **Model listing instability**: curate allowlist + cache; expose feature flags to filter models.
- **Streaming differences**: normalize to `StreamChunk` and provide robust SSE helpers.

---
