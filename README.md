# llmhub

Cross-provider LLM hub with reusable libraries for Node.js (TypeScript) and Go. The hub keeps a normalized model registry, adapts request/response shapes across OpenAI, Anthropic, xAI (Grok), and Google Gemini, and exposes HTTP helpers for `/provider-models`, `/generate`, and `/generate/stream` endpoints.

## Repository layout

- `packages/node` – TypeScript package `@volpestyle/llmhub-node` with adapters, hub factory, and Express-friendly HTTP handlers.
- `packages/go` – Go module `github.com/Volpestyle/llmhub` with mirrored domain types, adapters, and `net/http` handlers.
- `packages/go/curated_models.json` – curated capability overlay shared by both libraries.
- `servers/openapi.yaml` – Minimal OpenAPI description for the reference endpoints.
- `docs/` – High-level design and implementation guides provided by product.

## Node.js usage

```ts
import express from "express";
import { createHub, httpHandlers, Provider } from "@volpestyle/llmhub-node";

const hub = createHub({
  providers: {
    [Provider.OpenAI]: { apiKey: process.env.OPENAI_API_KEY! },
    [Provider.Anthropic]: { apiKey: process.env.ANTHROPIC_API_KEY! },
    [Provider.XAI]: { apiKey: process.env.XAI_API_KEY! },
    [Provider.Google]: { apiKey: process.env.GOOGLE_API_KEY! },
  },
});

const app = express();
app.use(express.json());
const handlers = httpHandlers(hub);
app.get("/provider-models", handlers.models());
app.post("/generate", handlers.generate());
app.post("/generate/stream", handlers.generateSSE());
```

Run `npm install` then `npm run build` inside `packages/node` to emit CJS/ESM bundles.

## Go usage

```go
import (
    "log"
    "github.com/Volpestyle/llmhub"
)

hub, err := llmhub.New(llmhub.Config{
    OpenAI: &llmhub.OpenAIConfig{APIKey: os.Getenv("OPENAI_API_KEY")},
    Anthropic: &llmhub.AnthropicConfig{APIKey: os.Getenv("ANTHROPIC_API_KEY")},
    XAI: &llmhub.XAIConfig{APIKey: os.Getenv("XAI_API_KEY")},
    Google: &llmhub.GoogleConfig{APIKey: os.Getenv("GOOGLE_API_KEY")},
})
if err != nil { log.Fatal(err) }

http.HandleFunc("/provider-models", llmhub.ModelsHandler(hub, nil))
http.HandleFunc("/generate", llmhub.GenerateHandler(hub))
http.HandleFunc("/generate/stream", llmhub.GenerateSSEHandler(hub))
```

Build with `go build ./...` from the repo root.

> The Go OpenAI adapter automatically switches to the Responses API whenever you request a JSON-schema `responseFormat`, ensuring strict structured outputs that mirror the TypeScript implementation.

## Streaming contracts

- Node adapters expose `hub.streamGenerate(input)` returning an `AsyncIterable<StreamChunk>` normalized across providers.
- Go adapters return `<-chan StreamChunk` with the same shape; HTTP helpers emit Server-Sent Events with `event: chunk` entries.

## Testing

- Node: from `packages/node`, run `npm install` once and then `npm test` (Vitest). Contract fixtures ensure each provider adapter returns identical normalized output and the HTTP handlers emit the expected JSON/SSE shapes.
- Go: from the repo root run `go test ./packages/go/...` to exercise the adapter contracts and the `net/http` handlers.

## Next steps

- Flesh out structured-output enforcement for Anthropic’s beta schema surface.
- Harden HTTP handlers with authentication hooks and request validation.
- Add contract tests that replay the same `GenerateInput` across adapters for regression coverage.
