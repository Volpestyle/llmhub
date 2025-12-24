# llmhub

Cross-provider LLM hub with reusable libraries for Node.js (TypeScript), Go, and Python. The hub keeps a normalized model registry, adapts request/response shapes across OpenAI, Anthropic, xAI (Grok), and Google Gemini, and exposes HTTP helpers for `/provider-models`, `/generate`, and `/generate/stream` endpoints.

## Repository layout

- `packages/node` – TypeScript package `@volpestyle/llmhub-node` with adapters, hub factory, and Express-friendly HTTP handlers.
- `packages/go` – Go module `github.com/Volpestyle/llmhub` with mirrored domain types, adapters, and `net/http` handlers.
- `packages/python` – Python package `llmhub` with adapters, registry/router, and hub helper.
- `models/curated_models.json` – canonical curated capability overlay consumed by both libraries.
- `servers/openapi.yaml` – Minimal OpenAPI description for the reference endpoints.
- `docs/` – High-level design and implementation guides provided by product.

## Node.js usage

```ts
import express from "express";
import { createHub, httpHandlers, Provider } from "@volpestyle/llmhub-node";

const hub = createHub({
  providers: {
    [Provider.OpenAI]: { apiKey: process.env.OPENAI_API_KEY!, apiKeys: process.env.OPENAI_API_KEYS?.split(",") },
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
    OpenAI: &llmhub.OpenAIConfig{APIKey: os.Getenv("OPENAI_API_KEY"), APIKeys: strings.Split(os.Getenv("OPENAI_API_KEYS"), ",")},
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

## Local LLMs (Ollama, LM Studio, llama.cpp)

The package works with any OpenAI-compatible local LLM server by setting a custom `baseURL`:

```ts
const hub = createHub({
  providers: {
    [Provider.OpenAI]: {
      apiKey: "local",  // Local servers ignore this, but a value is required
      baseURL: "http://localhost:11434/v1",  // Ollama
      // baseURL: "http://localhost:1234/v1",  // LM Studio
      // baseURL: "http://localhost:8080/v1",  // llama.cpp server
    },
  },
});

// Use any locally-running model
const response = await hub.generate({
  provider: Provider.OpenAI,
  model: "llama3.2",
  messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
});
```

SSE streaming works identically—local servers emit events in OpenAI-compatible format, enabling progressive rendering in UI consumers.

## Python usage

```py
import os
from llmhub import Hub, HubConfig
from llmhub.providers import OpenAIConfig

hub = Hub(
    HubConfig(
        providers={
            "openai": OpenAIConfig(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                api_keys=os.environ.get("OPENAI_API_KEYS", "").split(","),
            ),
        }
    )
)

models = hub.list_models()
```

## Testing

- Node: from `packages/node`, run `npm install` once and then `npm test` (Vitest). Contract fixtures ensure each provider adapter returns identical normalized output and the HTTP handlers emit the expected JSON/SSE shapes.
- Go: from the repo root run `go test ./packages/go/...` to exercise the adapter contracts and the `net/http` handlers.

## Refreshing curated models

- Set the provider API keys you want to pull live metadata from (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GOOGLE_API_KEY`) — either export them or drop them in `packages/node/.env`/`.env.local` (the refresh script loads both via `dotenv`).
- From `packages/node`, run `npm run refresh:models` to call each configured provider (once, with `refresh: true`) and rewrite `models/curated_models.json`, the capability/pricing overlay both runtimes load at startup.
- At runtime the model registry always pulls from provider APIs (cached by default). The curated JSON is used as an overlay to enrich display names, capabilities, context windows, and pricing metadata.
- Append `?refresh=true` (or call `hub.listModels({ refresh: true })`) to bypass the registry cache when you explicitly need a live refresh.
- See `docs/design_doc.md` and `docs/implementation_guide.md` for more detail on how the curated overlay and refresh script fit into the registry strategy.
