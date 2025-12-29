# ai-kit

Provider-agnostic inference tooling for Node.js, Go, and Python. The repo standardizes model
listing, routing, generation, streaming (SSE), and cost estimation across OpenAI, Anthropic,
Google Gemini, xAI, and local Ollama endpoints. It also ships shared curated model metadata
and a reference OpenAPI spec for HTTP servers.

## Packages
- `packages/node`: Node.js SDK and HTTP handlers
- `packages/go`: Go SDK and HTTP handlers
- `packages/python`: Python SDK + local pipelines for basic vision tasks
- `models`: shared curated model metadata and optional catalog models
- `servers/openapi.yaml`: reference HTTP API
- `docs`: architecture overview and HTTP notes

## Quickstart
### Node.js
```bash
pnpm install
pnpm --filter @volpestyle/ai-kit-node build
```
```ts
import { createKit, Provider } from "@volpestyle/ai-kit-node";

const kit = createKit({
  providers: {
    [Provider.OpenAI]: { apiKey: process.env.OPENAI_API_KEY ?? "" },
  },
});

const output = await kit.generate({
  provider: Provider.OpenAI,
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
});

console.log(output.text);
```

### Go
```bash
go get github.com/Volpestyle/ai-kit/packages/go
```
```go
package main

import (
  "context"
  "fmt"
  "os"

  aikit "github.com/Volpestyle/ai-kit/packages/go"
)

func main() {
  kit, err := aikit.New(aikit.Config{
    OpenAI: &aikit.OpenAIConfig{APIKey: os.Getenv("OPENAI_API_KEY")},
  })
  if err != nil {
    panic(err)
  }

  out, err := kit.Generate(context.Background(), aikit.GenerateInput{
    Provider: aikit.ProviderOpenAI,
    Model:    "gpt-4o-mini",
    Messages: []aikit.Message{{
      Role: "user",
      Content: []aikit.ContentPart{{
        Type: "text",
        Text: "Hello",
      }},
    }},
  })
  if err != nil {
    panic(err)
  }

  fmt.Println(out.Text)
}
```

### Python
```bash
python -m pip install -e packages/python
```
```py
import os
from ai_kit import Kit, KitConfig, GenerateInput, Message, ContentPart
from ai_kit.providers import OpenAIConfig

kit = Kit(
    KitConfig(
        providers={
            "openai": OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY", ""))
        }
    )
)

out = kit.generate(
    GenerateInput(
        provider="openai",
        model="gpt-4o-mini",
        messages=[Message(role="user", content=[ContentPart(type="text", text="Hello")])],
    )
)

print(out.text)
```

## Ollama (local)
Ollama speaks the OpenAI-compatible API on `http://localhost:11434`. Configure the
`ollama` provider without an API key.

### Node.js
```ts
import { createKit, Provider } from "@volpestyle/ai-kit-node";

const kit = createKit({
  providers: {
    [Provider.Ollama]: { baseURL: "http://localhost:11434" },
  },
});
```

### Go
```go
kit, err := aikit.New(aikit.Config{
  Ollama: &aikit.OllamaConfig{BaseURL: "http://localhost:11434"},
})
```

### Python
```py
from ai_kit import Kit, KitConfig
from ai_kit.providers import OllamaConfig

kit = Kit(KitConfig(providers={"ollama": OllamaConfig(base_url="http://localhost:11434")}))
```

## Examples
### Auto-select the cheapest compatible model
```ts
import { ModelRouter, Provider } from "@volpestyle/ai-kit-node";

const models = await kit.listModelRecords();
const router = new ModelRouter();
const resolved = router.resolve(models, {
  constraints: { requireTools: true, maxCostUsd: 2.0 },
  preferredModels: ["openai:gpt-4o-mini"],
});

const output = await kit.generate({
  provider: resolved.primary.provider,
  model: resolved.primary.providerModelId,
  messages: [{ role: "user", content: [{ type: "text", text: "Summarize this" }] }],
});
```

### Stream SSE for progressive UI rendering (Node)
```ts
import express from "express";
import { createKit, httpHandlers, Provider } from "@volpestyle/ai-kit-node";

const app = express();
app.use(express.json());

const kit = createKit({
  providers: {
    [Provider.OpenAI]: { apiKey: process.env.OPENAI_API_KEY ?? "" },
  },
});

const handlers = httpHandlers(kit);
app.post("/generate", handlers.generate());
app.post("/generate/stream", handlers.generateSSE());
app.get("/provider-models", handlers.models());

app.listen(3000);
```

More details live in `docs/README.md`. Testing fixtures are documented in `docs/testing.md`.
