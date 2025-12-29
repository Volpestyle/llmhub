# inference-kit

Provider-agnostic inference tooling for Node.js, Go, and Python. The repo standardizes model
listing, routing, generation, streaming (SSE), and cost estimation across OpenAI, Anthropic,
Google Gemini, and xAI. It also ships shared curated model metadata and a reference OpenAPI
spec for HTTP servers.

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
pnpm --filter @volpestyle/inference-kit-node build
```
```ts
import { createHub, Provider } from "@volpestyle/inference-kit-node";

const hub = createHub({
  providers: {
    [Provider.OpenAI]: { apiKey: process.env.OPENAI_API_KEY ?? "" },
  },
});

const output = await hub.generate({
  provider: Provider.OpenAI,
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: [{ type: "text", text: "Hello" }] }],
});

console.log(output.text);
```

### Go
```bash
go get github.com/Volpestyle/inference-kit/packages/go
```
```go
package main

import (
  "context"
  "fmt"
  "os"

  inferencekit "github.com/Volpestyle/inference-kit/packages/go"
)

func main() {
  hub, err := inferencekit.New(inferencekit.Config{
    OpenAI: &inferencekit.OpenAIConfig{APIKey: os.Getenv("OPENAI_API_KEY")},
  })
  if err != nil {
    panic(err)
  }

  out, err := hub.Generate(context.Background(), inferencekit.GenerateInput{
    Provider: inferencekit.ProviderOpenAI,
    Model:    "gpt-4o-mini",
    Messages: []inferencekit.Message{{
      Role: "user",
      Content: []inferencekit.ContentPart{{
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
from inference_kit import Hub, HubConfig, GenerateInput, Message, ContentPart
from inference_kit.providers import OpenAIConfig

hub = Hub(
    HubConfig(
        providers={
            "openai": OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY", ""))
        }
    )
)

out = hub.generate(
    GenerateInput(
        provider="openai",
        model="gpt-4o-mini",
        messages=[Message(role="user", content=[ContentPart(type="text", text="Hello")])],
    )
)

print(out.text)
```

## Examples
### Auto-select the cheapest compatible model
```ts
import { ModelRouter, Provider } from "@volpestyle/inference-kit-node";

const models = await hub.listModelRecords();
const router = new ModelRouter();
const resolved = router.resolve(models, {
  constraints: { requireTools: true, maxCostUsd: 2.0 },
  preferredModels: ["openai:gpt-4o-mini"],
});

const output = await hub.generate({
  provider: resolved.primary.provider,
  model: resolved.primary.providerModelId,
  messages: [{ role: "user", content: [{ type: "text", text: "Summarize this" }] }],
});
```

### Stream SSE for progressive UI rendering (Node)
```ts
import express from "express";
import { createHub, httpHandlers, Provider } from "@volpestyle/inference-kit-node";

const app = express();
app.use(express.json());

const hub = createHub({
  providers: {
    [Provider.OpenAI]: { apiKey: process.env.OPENAI_API_KEY ?? "" },
  },
});

const handlers = httpHandlers(hub);
app.post("/generate", handlers.generate());
app.post("/generate/stream", handlers.generateSSE());
app.get("/provider-models", handlers.models());

app.listen(3000);
```

More details live in `docs/README.md`.
