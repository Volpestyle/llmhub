# ai-kit (Go)

Provider-agnostic model registry and inference adapter for Go. Includes a Kit, model router,
SSE streaming, and HTTP handlers for a small REST surface.

## Quickstart
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

## Examples
### HTTP handlers with SSE
```go
import (
  "net/http"

  aikit "github.com/Volpestyle/ai-kit/packages/go"
)

kit, _ := aikit.New(aikit.Config{
  OpenAI: &aikit.OpenAIConfig{APIKey: os.Getenv("OPENAI_API_KEY")},
})

http.HandleFunc("/provider-models", aikit.ModelsHandler(kit, nil))
http.HandleFunc("/generate", aikit.GenerateHandler(kit))
http.HandleFunc("/generate/stream", aikit.GenerateSSEHandler(kit))
http.ListenAndServe(":3000", nil)
```

### Route to a preferred model
```go
router := &aikit.ModelRouter{}
records, _ := kit.ListModelRecords(context.Background(), nil)
resolved, _ := router.Resolve(records, aikit.ModelResolutionRequest{
  Constraints: aikit.ModelConstraints{RequireTools: true},
  PreferredModels: []string{"openai:gpt-4o-mini"},
})

_ = resolved.Primary
```
