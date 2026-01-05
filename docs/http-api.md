# HTTP API

The repo ships an OpenAPI reference in `servers/openapi.yaml`. Node and Go expose helpers for
wiring HTTP endpoints to a Kit instance, and Python ships an ASGI adapter for the same surface.
For SSE, the ASGI adapter is a good fit because it supports native streaming.

## Suggested endpoints
- `GET /provider-models` -> list models (query `providers=openai,anthropic,google,bedrock,xai,ollama,local` and `refresh=true` to bypass cache)
- `POST /generate` -> text generation
- `POST /image` -> image generation
- `POST /mesh` -> mesh generation
- `POST /speech` -> speech generation
- `POST /transcribe` -> audio transcription
- `POST /generate/stream` -> SSE stream

## Example: list models
```bash
curl "http://localhost:3000/provider-models?providers=openai,anthropic,bedrock,ollama"
```

Add `refresh=true` to bypass the registry cache.

Ollama uses the OpenAI-compatible API on `http://localhost:11434`; it does not require an API key.

## Example: generate
```bash
curl -X POST http://localhost:3000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Say hi"}]}
    ]
  }'
```

Optional fields like `toolChoice` and `responseFormat` are supported; see `servers/openapi.yaml` for the schema.

## Example: stream (SSE)
```bash
curl -N -X POST http://localhost:3000/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Stream me"}]}
    ]
  }'
```

SSE responses emit `event: chunk` payloads as JSON and finish with `event: done`.

## Example: transcribe
```bash
curl -X POST http://localhost:3000/transcribe \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini-transcribe",
    "audio": { "base64": "<base64 audio>" },
    "responseFormat": "verbose_json",
    "timestampGranularities": ["segment", "word"]
  }'
```

## Example: speech
```bash
curl -X POST http://localhost:3000/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini-tts",
    "text": "Hello from ai-kit.",
    "voice": "alloy",
    "responseFormat": "mp3"
  }'
```

For xAI realtime voice output, use `responseFormat` values `pcm`, `pcmu`, or `pcma`. You can pass `parameters.sampleRate` for PCM output, and the `model` field is ignored by the realtime API (use a placeholder like `grok-voice`). To force OpenAI-compat speech instead, set `metadata` to `{ "xai:speech-mode": "openai" }` or configure `speechMode` on the xAI provider.

## Python (ASGI adapter)
The Python SDK exposes a minimal ASGI app that you can mount in an existing server. It serves
the same endpoints listed above and supports SSE streaming.

```py
import os
from fastapi import FastAPI
from ai_kit import Kit, KitConfig, create_asgi_app
from ai_kit.providers import OpenAIConfig

kit = Kit(
    KitConfig(
        providers={
            "openai": OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY", "")),
        }
    )
)

app = FastAPI()
app.mount("/inference", create_asgi_app(kit))
```

If you are serving the ASGI app directly and want a prefix, pass `base_path="/inference"`.
