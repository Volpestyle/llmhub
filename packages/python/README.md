# ai-kit (Python)

Core ai-kit types, catalog, and pricing helpers. Inference adapters and local
pipelines live in separate optional packages.

## Quickstart
Core only:
```bash
python -m pip install -e packages/python
```

Inference adapters:
```bash
python -m pip install -e packages/python-inference
```

Local pipelines:
```bash
python -m pip install -e packages/python-local
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

## Amazon Bedrock
```py
from ai_kit import Kit, KitConfig
from ai_kit.providers import BedrockConfig

kit = Kit(
    KitConfig(
        providers={
            "bedrock": BedrockConfig(
                region=os.environ.get("AWS_REGION", ""),
                access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                session_token=os.environ.get("AWS_SESSION_TOKEN"),
            )
        }
    )
)
```

## Examples
### Stream tokens
```py
from ai_kit import GenerateInput, Message, ContentPart

stream = kit.stream_generate(
    GenerateInput(
        provider="openai",
        model="gpt-4o-mini",
        messages=[Message(role="user", content=[ContentPart(type="text", text="Stream")])],
        stream=True,
    )
)

for chunk in stream:
    if chunk.type == "delta" and chunk.textDelta:
        print(chunk.textDelta, end="")
```

### Grok voice agent (xAI)
Requires `packages/python-inference` for the xAI provider.
```py
import os
from ai_kit import Kit, KitConfig, VoiceAgentInput
from ai_kit.providers import XAIConfig

kit = Kit(
    KitConfig(
        providers={"xai": XAIConfig(api_key=os.environ.get("XAI_API_KEY", ""))}
    )
)

out = kit.generate_voice_agent(
    VoiceAgentInput(
        provider="xai",
        model="grok-voice",  # placeholder for realtime API
        instructions="You are a warm, romantic guide.",
        voice="Ara",
        userText="Plan a date night in Paris.",
        responseModalities=["text", "audio"],
    )
)

print(out.transcript, out.audio.mime if out.audio else None)
```

### Local pipelines (optional)
Local pipelines rely on `torch`, `transformers`, `Pillow`, and `numpy`.
Install `packages/python-local` first.
`ai-kit-local` also ships a `LocalWhisperAdapter` for offline transcription.
```py
from pathlib import Path
from ai_kit.local import get_pipeline, load_rgb

pipe = get_pipeline("image-segmentation", "rmbg-1.4")
image = load_rgb(Path("input.jpg"))
result = pipe(image)
```
