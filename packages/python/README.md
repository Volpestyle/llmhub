# ai-kit (Python)

Python implementation of ai-kit provider adapters, model registry, router, and
optional local pipelines for basic vision tasks.

## Quickstart
```bash
python -m pip install -e packages/python
```
```py
import os
from ai_kit import Hub, HubConfig, GenerateInput, Message, ContentPart
from ai_kit.providers import OpenAIConfig

kit = Hub(
    HubConfig(
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

### Local pipelines (optional)
Local pipelines rely on `torch`, `transformers`, `Pillow`, and `numpy`.
```py
from pathlib import Path
from ai_kit.local import get_pipeline, load_rgb

pipe = get_pipeline("image-segmentation", "rmbg-1.4")
image = load_rgb(Path("input.jpg"))
result = pipe(image)
```
