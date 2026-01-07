from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ..catalog import load_catalog_models
from ..errors import ErrorKind, KitErrorPayload, AiKitError
from ..types import ImageGenerateInput, ImageGenerateOutput, ImageInput, ModelCapabilities, ModelMetadata
from ..clients.replicate_client import ReplicateClient


_NANO_BANANA_PREFIX = "google/nano-banana"


@dataclass
class ReplicateConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None


class ReplicateAdapter:
    def __init__(self, config: ReplicateConfig) -> None:
        self.config = config
        if config.api_key:
            os.environ["REPLICATE_API_TOKEN"] = config.api_key
        self._client = ReplicateClient()

    def list_models(self) -> List[ModelMetadata]:
        models = [model for model in load_catalog_models() if model.provider == "replicate"]
        if models:
            return models
        return [
            ModelMetadata(
                id=_NANO_BANANA_PREFIX,
                displayName=_NANO_BANANA_PREFIX,
                provider="replicate",
                family="image-edit",
                capabilities=ModelCapabilities(
                    text=True,
                    vision=True,
                    image=True,
                    tool_use=False,
                    structured_output=False,
                    reasoning=False,
                ),
            )
        ]

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        if not input.model.startswith(_NANO_BANANA_PREFIX):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Replicate adapter only supports {_NANO_BANANA_PREFIX}",
                    provider="replicate",
                )
            )
        image_inputs = _coerce_image_inputs(input.inputImages)
        parameters = dict(input.parameters or {})
        output_format = parameters.pop("output_format", "png")
        aspect_ratio = parameters.pop("aspect_ratio", None)
        result = self._client.nano_banana(
            model=input.model,
            prompt=input.prompt,
            image_input=image_inputs,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            parameters=parameters or None,
        )
        data = base64.b64encode(result).decode("ascii")
        mime = f"image/{output_format}" if output_format else "image/png"
        return ImageGenerateOutput(mime=mime, data=data)

    def generate(self, input):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Replicate text generation is not supported",
                provider="replicate",
            )
        )

    def generate_mesh(self, input):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Replicate mesh generation is not supported",
                provider="replicate",
            )
        )

    def generate_speech(self, input):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Replicate speech generation is not supported",
                provider="replicate",
            )
        )


def _coerce_image_inputs(raw: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    if not raw:
        return None
    inputs: List[Any] = []
    for entry in raw:
        image = _normalize_image_entry(entry)
        if image is None:
            continue
        inputs.append(image)
    return inputs or None


def _normalize_image_entry(entry: Any) -> Any:
    if isinstance(entry, ImageInput):
        return _image_input_to_payload(entry)
    if isinstance(entry, dict):
        return _image_input_to_payload(ImageInput(**entry))
    if isinstance(entry, str):
        if entry.startswith("data:"):
            return io.BytesIO(_decode_base64(entry))
        return entry
    return entry


def _image_input_to_payload(entry: ImageInput) -> Any:
    if entry.url:
        return entry.url
    if entry.base64:
        data = _decode_base64(entry.base64)
        return io.BytesIO(data)
    return None


def _decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        payload = value.split(",", 1)[1]
        return base64.b64decode(payload)
    return base64.b64decode(value)
