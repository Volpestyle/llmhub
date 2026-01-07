from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ..catalog import load_catalog_models
from ..errors import ErrorKind, KitErrorPayload, AiKitError
from ..types import (
    ImageGenerateInput,
    ImageGenerateOutput,
    ImageInput,
    LipsyncGenerateInput,
    LipsyncGenerateOutput,
    ModelCapabilities,
    ModelMetadata,
    VideoGenerateInput,
    VideoGenerateOutput,
)
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

    def generate_video(self, input: VideoGenerateInput) -> VideoGenerateOutput:
        parameters = dict(input.parameters or {})
        payload: dict[str, Any] = {"prompt": input.prompt}
        start_image = _resolve_start_image(input.startImage, input.inputImages)
        if start_image is not None:
            payload["start_image"] = start_image
        if input.duration is not None:
            payload["duration"] = input.duration
        if input.aspectRatio:
            payload["aspect_ratio"] = input.aspectRatio
        if input.negativePrompt:
            payload["negative_prompt"] = input.negativePrompt
        if input.generateAudio is not None:
            payload["generate_audio"] = input.generateAudio
        if parameters:
            payload.update(parameters)

        output = self._client.run(input.model, inputs=payload)
        data = _coerce_video_bytes(output)
        return VideoGenerateOutput(
            mime="video/mp4",
            data=base64.b64encode(data).decode("ascii"),
            raw=output,
        )

    def generate_lipsync(self, input: LipsyncGenerateInput) -> LipsyncGenerateOutput:
        payload: dict[str, Any] = {}
        video_payload = _coerce_media_input(input.videoUrl, input.videoBase64)
        if video_payload is not None:
            payload["video_url"] = video_payload
            payload["video"] = video_payload
        audio_payload = _coerce_media_input(input.audioUrl, input.audioBase64)
        if audio_payload is not None:
            payload["audio_file"] = audio_payload
            payload["audio"] = audio_payload
        if input.text:
            payload["text"] = input.text
        if input.voiceId:
            payload["voice_id"] = input.voiceId
        if input.voiceSpeed is not None:
            payload["voice_speed"] = input.voiceSpeed
        if input.parameters:
            payload.update(input.parameters)

        if not payload.get("video_url") and not payload.get("video"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message="Replicate lipsync requires a video input",
                    provider="replicate",
                )
            )
        if not payload.get("audio_file") and not payload.get("audio") and not input.text:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message="Replicate lipsync requires audio or text input",
                    provider="replicate",
                )
            )

        output = self._client.run(input.model, inputs=payload)
        data = _coerce_video_bytes(output)
        return LipsyncGenerateOutput(
            mime="video/mp4",
            data=base64.b64encode(data).decode("ascii"),
            raw=output,
        )

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


def _resolve_start_image(start_image: Optional[str], input_images: Optional[Sequence[Any]]) -> Any:
    if start_image:
        return _normalize_image_entry(start_image)
    if not input_images:
        return None
    for entry in input_images:
        normalized = _normalize_image_entry(entry)
        if normalized is not None:
            return normalized
    return None


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


def _coerce_media_input(url: Optional[str], data: Optional[str]) -> Any:
    if url:
        return url
    if data:
        return io.BytesIO(_decode_base64(data))
    return None


def _coerce_video_bytes(output: Any) -> bytes:
    if output is None:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNKNOWN,
                message="Replicate video output missing",
                provider="replicate",
            )
        )
    if isinstance(output, bytes):
        return output
    if isinstance(output, str):
        if output.startswith("http"):
            return _download_url(output)
        return base64.b64decode(output)
    if isinstance(output, dict):
        url = output.get("url") or output.get("video")
        if isinstance(url, str):
            return _download_url(url)
    if isinstance(output, (list, tuple)) and output:
        return _coerce_video_bytes(output[0])
    raise AiKitError(
        KitErrorPayload(
            kind=ErrorKind.UNKNOWN,
            message=f"Unsupported Replicate video output type: {type(output)}",
            provider="replicate",
        )
    )


def _download_url(url: str) -> bytes:
    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content
