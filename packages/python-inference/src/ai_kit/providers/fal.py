from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..catalog import load_catalog_models
from ..errors import ErrorKind, KitErrorPayload, AiKitError
from ..media import data_url_media_type, guess_extension, write_temp_file
from ..types import (
    ImageInput,
    LipsyncGenerateInput,
    LipsyncGenerateOutput,
    ModelCapabilities,
    ModelMetadata,
    VideoGenerateInput,
    VideoGenerateOutput,
)
from ..clients.fal_client import FalClient


@dataclass
class FalConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None
    timeout_s: Optional[float] = None


class FalAdapter:
    def __init__(self, config: FalConfig) -> None:
        self.config = config
        api_key = (
            config.api_key
            or (config.api_keys[0] if config.api_keys else "")
            or os.getenv("AI_KIT_FAL_API_KEY")
            or os.getenv("FAL_API_KEY")
            or os.getenv("FAL_KEY")
        )
        if not api_key:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message="Fal API key is required",
                    provider="fal",
                )
            )
        self._client = FalClient(api_key=api_key, timeout_s=config.timeout_s)

    def list_models(self) -> List[ModelMetadata]:
        models = [model for model in load_catalog_models() if model.provider == "fal"]
        if models:
            return models
        return [
            ModelMetadata(
                id="fal-ai/sync-lipsync/v2/pro",
                displayName="fal-ai/sync-lipsync/v2/pro",
                provider="fal",
                family="lipsync",
                capabilities=ModelCapabilities(
                    text=False,
                    vision=False,
                    image=False,
                    video=True,
                    tool_use=False,
                    structured_output=False,
                    reasoning=False,
                    audio_in=True,
                    video_in=True,
                ),
            )
        ]

    def generate_video(self, input: VideoGenerateInput) -> VideoGenerateOutput:
        temp_paths: List[Path] = []
        try:
            allowed_inputs, required_inputs = _model_input_names(input.model)
            image_url = _resolve_image_url(self._client, input.startImage, input.inputImages, temp_paths)
            if not image_url:
                raise AiKitError(
                    KitErrorPayload(
                        kind=ErrorKind.VALIDATION,
                        message="Fal video generation requires an image input",
                        provider="fal",
                    )
                )

            audio_url = _resolve_audio_url(self._client, input.audioUrl, input.audioBase64, temp_paths)
            if "audio_url" in required_inputs and not audio_url:
                raise AiKitError(
                    KitErrorPayload(
                        kind=ErrorKind.VALIDATION,
                        message="Fal video generation requires an audio input",
                        provider="fal",
                    )
                )

            payload: Dict[str, Any] = {}
            if _allow_input("prompt", allowed_inputs):
                payload["prompt"] = input.prompt
            if _allow_input("image_url", allowed_inputs):
                payload["image_url"] = image_url
            if audio_url and _allow_input("audio_url", allowed_inputs):
                payload["audio_url"] = audio_url
            if input.duration is not None and _allow_input("duration", allowed_inputs):
                payload["duration"] = _format_duration(input.duration)
            if input.negativePrompt and _allow_input("negative_prompt", allowed_inputs):
                payload["negative_prompt"] = input.negativePrompt
            if input.generateAudio is not None and _allow_input("generate_audio", allowed_inputs):
                payload["generate_audio"] = input.generateAudio
            if input.parameters:
                for key, value in input.parameters.items():
                    if _allow_input(key, allowed_inputs):
                        payload[key] = value

            result = self._client.subscribe(input.model, arguments=payload)
            video = result.get("video") if isinstance(result, dict) else None
            out_url = video.get("url") if isinstance(video, dict) else None
            if not out_url:
                raise AiKitError(
                    KitErrorPayload(
                        kind=ErrorKind.UNKNOWN,
                        message="Fal video response missing video url",
                        provider="fal",
                    )
                )

            data = self._client.download_url(out_url)
            return VideoGenerateOutput(
                mime="video/mp4",
                data=base64.b64encode(data).decode("ascii"),
                raw=result,
            )
        finally:
            for path in temp_paths:
                try:
                    path.unlink()
                except OSError:
                    pass

    def generate_lipsync(self, input: LipsyncGenerateInput) -> LipsyncGenerateOutput:
        video_url = input.videoUrl
        video_tmp: List[Path] = []
        if not video_url and input.videoBase64:
            temp_path = write_temp_file(input.videoBase64, ".mp4")
            video_tmp.append(temp_path)
            video_url = self._client.upload_file(temp_path)

        audio_url = input.audioUrl
        audio_tmp: List[Path] = []
        if not audio_url and input.audioBase64:
            suffix = guess_extension(data_url_media_type(input.audioBase64), default=".wav")
            temp_path = write_temp_file(input.audioBase64, suffix)
            audio_tmp.append(temp_path)
            audio_url = self._client.upload_file(temp_path)

        temp_paths = [*video_tmp, *audio_tmp]
        try:
            if not video_url:
                raise AiKitError(
                    KitErrorPayload(
                        kind=ErrorKind.VALIDATION,
                        message="Fal lipsync requires a video input",
                        provider="fal",
                    )
                )
            if not audio_url:
                raise AiKitError(
                    KitErrorPayload(
                        kind=ErrorKind.VALIDATION,
                        message="Fal lipsync requires an audio input",
                        provider="fal",
                    )
                )

            payload: Dict[str, Any] = {"video_url": video_url, "audio_url": audio_url}
            if input.parameters:
                payload.update(input.parameters)

            result = self._client.subscribe(input.model, arguments=payload)
            video = result.get("video") if isinstance(result, dict) else None
            out_url = video.get("url") if isinstance(video, dict) else None
            if not out_url:
                raise AiKitError(
                    KitErrorPayload(
                        kind=ErrorKind.UNKNOWN,
                        message="Fal lipsync response missing video url",
                        provider="fal",
                    )
                )

            data = self._client.download_url(out_url)
            return LipsyncGenerateOutput(
                mime="video/mp4",
                data=base64.b64encode(data).decode("ascii"),
                raw=result,
            )
        finally:
            for path in temp_paths:
                try:
                    path.unlink()
                except OSError:
                    pass


def _resolve_image_url(
    client: FalClient,
    start_image: Optional[str],
    input_images: Optional[List[Any]],
    temp_paths: List[Path],
) -> Optional[str]:
    if start_image:
        url = _coerce_image_entry(client, start_image, temp_paths)
        if url:
            return url
    for entry in input_images or []:
        url = _coerce_image_entry(client, entry, temp_paths)
        if url:
            return url
    return None


def _coerce_image_entry(
    client: FalClient,
    entry: Any,
    temp_paths: List[Path],
) -> Optional[str]:
    image = _image_input_from_entry(entry)
    if not image:
        return None
    if image.url:
        return _coerce_image_url(client, image.url, image.mediaType, temp_paths)
    if image.base64:
        return _upload_base64_image(client, image.base64, image.mediaType, temp_paths)
    return None


def _resolve_audio_url(
    client: FalClient,
    audio_url: Optional[str],
    audio_base64: Optional[str],
    temp_paths: List[Path],
) -> Optional[str]:
    if audio_url:
        return _coerce_audio_url(client, audio_url, temp_paths)
    if audio_base64:
        suffix = guess_extension(data_url_media_type(audio_base64), default=".wav")
        temp_path = write_temp_file(audio_base64, suffix)
        temp_paths.append(temp_path)
        return client.upload_file(temp_path)
    return None


def _image_input_from_entry(entry: Any) -> Optional[ImageInput]:
    if isinstance(entry, ImageInput):
        return entry
    if isinstance(entry, dict):
        return ImageInput(
            url=entry.get("url"),
            base64=entry.get("base64"),
            mediaType=entry.get("mediaType"),
        )
    if isinstance(entry, str):
        return ImageInput(url=entry)
    return None


def _coerce_image_url(
    client: FalClient,
    value: str,
    media_type: Optional[str],
    temp_paths: List[Path],
) -> Optional[str]:
    if value.startswith("data:"):
        suffix = guess_extension(data_url_media_type(value) or media_type)
        temp_path = write_temp_file(value, suffix)
        temp_paths.append(temp_path)
        return client.upload_file(temp_path)
    if value.startswith("file://"):
        path = Path(value[7:])
        if path.exists():
            return client.upload_file(path)
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    path = Path(value).expanduser()
    if path.exists():
        return client.upload_file(path)
    return value


def _coerce_audio_url(
    client: FalClient,
    value: str,
    temp_paths: List[Path],
) -> Optional[str]:
    if value.startswith("data:"):
        suffix = guess_extension(data_url_media_type(value), default=".wav")
        temp_path = write_temp_file(value, suffix)
        temp_paths.append(temp_path)
        return client.upload_file(temp_path)
    if value.startswith("file://"):
        path = Path(value[7:])
        if path.exists():
            return client.upload_file(path)
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    path = Path(value).expanduser()
    if path.exists():
        return client.upload_file(path)
    return value


def _upload_base64_image(
    client: FalClient,
    data: str,
    media_type: Optional[str],
    temp_paths: List[Path],
) -> str:
    suffix = guess_extension(media_type)
    temp_path = write_temp_file(data, suffix)
    temp_paths.append(temp_path)
    return client.upload_file(temp_path)


def _allow_input(name: str, allowed: Optional[set[str]]) -> bool:
    return not allowed or name in allowed


def _model_input_names(model_id: str) -> tuple[Optional[set[str]], set[str]]:
    for model in load_catalog_models():
        if model.provider != "fal" or model.id != model_id:
            continue
        inputs = model.inputs
        if not isinstance(inputs, list) or not inputs:
            return None, set()
        allowed: set[str] = set()
        required: set[str] = set()
        for entry in inputs:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            allowed.add(name)
            if entry.get("required") is True:
                required.add(name)
        return (allowed or None), required
    return None, set()


def _format_duration(value: Any) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)
