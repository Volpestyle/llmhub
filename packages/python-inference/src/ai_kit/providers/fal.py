from __future__ import annotations

import base64
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..catalog import load_catalog_models
from ..errors import ErrorKind, KitErrorPayload, AiKitError
from ..types import LipsyncGenerateInput, LipsyncGenerateOutput, ModelCapabilities, ModelMetadata
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

    def generate_lipsync(self, input: LipsyncGenerateInput) -> LipsyncGenerateOutput:
        video_url = input.videoUrl
        video_tmp: List[Path] = []
        if not video_url and input.videoBase64:
            temp_path = _write_temp_file(input.videoBase64, ".mp4")
            video_tmp.append(temp_path)
            video_url = self._client.upload_file(temp_path)

        audio_url = input.audioUrl
        audio_tmp: List[Path] = []
        if not audio_url and input.audioBase64:
            temp_path = _write_temp_file(input.audioBase64, ".wav")
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


def _write_temp_file(data: str, suffix: str) -> Path:
    raw = _decode_base64(data)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp.write(raw)
        temp.flush()
    finally:
        temp.close()
    return Path(temp.name)


def _decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        payload = value.split(",", 1)[1]
        return base64.b64decode(payload)
    return base64.b64decode(value)
