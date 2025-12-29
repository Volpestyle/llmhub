from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .openai import OpenAIAdapter, OpenAIConfig
from ..errors import ErrorKind, KitErrorPayload, InferenceKitError
from ..types import ImageGenerateInput, ImageGenerateOutput
from ..types import Provider


@dataclass
class XAIConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None
    base_url: str = "https://api.x.ai"
    compatibility_mode: str = "openai"
    timeout: Optional[float] = None


class XAIAdapter(OpenAIAdapter):
    def __init__(self, config: XAIConfig, provider: Provider = "xai") -> None:
        openai_config = OpenAIConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            default_use_responses=True,
            timeout=config.timeout,
        )
        super().__init__(openai_config, provider=provider)
        self.compatibility_mode = config.compatibility_mode

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="xAI image generation is not supported",
                provider=self.provider,
            )
        )

    def generate_mesh(self, input: "MeshGenerateInput"):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="xAI mesh generation is not supported",
                provider=self.provider,
            )
        )
