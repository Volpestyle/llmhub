from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .openai import OpenAIAdapter, OpenAIConfig
from ..errors import ErrorKind, InferenceKitError, KitErrorPayload
from ..types import ImageGenerateInput, ImageGenerateOutput, Provider


@dataclass
class OllamaConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None
    base_url: str = "http://localhost:11434"
    default_use_responses: bool = False
    timeout: Optional[float] = None


class OllamaAdapter(OpenAIAdapter):
    def __init__(self, config: OllamaConfig, provider: Provider = "ollama") -> None:
        openai_config = OpenAIConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            default_use_responses=config.default_use_responses,
            timeout=config.timeout,
        )
        super().__init__(openai_config, provider=provider)

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Ollama image generation is not supported",
                provider=self.provider,
            )
        )

    def generate_mesh(self, input: "MeshGenerateInput"):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Ollama mesh generation is not supported",
                provider=self.provider,
            )
        )
