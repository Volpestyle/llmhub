from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .openai import OpenAIAdapter, OpenAIConfig
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
