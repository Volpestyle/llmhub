from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


def _load_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(
            "google-genai is required for Gemini image generation. "
            "Install it with `pip install google-genai`."
        ) from exc
    return genai, types


def _env_api_key() -> str:
    return (
        os.getenv("AI_KIT_GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )


def _coerce_image(value: Union[Image.Image, bytes, bytearray, str, Path]) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        with Image.open(BytesIO(value)) as img:
            return img.copy()
    if isinstance(value, (str, Path)):
        with Image.open(value) as img:
            return img.copy()
    raise TypeError(f"Unsupported image input type: {type(value)}")


def _extract_images(response: Any) -> List[bytes]:
    images: List[bytes] = []
    parts = getattr(response, "parts", None) or []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        img = part.as_image()
        buf = BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())
    if images:
        return images
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            img = part.as_image()
            buf = BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
    return images


class GeminiImageClient:
    def __init__(self, *, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or _env_api_key()
        self._client = None

    def generate_images(
        self,
        *,
        model: str,
        prompt: str,
        input_image: Union[Image.Image, bytes, bytearray, str, Path],
        response_modalities: Optional[List[str]] = None,
        image_config: Optional[Dict[str, Any]] = None,
    ) -> List[bytes]:
        if not prompt:
            raise ValueError("Prompt is required for Gemini image generation")
        genai, types = _load_genai()
        client = self._get_client(genai)
        image = _coerce_image(input_image)
        config_kwargs: Dict[str, Any] = {}
        modalities = response_modalities or ["Image"]
        if modalities:
            config_kwargs["response_modalities"] = modalities
        if image_config:
            config_kwargs["image_config"] = types.ImageConfig(**image_config)
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image],
            config=config,
        )
        images = _extract_images(response)
        if not images:
            raise RuntimeError("Gemini response did not include image data")
        return images

    def _get_client(self, genai_module):
        if self._client is None:
            if self.api_key:
                self._client = genai_module.Client(api_key=self.api_key)
            else:
                self._client = genai_module.Client()
        return self._client
