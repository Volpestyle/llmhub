from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..http import request_json, request_stream
from ..overlays import lookup_curated
from ..sse import iter_sse_events
from ..types import (
    GenerateInput,
    GenerateOutput,
    ModelCapabilities,
    ModelMetadata,
    Provider,
    StreamChunk,
    ensure_messages,
)


@dataclass
class GeminiConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None
    base_url: str = "https://generativelanguage.googleapis.com"
    timeout: Optional[float] = None


class GeminiAdapter:
    def __init__(self, config: GeminiConfig, provider: Provider = "google") -> None:
        self.config = config
        self.provider = provider
        self.base_url = config.base_url.rstrip("/")

    def list_models(self) -> List[ModelMetadata]:
        url = f"{self.base_url}/v1beta/models?key={self.config.api_key}"
        payload = request_json("GET", url, self._headers(), timeout=self.config.timeout)
        models: List[ModelMetadata] = []
        for model in payload.get("models", []) or []:
            name = model.get("name")
            if not name:
                continue
            model_id = name.replace("models/", "")
            curated = lookup_curated(self.provider, model_id)
            if curated:
                models.append(curated)
                continue
            models.append(
                ModelMetadata(
                    id=model_id,
                    displayName=model.get("displayName") or model_id,
                    provider=self.provider,
                    family=_derive_family(model_id),
                    capabilities=ModelCapabilities(
                        text=True,
                        vision=True,
                        tool_use=True,
                        structured_output=True,
                        reasoning=False,
                    ),
                    contextWindow=model.get("inputTokenLimit"),
                )
            )
        return models

    def generate(self, input: GenerateInput) -> GenerateOutput:
        url = f"{self.base_url}/v1beta/models/{_ensure_models_prefix(input.model)}:generateContent?key={self.config.api_key}"
        payload = request_json(
            "POST",
            url,
            self._headers(),
            json_body=_build_payload(input),
            timeout=self.config.timeout,
        )
        return _normalize_output(payload)

    def stream_generate(self, input: GenerateInput) -> Iterable[StreamChunk]:
        url = f"{self.base_url}/v1beta/models/{_ensure_models_prefix(input.model)}:streamGenerateContent?alt=sse&key={self.config.api_key}"
        response = request_stream(
            "POST",
            url,
            self._headers(),
            json_body=_build_payload(input),
            timeout=self.config.timeout,
        )
        for event in iter_sse_events(response.iter_lines(decode_unicode=True)):
            data = event.get("data")
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            text = _extract_text(payload)
            if text:
                yield StreamChunk(type="delta", textDelta=text)
        response.close()

    def _headers(self) -> Dict[str, str]:
        return {"content-type": "application/json"}


def _build_payload(input: GenerateInput) -> Dict[str, Any]:
    messages = ensure_messages(input.messages)
    contents = []
    for message in messages:
        parts = []
        for part in message.get("content", []) or []:
            if part.get("type") == "text":
                parts.append({"text": part.get("text")})
        if parts:
            contents.append({"role": message.get("role"), "parts": parts})
    payload: Dict[str, Any] = {"contents": contents}
    generation_config: Dict[str, Any] = {}
    if input.temperature is not None:
        generation_config["temperature"] = input.temperature
    if input.topP is not None:
        generation_config["topP"] = input.topP
    if input.maxTokens is not None:
        generation_config["maxOutputTokens"] = input.maxTokens
    if generation_config:
        payload["generationConfig"] = generation_config
    return payload


def _normalize_output(payload: Dict[str, Any]) -> GenerateOutput:
    text = _extract_text(payload)
    return GenerateOutput(text=text or None, raw=payload)


def _extract_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates", []) or []
    if not candidates:
        return ""
    content = candidates[0].get("content", {}) or {}
    parts = content.get("parts", []) or []
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return "".join(texts)


def _ensure_models_prefix(model_id: str) -> str:
    if model_id.startswith("models/"):
        return model_id
    return f"models/{model_id}"


def _derive_family(model_id: str) -> str:
    parts = model_id.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else model_id
