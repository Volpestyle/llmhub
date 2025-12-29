from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..errors import ErrorKind, KitErrorPayload, InferenceKitError
from ..http import request_json, request_stream
from ..sse import iter_sse_events
from ..types import (
    GenerateInput,
    GenerateOutput,
    ImageGenerateInput,
    ImageGenerateOutput,
    Message,
    ModelCapabilities,
    ModelMetadata,
    Provider,
    StreamChunk,
    ToolChoice,
    ToolDefinition,
    Usage,
    as_json_dict,
    ensure_messages,
)


DEFAULT_VERSION = "2023-06-01"


@dataclass
class AnthropicConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None
    base_url: str = "https://api.anthropic.com"
    version: str = DEFAULT_VERSION
    timeout: Optional[float] = None


class AnthropicAdapter:
    def __init__(self, config: AnthropicConfig, provider: Provider = "anthropic") -> None:
        self.config = config
        self.provider = provider
        self.base_url = config.base_url.rstrip("/")

    def list_models(self) -> List[ModelMetadata]:
        url = f"{self.base_url}/v1/models"
        payload = request_json("GET", url, self._headers(), timeout=self.config.timeout)
        models: List[ModelMetadata] = []
        for model in payload.get("data", []) or []:
            model_id = model.get("id")
            if not model_id:
                continue
            models.append(
                ModelMetadata(
                    id=model_id,
                    displayName=model_id,
                    provider=self.provider,
                    family=_derive_family(model_id),
                    capabilities=ModelCapabilities(
                        text=True,
                        vision=False,
                        tool_use=True,
                        structured_output=False,
                        reasoning=False,
                    ),
                )
            )
        return models

    def generate(self, input: GenerateInput) -> GenerateOutput:
        url = f"{self.base_url}/v1/messages"
        payload = request_json(
            "POST",
            url,
            self._headers(),
            json_body=_build_payload(input, stream=False),
            timeout=self.config.timeout,
        )
        return _normalize_output(payload)

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Anthropic image generation is not supported",
                provider=self.provider,
            )
        )

    def generate_mesh(self, input: "MeshGenerateInput"):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Anthropic mesh generation is not supported",
                provider=self.provider,
            )
        )

    def stream_generate(self, input: GenerateInput) -> Iterable[StreamChunk]:
        url = f"{self.base_url}/v1/messages"
        response = request_stream(
            "POST",
            url,
            self._headers(),
            json_body=_build_payload(input, stream=True),
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
            event_type = event.get("event")
            if event_type == "content_block_delta":
                delta = payload.get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield StreamChunk(type="delta", textDelta=delta.get("text"))
            elif event_type == "message_stop":
                yield StreamChunk(type="message_end", finishReason="stop")
        response.close()

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.config.version,
            "content-type": "application/json",
        }


def _build_payload(input: GenerateInput, stream: bool) -> Dict[str, Any]:
    messages = _map_messages(input.messages)
    payload = {
        "model": input.model,
        "messages": messages,
        "temperature": input.temperature,
        "top_p": input.topP,
        "max_tokens": input.maxTokens or 1024,
        "stream": stream,
    }
    if input.tools:
        payload["tools"] = _map_tools(input.tools)
    if input.toolChoice:
        payload["tool_choice"] = _map_tool_choice(input.toolChoice)
    return payload


def _map_messages(messages: Iterable[Message | Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = ensure_messages(messages)
    output = []
    for message in normalized:
        parts = []
        for part in message.get("content", []) or []:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text")})
            elif part.get("type") == "image":
                image = part.get("image") or {}
                if image.get("base64"):
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.get("mediaType"),
                                "data": image.get("base64"),
                            },
                        }
                    )
        output.append({"role": message.get("role"), "content": parts})
    return output


def _map_tools(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    serialized = []
    for tool in tools:
        payload = as_json_dict(tool)
        serialized.append(
            {
                "name": payload.get("name"),
                "description": payload.get("description"),
                "input_schema": payload.get("parameters"),
            }
        )
    return serialized


def _map_tool_choice(choice: ToolChoice) -> Dict[str, Any]:
    payload = as_json_dict(choice)
    if payload.get("type") in ("auto", "none"):
        return {"type": payload.get("type")}
    return {"type": "tool", "name": payload.get("name")}


def _normalize_output(payload: Dict[str, Any]) -> GenerateOutput:
    text_parts = []
    for part in payload.get("content", []) or []:
        if part.get("type") == "text" and part.get("text"):
            text_parts.append(part.get("text"))
    return GenerateOutput(
        text="".join(text_parts) if text_parts else None,
        finishReason=payload.get("stop_reason"),
        usage=_map_usage(payload.get("usage")),
        raw=payload,
    )


def _map_usage(raw: Optional[Dict[str, Any]]) -> Optional[Usage]:
    if not raw:
        return None
    return Usage(
        inputTokens=raw.get("input_tokens"),
        outputTokens=raw.get("output_tokens"),
        totalTokens=(raw.get("input_tokens") or 0) + (raw.get("output_tokens") or 0),
    )


def _derive_family(model_id: str) -> str:
    if not model_id:
        return ""
    parts = model_id.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else model_id
