from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..http import request_json, request_stream
from ..errors import ErrorKind, KitErrorPayload, InferenceKitError
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
    ToolCall,
    ToolChoice,
    ToolDefinition,
    Usage,
    as_json_dict,
    ensure_messages,
)


@dataclass
class OpenAIConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None
    base_url: str = "https://api.openai.com"
    organization: Optional[str] = None
    default_use_responses: bool = True
    timeout: Optional[float] = None


class OpenAIAdapter:
    def __init__(self, config: OpenAIConfig, provider: Provider = "openai") -> None:
        self.config = config
        self.provider = provider
        self.base_url = config.base_url.rstrip("/")

    def list_models(self) -> List[ModelMetadata]:
        url = f"{self.base_url}/v1/models"
        payload = request_json("GET", url, self._headers(), timeout=self.config.timeout)
        models: List[ModelMetadata] = []
        for model in payload.get("data", []):
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
                    contextWindow=None,
                    tokenPrices=None,
                )
            )
        return models

    def generate(self, input: GenerateInput) -> GenerateOutput:
        if self._should_use_responses(input):
            return self._generate_responses(input)
        return self._generate_chat(input)

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        if input.inputImages:
            raise InferenceKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message="OpenAI image edits are not supported in this adapter",
                    provider=self.provider,
                )
            )
        url = f"{self.base_url}/v1/images"
        payload = request_json(
            "POST",
            url,
            self._headers(),
            json_body={
                "model": input.model,
                "prompt": input.prompt,
                "size": input.size or "1024x1024",
                "response_format": "b64_json",
                "n": 1,
            },
            timeout=self.config.timeout,
        )
        data = payload.get("data", []) or []
        image = data[0] if data else {}
        b64 = image.get("b64_json")
        if not b64:
            raise InferenceKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNKNOWN,
                    message="OpenAI image response missing base64 data",
                    provider=self.provider,
                )
            )
        return ImageGenerateOutput(mime="image/png", data=b64, raw=payload)

    def generate_mesh(self, input: "MeshGenerateInput"):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="OpenAI mesh generation is not supported",
                provider=self.provider,
            )
        )

    def stream_generate(self, input: GenerateInput) -> Iterable[StreamChunk]:
        if self._should_use_responses(input):
            return self._stream_responses(input)
        return self._stream_chat(input)

    def _generate_responses(self, input: GenerateInput) -> GenerateOutput:
        url = f"{self.base_url}/v1/responses"
        payload = request_json(
            "POST",
            url,
            self._headers(),
            json_body=_build_responses_payload(input, stream=False),
            timeout=self.config.timeout,
        )
        return _normalize_responses_output(payload)

    def _generate_chat(self, input: GenerateInput) -> GenerateOutput:
        url = f"{self.base_url}/v1/chat/completions"
        payload = request_json(
            "POST",
            url,
            self._headers(),
            json_body=_build_chat_payload(input, stream=False),
            timeout=self.config.timeout,
        )
        return _normalize_chat_output(payload)

    def _stream_responses(self, input: GenerateInput) -> Iterable[StreamChunk]:
        url = f"{self.base_url}/v1/responses"
        response = request_stream(
            "POST",
            url,
            self._headers(),
            json_body=_build_responses_payload(input, stream=True),
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
            if event_type in ("response.output_text.delta", "response.refusal.delta"):
                delta = payload.get("delta", {})
                text = delta.get("text") if isinstance(delta, dict) else delta
                if text:
                    yield StreamChunk(type="delta", textDelta=text)
            elif event_type == "response.completed":
                usage = _map_responses_usage(payload.get("response", {}).get("usage"))
                yield StreamChunk(type="message_end", usage=usage, finishReason=payload.get("response", {}).get("status"))
            elif event_type == "response.error":
                error = payload.get("error", {})
                yield StreamChunk(
                    type="error",
                    error={
                        "kind": "upstream_error",
                        "message": error.get("message", "OpenAI streaming error"),
                        "upstreamCode": error.get("code"),
                    },
                )
                return
        response.close()

    def _stream_chat(self, input: GenerateInput) -> Iterable[StreamChunk]:
        url = f"{self.base_url}/v1/chat/completions"
        response = request_stream(
            "POST",
            url,
            self._headers(),
            json_body=_build_chat_payload(input, stream=True),
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
            for choice in payload.get("choices", []) or []:
                delta = choice.get("delta", {})
                content = delta.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                            yield StreamChunk(type="delta", textDelta=part["text"])
                elif isinstance(content, str):
                    yield StreamChunk(type="delta", textDelta=content)
                if choice.get("finish_reason"):
                    usage = _map_chat_usage(payload.get("usage"))
                    yield StreamChunk(type="message_end", usage=usage, finishReason=choice.get("finish_reason"))
        response.close()

    def _should_use_responses(self, input: GenerateInput) -> bool:
        return self.config.default_use_responses

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        return headers


def _build_responses_payload(input: GenerateInput, stream: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": input.model,
        "input": _map_messages_to_responses(input.messages),
        "tools": _map_tools(input.tools),
        "tool_choice": _map_tool_choice(input.toolChoice),
        "temperature": input.temperature,
        "top_p": input.topP,
        "max_output_tokens": input.maxTokens,
        "metadata": input.metadata,
        "stream": stream,
    }
    response_format = _map_response_format(input.responseFormat)
    if response_format is not None:
        payload["text"] = {"format": response_format}
    return {k: v for k, v in payload.items() if v is not None}


def _build_chat_payload(input: GenerateInput, stream: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": input.model,
        "messages": _map_messages_to_chat(input.messages),
        "temperature": input.temperature,
        "top_p": input.topP,
        "max_tokens": input.maxTokens,
        "stream": stream,
        "tools": _map_tools(input.tools),
        "tool_choice": _map_tool_choice(input.toolChoice),
    }
    response_format = _map_response_format(input.responseFormat)
    if response_format is not None:
        payload["response_format"] = response_format
    return {k: v for k, v in payload.items() if v is not None}


def _map_messages_to_responses(messages: Iterable[Message | Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = ensure_messages(messages)
    output: List[Dict[str, Any]] = []
    for message in normalized:
        parts = []
        for part in message.get("content", []) or []:
            if part.get("type") == "text":
                parts.append({"type": "input_text", "text": part.get("text")})
            elif part.get("type") == "image":
                image = part.get("image") or {}
                payload: Dict[str, Any] = {
                    "type": "input_image",
                    "media_type": image.get("mediaType"),
                }
                if image.get("url"):
                    payload["image_url"] = image.get("url")
                if image.get("base64"):
                    payload["image_base64"] = image.get("base64")
                parts.append(payload)
        entry: Dict[str, Any] = {"role": message.get("role"), "content": parts}
        if message.get("toolCallId"):
            entry["tool_call_id"] = message.get("toolCallId")
        if message.get("name"):
            entry["name"] = message.get("name")
        output.append(entry)
    return output


def _map_messages_to_chat(messages: Iterable[Message | Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = ensure_messages(messages)
    output: List[Dict[str, Any]] = []
    for message in normalized:
        parts = []
        for part in message.get("content", []) or []:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text")})
            elif part.get("type") == "image":
                image = part.get("image") or {}
                image_payload: Dict[str, Any] = {}
                if image.get("url"):
                    image_payload["url"] = image.get("url")
                if image.get("base64"):
                    image_payload["b64_json"] = image.get("base64")
                parts.append({"type": "image_url", "image_url": image_payload})
        content: Any
        if len(parts) == 1 and parts[0].get("type") == "text":
            content = parts[0].get("text")
        else:
            content = parts
        output.append({"role": message.get("role"), "content": content})
    return output


def _map_tools(tools: Optional[List[ToolDefinition]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    serialized = []
    for tool in tools:
        payload = as_json_dict(tool)
        serialized.append(
            {
                "type": "function",
                "function": {
                    "name": payload.get("name"),
                    "description": payload.get("description"),
                    "parameters": payload.get("parameters"),
                },
            }
        )
    return serialized


def _map_tool_choice(choice: Optional[ToolChoice]) -> Optional[Any]:
    if not choice:
        return None
    payload = as_json_dict(choice)
    if payload.get("type") in ("auto", "none"):
        return payload.get("type")
    return {
        "type": "function",
        "function": {"name": payload.get("name")},
    }


def _map_response_format(format_obj: Optional[Any]) -> Optional[Dict[str, Any]]:
    if not format_obj:
        return None
    payload = as_json_dict(format_obj)
    if payload.get("type") == "json_schema" and payload.get("jsonSchema"):
        schema = payload.get("jsonSchema")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("name"),
                "schema": schema.get("schema"),
                "strict": schema.get("strict", True),
            },
        }
    return payload


def _normalize_chat_output(payload: Dict[str, Any]) -> GenerateOutput:
    choices = payload.get("choices", []) or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) or {}
    content = message.get("content")
    text_parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
    elif isinstance(content, str):
        text_parts.append(content)
    return GenerateOutput(
        text="".join(text_parts) if text_parts else None,
        toolCalls=_map_tool_calls(message.get("tool_calls")),
        finishReason=choice.get("finish_reason"),
        usage=_map_chat_usage(payload.get("usage")),
        raw=payload,
    )


def _normalize_responses_output(payload: Dict[str, Any]) -> GenerateOutput:
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    for output in payload.get("output", []) or []:
        for content in output.get("content", []) or []:
            if content.get("type") == "output_text" and content.get("text"):
                text_parts.append(content.get("text"))
            if content.get("type") == "refusal" and content.get("refusal"):
                text_parts.append(content.get("refusal"))
            if content.get("type") == "tool_call":
                tool_calls.append(
                    ToolCall(
                        id=content.get("id") or f"tool_{len(tool_calls)}",
                        name=content.get("name") or "",
                        argumentsJson=json.dumps(content.get("arguments", {})),
                    )
                )
    return GenerateOutput(
        text="".join(text_parts) if text_parts else None,
        toolCalls=tool_calls or None,
        finishReason=payload.get("status"),
        usage=_map_responses_usage(payload.get("usage")),
        raw=payload,
    )


def _map_tool_calls(raw_calls: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolCall]]:
    if not raw_calls:
        return None
    return [
        ToolCall(
            id=call.get("id"),
            name=call.get("function", {}).get("name", ""),
            argumentsJson=call.get("function", {}).get("arguments", ""),
        )
        for call in raw_calls
    ]


def _map_chat_usage(raw: Optional[Dict[str, Any]]) -> Optional[Usage]:
    if not raw:
        return None
    return Usage(
        inputTokens=raw.get("prompt_tokens"),
        outputTokens=raw.get("completion_tokens"),
        totalTokens=raw.get("total_tokens"),
    )


def _map_responses_usage(raw: Optional[Dict[str, Any]]) -> Optional[Usage]:
    if not raw:
        return None
    return Usage(
        inputTokens=raw.get("input_tokens"),
        outputTokens=raw.get("output_tokens"),
        totalTokens=raw.get("total_tokens"),
    )


def _derive_family(model_id: str) -> str:
    if not model_id:
        return ""
    parts = model_id.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return model_id
