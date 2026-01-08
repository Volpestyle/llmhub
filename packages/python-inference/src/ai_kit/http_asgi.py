from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qs

from .errors import ErrorKind, KitErrorPayload, AiKitError, to_kit_error
from .hub import Kit
from .types import (
    GenerateInput,
    ImageGenerateInput,
    MeshGenerateInput,
    SpeechGenerateInput,
    VideoGenerateInput,
    TranscribeInput,
    as_json_dict,
)

ASGIApp = Callable[[Dict[str, Any], Callable[..., Awaitable[Dict[str, Any]]], Callable[..., Awaitable[None]]], Awaitable[None]]


def create_asgi_app(kit: Kit, base_path: str = "") -> ASGIApp:
    base = _normalize_base_path(base_path)

    async def app(scope: Dict[str, Any], receive, send) -> None:
        if scope.get("type") != "http":
            return
        method = (scope.get("method") or "GET").upper()
        path = scope.get("path") or "/"
        if base:
            root_path = _normalize_base_path(scope.get("root_path") or "")
            if root_path != base:
                if not path.startswith(base):
                    await _respond_text(send, 404, "not found")
                    return
                path = path[len(base) :] or "/"

        if path == "/provider-models":
            if method != "GET":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_models(kit, scope, send)
            return
        if path == "/generate":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_generate(kit, receive, send)
            return
        if path == "/image":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_image(kit, receive, send)
            return
        if path == "/mesh":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_mesh(kit, receive, send)
            return
        if path == "/video":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_video(kit, receive, send)
            return
        if path == "/speech":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_speech(kit, receive, send)
            return
        if path == "/transcribe":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_transcribe(kit, receive, send)
            return
        if path == "/generate/stream":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_stream(kit, receive, send)
            return

        await _respond_text(send, 404, "not found")

    return app


def _normalize_base_path(value: str) -> str:
    if not value:
        return ""
    base = value.rstrip("/")
    if not base.startswith("/"):
        base = "/" + base
    return base


async def _handle_models(kit: Kit, scope: Dict[str, Any], send) -> None:
    try:
        query = _parse_query(scope)
        providers = _parse_providers(_first_query_value(query, "providers"))
        refresh = _should_refresh(query.get("refresh", []))
        models = kit.list_models(providers=providers, refresh=refresh)
        await _respond_json(send, 200, as_json_dict(models))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_generate(kit: Kit, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_generate_input(payload)
        output = kit.generate(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_image(kit: Kit, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_image_input(payload)
        output = kit.generate_image(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_mesh(kit: Kit, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_mesh_input(payload)
        output = kit.generate_mesh(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_video(kit: Kit, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_video_input(payload)
        output = kit.generate_video(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_speech(kit: Kit, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_speech_input(payload)
        output = kit.generate_speech(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_transcribe(kit: Kit, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_transcribe_input(payload)
        output = kit.transcribe(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_stream(kit: Kit, receive, send) -> None:
    started = False
    try:
        payload = await _read_json(receive)
        input_data = _normalize_generate_input(payload, force_stream=True)
        await _start_sse(send)
        started = True
        for chunk in kit.stream_generate(input_data):
            await _send_sse_event(send, "chunk", as_json_dict(chunk), more_body=True)
        await _send_sse_event(send, "done", {"ok": True}, more_body=False)
    except Exception as err:
        kit_err = to_kit_error(err)
        if not started:
            await _send_json_error(send, kit_err)
            return
        await _send_sse_event(
            send,
            "error",
            {
                "kind": kit_err.kind.value,
                "message": kit_err.message,
                "requestId": kit_err.requestId,
            },
            more_body=False,
        )


async def _read_body(receive) -> bytes:
    body = b""
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            continue
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    return body


async def _read_json(receive) -> Any:
    body = await _read_body(receive)
    if not body:
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="Body must be valid JSON")
        )
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="Body must be valid JSON")
        ) from exc


def _parse_query(scope: Dict[str, Any]) -> Dict[str, List[str]]:
    raw = scope.get("query_string") or b""
    parsed = parse_qs(raw.decode("utf-8"))
    return {k: [str(v) for v in values] for k, values in parsed.items()}


def _first_query_value(query: Dict[str, List[str]], key: str) -> Optional[str]:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def _parse_providers(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    providers = [entry.strip() for entry in raw.split(",") if entry.strip()]
    return providers or None


def _should_refresh(values: Iterable[str]) -> bool:
    value = next(iter(values), None)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in ("", "1", "true", "yes", "on")


def _normalize_generate_input(payload: Any, force_stream: bool = False) -> GenerateInput:
    if not isinstance(payload, dict):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be a GenerateInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    messages = payload.get("messages")
    if not isinstance(provider, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(messages, list):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="messages is required and must be an array")
        )
    return GenerateInput(
        provider=provider,
        model=model,
        messages=messages,
        tools=payload.get("tools"),
        toolChoice=payload.get("toolChoice"),
        responseFormat=payload.get("responseFormat"),
        temperature=payload.get("temperature"),
        topP=payload.get("topP"),
        maxTokens=payload.get("maxTokens"),
        stream=True if force_stream else payload.get("stream"),
        metadata=payload.get("metadata"),
    )


def _normalize_image_input(payload: Any) -> ImageGenerateInput:
    if not isinstance(payload, dict):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be an ImageGenerateInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not isinstance(provider, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(prompt, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="prompt is required and must be a string")
        )
    return ImageGenerateInput(
        provider=provider,
        model=model,
        prompt=prompt,
        size=payload.get("size"),
        inputImages=payload.get("inputImages"),
        parameters=payload.get("parameters"),
    )


def _normalize_mesh_input(payload: Any) -> MeshGenerateInput:
    if not isinstance(payload, dict):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be a MeshGenerateInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not isinstance(provider, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(prompt, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="prompt is required and must be a string")
        )
    return MeshGenerateInput(
        provider=provider,
        model=model,
        prompt=prompt,
        inputImages=payload.get("inputImages"),
        format=payload.get("format"),
    )


def _normalize_video_input(payload: Any) -> VideoGenerateInput:
    if not isinstance(payload, dict):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be a VideoGenerateInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not isinstance(provider, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(prompt, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="prompt is required and must be a string")
        )
    return VideoGenerateInput(
        provider=provider,
        model=model,
        prompt=prompt,
        startImage=payload.get("startImage"),
        inputImages=payload.get("inputImages"),
        audioUrl=payload.get("audioUrl"),
        audioBase64=payload.get("audioBase64"),
        duration=payload.get("duration"),
        aspectRatio=payload.get("aspectRatio"),
        negativePrompt=payload.get("negativePrompt"),
        generateAudio=payload.get("generateAudio"),
        parameters=payload.get("parameters"),
    )


def _normalize_speech_input(payload: Any) -> SpeechGenerateInput:
    if not isinstance(payload, dict):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be a SpeechGenerateInput object",
            )
        )
    input_data = dict(payload)
    if "text" not in input_data and isinstance(input_data.get("input"), str):
        input_data["text"] = input_data.get("input")
    input_data.pop("input", None)
    if "responseFormat" not in input_data and isinstance(input_data.get("response_format"), str):
        input_data["responseFormat"] = input_data.get("response_format")
    input_data.pop("response_format", None)
    if "responseFormat" not in input_data and isinstance(input_data.get("format"), str):
        input_data["responseFormat"] = input_data.get("format")
    provider = input_data.get("provider")
    if not isinstance(provider, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    model = input_data.get("model")
    if not isinstance(model, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    text = input_data.get("text")
    if not isinstance(text, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="text is required and must be a string")
        )
    return SpeechGenerateInput(**input_data)


def _normalize_transcribe_input(payload: Any) -> TranscribeInput:
    if not isinstance(payload, dict):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be a TranscribeInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    audio = payload.get("audio")
    if not isinstance(provider, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(audio, dict):
        raise AiKitError(
            KitErrorPayload(kind=ErrorKind.VALIDATION, message="audio is required and must be an object")
        )
    response_format = payload.get("responseFormat")
    if response_format is None:
        response_format = payload.get("response_format")
    timestamp_granularities = payload.get("timestampGranularities")
    if timestamp_granularities is None:
        timestamp_granularities = payload.get("timestamp_granularities")
    if isinstance(timestamp_granularities, str):
        timestamp_granularities = [timestamp_granularities]
    return TranscribeInput(
        provider=provider,
        model=model,
        audio=audio,
        language=payload.get("language"),
        prompt=payload.get("prompt"),
        temperature=payload.get("temperature"),
        responseFormat=response_format,
        timestampGranularities=timestamp_granularities,
        metadata=payload.get("metadata"),
    )


async def _respond_json(send, status: int, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _respond_text(send, status: int, text: str) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": text.encode("utf-8")})


async def _start_sse(send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/event-stream"),
                (b"cache-control", b"no-cache"),
                (b"connection", b"keep-alive"),
            ],
        }
    )


async def _send_sse_event(send, event: str, data: Any, more_body: bool) -> None:
    payload = f"event: {event}\n" + f"data: {json.dumps(data)}\n\n"
    await send(
        {
            "type": "http.response.body",
            "body": payload.encode("utf-8"),
            "more_body": more_body,
        }
    )


async def _send_json_error(send, err: Exception) -> None:
    kit_err = to_kit_error(err)
    status = _map_status(kit_err)
    payload = {"error": {"kind": kit_err.kind.value, "message": kit_err.message}}
    await _respond_json(send, status, payload)


def _map_status(err: AiKitError) -> int:
    if err.kind in (ErrorKind.VALIDATION, ErrorKind.UNSUPPORTED):
        return 400
    if err.kind == ErrorKind.PROVIDER_AUTH:
        return 401
    if err.kind == ErrorKind.PROVIDER_RATE_LIMIT:
        return 429
    if err.kind == ErrorKind.PROVIDER_UNAVAILABLE:
        return 503
    return 500
