from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qs

from .errors import ErrorKind, HubErrorPayload, InferenceKitError, to_hub_error
from .hub import Hub
from .types import GenerateInput, ImageGenerateInput, MeshGenerateInput, as_json_dict

ASGIApp = Callable[[Dict[str, Any], Callable[..., Awaitable[Dict[str, Any]]], Callable[..., Awaitable[None]]], Awaitable[None]]


def create_asgi_app(hub: Hub, base_path: str = "") -> ASGIApp:
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
            await _handle_models(hub, scope, send)
            return
        if path == "/generate":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_generate(hub, receive, send)
            return
        if path == "/image":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_image(hub, receive, send)
            return
        if path == "/mesh":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_mesh(hub, receive, send)
            return
        if path == "/generate/stream":
            if method != "POST":
                await _respond_text(send, 405, "method not allowed")
                return
            await _handle_stream(hub, receive, send)
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


async def _handle_models(hub: Hub, scope: Dict[str, Any], send) -> None:
    try:
        query = _parse_query(scope)
        providers = _parse_providers(_first_query_value(query, "providers"))
        refresh = _should_refresh(query.get("refresh", []))
        models = hub.list_models(providers=providers, refresh=refresh)
        await _respond_json(send, 200, as_json_dict(models))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_generate(hub: Hub, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_generate_input(payload)
        output = hub.generate(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_image(hub: Hub, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_image_input(payload)
        output = hub.generate_image(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_mesh(hub: Hub, receive, send) -> None:
    try:
        payload = await _read_json(receive)
        input_data = _normalize_mesh_input(payload)
        output = hub.generate_mesh(input_data)
        await _respond_json(send, 200, as_json_dict(output))
    except Exception as err:
        await _send_json_error(send, err)


async def _handle_stream(hub: Hub, receive, send) -> None:
    started = False
    try:
        payload = await _read_json(receive)
        input_data = _normalize_generate_input(payload, force_stream=True)
        await _start_sse(send)
        started = True
        for chunk in hub.stream_generate(input_data):
            await _send_sse_event(send, "chunk", as_json_dict(chunk), more_body=True)
        await _send_sse_event(send, "done", {"ok": True}, more_body=False)
    except Exception as err:
        hub_err = to_hub_error(err)
        if not started:
            await _send_json_error(send, hub_err)
            return
        await _send_sse_event(
            send,
            "error",
            {
                "kind": hub_err.kind.value,
                "message": hub_err.message,
                "requestId": hub_err.requestId,
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
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="Body must be valid JSON")
        )
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="Body must be valid JSON")
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
        raise InferenceKitError(
            HubErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be a GenerateInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    messages = payload.get("messages")
    if not isinstance(provider, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(messages, list):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="messages is required and must be an array")
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
        raise InferenceKitError(
            HubErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be an ImageGenerateInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not isinstance(provider, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(prompt, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="prompt is required and must be a string")
        )
    return ImageGenerateInput(
        provider=provider,
        model=model,
        prompt=prompt,
        size=payload.get("size"),
        inputImages=payload.get("inputImages"),
    )


def _normalize_mesh_input(payload: Any) -> MeshGenerateInput:
    if not isinstance(payload, dict):
        raise InferenceKitError(
            HubErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Request body must be a MeshGenerateInput object",
            )
        )
    provider = payload.get("provider")
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not isinstance(provider, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="provider is required and must be a string")
        )
    if not isinstance(model, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="model is required and must be a string")
        )
    if not isinstance(prompt, str):
        raise InferenceKitError(
            HubErrorPayload(kind=ErrorKind.VALIDATION, message="prompt is required and must be a string")
        )
    return MeshGenerateInput(
        provider=provider,
        model=model,
        prompt=prompt,
        inputImages=payload.get("inputImages"),
        format=payload.get("format"),
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
    hub_err = to_hub_error(err)
    status = _map_status(hub_err)
    payload = {"error": {"kind": hub_err.kind.value, "message": hub_err.message}}
    await _respond_json(send, status, payload)


def _map_status(err: InferenceKitError) -> int:
    if err.kind in (ErrorKind.VALIDATION, ErrorKind.UNSUPPORTED):
        return 400
    if err.kind == ErrorKind.PROVIDER_AUTH:
        return 401
    if err.kind == ErrorKind.PROVIDER_RATE_LIMIT:
        return 429
    if err.kind == ErrorKind.PROVIDER_UNAVAILABLE:
        return 503
    return 500
