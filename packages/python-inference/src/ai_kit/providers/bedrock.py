from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote, urlparse, parse_qsl

import requests

from ..errors import ErrorKind, KitErrorPayload, AiKitError, classify_status
from ..types import (
    GenerateInput,
    GenerateOutput,
    ImageGenerateInput,
    ImageGenerateOutput,
    MeshGenerateInput,
    MeshGenerateOutput,
    ModelCapabilities,
    ModelMetadata,
    Provider,
    StreamChunk,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    TranscribeInput,
    TranscribeOutput,
    Usage,
    ensure_messages,
)


@dataclass
class BedrockConfig:
    region: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: Optional[str] = None
    endpoint: str = ""
    runtime_endpoint: str = ""
    control_plane_service: str = "bedrock"
    runtime_service: str = "bedrock-runtime"
    timeout: Optional[float] = None


class BedrockAdapter:
    def __init__(self, config: BedrockConfig, provider: Provider = "bedrock") -> None:
        self.config = _resolve_config(config)
        self.provider = provider

    def list_models(self) -> List[ModelMetadata]:
        url = f"{self.config.endpoint.rstrip('/')}/foundation-models"
        payload = self._request_json("GET", url, service=self.config.control_plane_service)
        summaries = payload.get("modelSummaries") or payload.get("models") or []
        models: List[ModelMetadata] = []
        for summary in summaries:
            model_id = (summary.get("modelId") or "").strip()
            if not model_id:
                continue
            input_modalities = {str(item).lower() for item in summary.get("inputModalities", [])}
            output_modalities = {str(item).lower() for item in summary.get("outputModalities", [])}
            models.append(
                ModelMetadata(
                    id=model_id,
                    displayName=summary.get("modelName") or model_id,
                    provider=self.provider,
                    family=_derive_family(model_id, summary.get("providerName")),
                    capabilities=ModelCapabilities(
                        text="text" in output_modalities,
                        vision="image" in input_modalities,
                        image="image" in output_modalities,
                        tool_use=_supports_tool_use(model_id),
                        structured_output=False,
                        reasoning=False,
                    ),
                )
            )
        return models

    def generate(self, input: GenerateInput) -> GenerateOutput:
        if input.responseFormat and input.responseFormat.type == "json_schema":
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message="Bedrock structured outputs are not supported",
                    provider=self.provider,
                )
            )
        payload = self._build_payload(input)
        url = f"{self.config.runtime_endpoint.rstrip('/')}/model/{quote(input.model, safe='')}/converse"
        response = self._request_json(
            "POST",
            url,
            json_body=payload,
            service=self.config.runtime_service,
        )
        return _normalize_converse_response(response)

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Bedrock image generation is not supported",
                provider=self.provider,
            )
        )

    def generate_mesh(self, input: MeshGenerateInput) -> MeshGenerateOutput:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Bedrock mesh generation is not supported",
                provider=self.provider,
            )
        )

    def transcribe(self, input: TranscribeInput) -> TranscribeOutput:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Bedrock transcription is not supported",
                provider=self.provider,
            )
        )

    def stream_generate(self, input: GenerateInput) -> Iterable[StreamChunk]:
        output = self.generate(input)
        if output.text:
            yield StreamChunk(type="delta", textDelta=output.text)
        if output.toolCalls:
            for call in output.toolCalls:
                yield StreamChunk(type="tool_call", call=call)
        yield StreamChunk(
            type="message_end",
            usage=output.usage,
            finishReason=output.finishReason,
        )

    def _build_payload(self, input: GenerateInput) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        system: List[Dict[str, str]] = []
        for message in ensure_messages(input.messages):
            role = str(message.get("role") or "")
            content = message.get("content") or []
            if role == "system":
                text = _collect_text(content)
                if text:
                    system.append({"text": text})
                continue
            if role == "tool":
                tool_content = [
                    {"text": part.get("text") or ""}
                    for part in content
                    if part.get("type") == "text"
                ]
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message.get("toolCallId") or "",
                                    "content": tool_content,
                                }
                            }
                        ],
                    }
                )
                continue
            blocks: List[Dict[str, Any]] = []
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    blocks.append({"text": part.get("text") or ""})
                elif part_type == "tool_use":
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": part.get("id") or "",
                                "name": part.get("name") or "",
                                "input": part.get("input"),
                            }
                        }
                    )
                elif part_type == "image":
                    image = part.get("image") or {}
                    blocks.append({"image": self._resolve_image(image)})
            messages.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": blocks,
                }
            )
        payload: Dict[str, Any] = {"messages": messages}
        if system:
            payload["system"] = system
        inference_config: Dict[str, Any] = {}
        if input.maxTokens is not None:
            inference_config["maxTokens"] = input.maxTokens
        if input.temperature is not None:
            inference_config["temperature"] = input.temperature
        if input.topP is not None:
            inference_config["topP"] = input.topP
        if inference_config:
            payload["inferenceConfig"] = inference_config
        if input.tools:
            tool_config = _build_tool_config(input.tools, input.toolChoice)
            if tool_config:
                payload["toolConfig"] = tool_config
        return payload

    def _resolve_image(self, image: Dict[str, Any]) -> Dict[str, Any]:
        base64_data = (image.get("base64") or "").strip()
        media_type = (image.get("mediaType") or "").strip()
        url = (image.get("url") or "").strip()
        if not base64_data and url:
            base64_data, media_type = _fetch_image(url)
        if not base64_data:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message="Bedrock image content requires a base64 payload or URL",
                    provider=self.provider,
                )
            )
        return {
            "format": _image_format(media_type, url),
            "source": {"bytes": base64_data},
        }

    def _request_json(
        self,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]] = None,
        service: str = "bedrock",
    ) -> Dict[str, Any]:
        body = json.dumps(json_body) if json_body is not None else ""
        headers = _sign_aws_request(
            method=method,
            url=url,
            body=body,
            region=self.config.region,
            service=service,
            access_key=self.config.access_key_id,
            secret_key=self.config.secret_access_key,
            session_token=self.config.session_token,
            headers={
                "content-type": "application/json",
                "accept": "application/json",
            },
        )
        response = requests.request(
            method,
            url,
            headers=headers,
            data=body if method != "GET" else None,
            timeout=self.config.timeout,
        )
        if response.status_code >= 400:
            message, code = _extract_bedrock_error(response.text or "")
            raise AiKitError(
                KitErrorPayload(
                    kind=classify_status(response.status_code),
                    message=message or f"Bedrock error ({response.status_code})",
                    provider=self.provider,
                    upstreamStatus=response.status_code,
                    upstreamCode=code,
                    requestId=response.headers.get("x-amzn-requestid"),
                )
            )
        return response.json()


def _resolve_config(config: BedrockConfig) -> BedrockConfig:
    region = config.region.strip() or os.getenv("AWS_REGION", "") or os.getenv("AWS_DEFAULT_REGION", "")
    if not region:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Bedrock region is required",
                provider="bedrock",
            )
        )
    access_key = config.access_key_id.strip() or os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = config.secret_access_key.strip() or os.getenv("AWS_SECRET_ACCESS_KEY", "")
    session_token = (config.session_token or "").strip() or os.getenv("AWS_SESSION_TOKEN", "")
    if not access_key or not secret_key:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message="Bedrock AWS credentials are required",
                provider="bedrock",
            )
        )
    endpoint = config.endpoint.strip() or f"https://bedrock.{region}.amazonaws.com"
    runtime_endpoint = config.runtime_endpoint.strip() or f"https://bedrock-runtime.{region}.amazonaws.com"
    return BedrockConfig(
        region=region,
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token=session_token or None,
        endpoint=endpoint,
        runtime_endpoint=runtime_endpoint,
        control_plane_service=config.control_plane_service or "bedrock",
        runtime_service=config.runtime_service or "bedrock-runtime",
        timeout=config.timeout,
    )


def _build_tool_config(tools: List[ToolDefinition], choice: Optional[ToolChoice]) -> Dict[str, Any]:
    if not tools:
        return {}
    tool_specs = []
    for tool in tools:
        tool_specs.append(
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.parameters},
                }
            }
        )
    config: Dict[str, Any] = {"tools": tool_specs}
    if choice:
        config["toolChoice"] = _normalize_tool_choice(choice)
    return config


def _normalize_tool_choice(choice: ToolChoice) -> Dict[str, Any]:
    if choice.type == "auto":
        return {"auto": {}}
    if choice.type == "none":
        return {"none": {}}
    if choice.type == "tool":
        if choice.name:
            return {"tool": {"name": choice.name}}
        return {"auto": {}}
    return {"auto": {}}


def _normalize_converse_response(response: Dict[str, Any]) -> GenerateOutput:
    blocks = response.get("output", {}).get("message", {}).get("content", []) or []
    text = "".join([block.get("text", "") for block in blocks if "text" in block])
    tool_calls: List[ToolCall] = []
    for block in blocks:
        tool_use = block.get("toolUse")
        if not isinstance(tool_use, dict):
            continue
        args = json.dumps(tool_use.get("input") or {})
        tool_calls.append(
            ToolCall(
                id=tool_use.get("toolUseId") or "",
                name=tool_use.get("name") or "",
                argumentsJson=args,
            )
        )
    usage_payload = response.get("usage") or {}
    usage = None
    if usage_payload:
        usage = Usage(
            inputTokens=usage_payload.get("inputTokens"),
            outputTokens=usage_payload.get("outputTokens"),
            totalTokens=usage_payload.get("totalTokens"),
        )
    finish_reason = response.get("stopReason") or ("tool_calls" if tool_calls else "stop")
    return GenerateOutput(
        text=text or None,
        toolCalls=tool_calls or None,
        usage=usage,
        finishReason=finish_reason,
        raw=response,
    )


def _collect_text(parts: Iterable[Dict[str, Any]]) -> str:
    return "".join([part.get("text") or "" for part in parts if part.get("type") == "text"])


def _image_format(media_type: str, url_value: str) -> str:
    normalized = (media_type or "").lower()
    if "png" in normalized:
        return "png"
    if "jpeg" in normalized or "jpg" in normalized:
        return "jpeg"
    if "webp" in normalized:
        return "webp"
    if "gif" in normalized:
        return "gif"
    lower_url = (url_value or "").lower()
    if lower_url.endswith(".png"):
        return "png"
    if lower_url.endswith(".jpg") or lower_url.endswith(".jpeg"):
        return "jpeg"
    if lower_url.endswith(".webp"):
        return "webp"
    if lower_url.endswith(".gif"):
        return "gif"
    return "png"


def _fetch_image(url: str) -> tuple[str, str]:
    response = requests.get(url, timeout=20)
    if response.status_code >= 400:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.VALIDATION,
                message=f"Failed to load image URL ({response.status_code})",
                provider="bedrock",
                upstreamStatus=response.status_code,
            )
        )
    media_type = response.headers.get("content-type") or ""
    encoded = base64.b64encode(response.content).decode("utf-8")
    return encoded, media_type


def _derive_family(model_id: str, provider_name: Optional[str]) -> str:
    if provider_name:
        return provider_name.lower()
    return model_id.split(".", 1)[0] if model_id else model_id


def _supports_tool_use(model_id: str) -> bool:
    return model_id.startswith("anthropic.") or model_id.startswith("cohere.")


def _sign_aws_request(
    method: str,
    url: str,
    body: str,
    region: str,
    service: str,
    access_key: str,
    secret_key: str,
    session_token: Optional[str],
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    parsed = urlparse(url)
    now = datetime.utcnow()
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    signed_headers = {
        "host": parsed.netloc,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }
    if session_token:
        signed_headers["x-amz-security-token"] = session_token
    if headers:
        signed_headers.update(headers)
    canonical_headers, signed_headers_list = _canonicalize_headers(signed_headers)
    canonical_request = "\n".join(
        [
            method.upper(),
            _canonical_uri(parsed.path),
            _canonical_query(parsed.query),
            canonical_headers,
            signed_headers_list,
            payload_hash,
        ]
    )
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    signed_headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers_list}, Signature={signature}"
    )
    return signed_headers


def _canonicalize_headers(headers: Dict[str, str]) -> tuple[str, str]:
    items = [(key.lower(), " ".join(value.split())) for key, value in headers.items()]
    items.sort()
    canonical = "".join([f"{key}:{value}\n" for key, value in items])
    signed = ";".join([key for key, _ in items])
    return canonical, signed


def _canonical_uri(path: str) -> str:
    if not path:
        return "/"
    return quote(path, safe="/-_.~")


def _canonical_query(query: str) -> str:
    if not query:
        return ""
    pairs = parse_qsl(query, keep_blank_values=True)
    pairs.sort()
    return "&".join([f"{_encode(k)}={_encode(v)}" for k, v in pairs])


def _encode(value: str) -> str:
    return quote(str(value), safe="-_.~")


def _get_signature_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    return _sign(k_service, "aws4_request")


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _extract_bedrock_error(body: str) -> tuple[str, Optional[str]]:
    if not body:
        return "", None
    try:
        parsed = json.loads(body)
        if isinstance(parsed.get("message"), str):
            return parsed.get("message", ""), parsed.get("code")
        if isinstance(parsed.get("error"), str):
            return parsed.get("error", ""), None
        if isinstance(parsed.get("error"), dict):
            return parsed.get("error", {}).get("message", ""), parsed.get("error", {}).get("code")
        if isinstance(parsed.get("__type"), str):
            return parsed.get("message", ""), parsed.get("__type")
    except Exception:
        return body, None
    return body, None
