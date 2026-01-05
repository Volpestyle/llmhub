from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from websocket import WebSocketTimeoutException, create_connection

from .openai import OpenAIAdapter, OpenAIConfig
from ..errors import ErrorKind, KitErrorPayload, AiKitError
from ..types import ImageGenerateInput, ImageGenerateOutput, SpeechGenerateInput, SpeechGenerateOutput
from ..types import Provider

_XAI_SPEECH_PROTOCOLS = ["realtime", "openai-beta.realtime-v1"]
_XAI_DEFAULT_VOICE = "ara"
_XAI_DEFAULT_SAMPLE_RATE = 24000

# xAI Voice Catalog - https://docs.x.ai/docs/guides/voice
XAI_VOICES = {
    "ara": {"type": "female", "tone": "warm, friendly", "description": "Default voice, balanced and conversational"},
    "rex": {"type": "male", "tone": "confident, clear", "description": "Professional, ideal for business"},
    "sal": {"type": "neutral", "tone": "smooth, balanced", "description": "Versatile for various contexts"},
    "eve": {"type": "female", "tone": "energetic, upbeat", "description": "Engaging, great for interactive experiences"},
    "leo": {"type": "male", "tone": "authoritative, strong", "description": "Decisive, suitable for instructional content"},
}


@dataclass
class XAIConfig:
    api_key: str = ""
    api_keys: Optional[List[str]] = None
    base_url: str = "https://api.x.ai"
    compatibility_mode: str = "openai"
    speech_mode: str = "realtime"
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
        self.speech_mode = (config.speech_mode or "realtime").lower()
        self._xai_base_url = config.base_url

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="xAI image generation is not supported",
                provider=self.provider,
            )
        )

    def generate_mesh(self, input: "MeshGenerateInput"):
        raise AiKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="xAI mesh generation is not supported",
                provider=self.provider,
            )
        )

    def generate_speech(self, input: SpeechGenerateInput) -> SpeechGenerateOutput:
        speech_mode = self._resolve_speech_mode(input.metadata)
        if speech_mode == "openai":
            return super().generate_speech(input)
        return self._generate_speech_realtime(input)

    def _generate_speech_realtime(self, input: SpeechGenerateInput) -> SpeechGenerateOutput:
        if not self.config.api_key:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.PROVIDER_AUTH,
                    message="xAI api key is required for realtime speech",
                    provider=self.provider,
                )
            )
        format_type, sample_rate, mime, session_overrides, response_overrides = _resolve_speech_options(input)
        session: Dict[str, Any] = {
            "voice": input.voice or _XAI_DEFAULT_VOICE,
            "turn_detection": {"type": None},
            "audio": {
                "output": {
                    "format": {
                        "type": format_type,
                        "rate": sample_rate,
                    }
                }
            },
        }
        if session_overrides:
            session.update(session_overrides)
        response: Dict[str, Any] = {"modalities": ["audio"]}
        if response_overrides:
            response.update(response_overrides)
        url = _resolve_realtime_url(self._xai_base_url)
        headers = [f"Authorization: Bearer {self.config.api_key}"]
        ws = None
        audio_chunks: List[bytes] = []
        session_sent = False
        response_sent = False
        try:
            ws = create_connection(
                url,
                header=headers,
                subprotocols=_XAI_SPEECH_PROTOCOLS,
                timeout=self.config.timeout,
            )
            while True:
                raw = ws.recv()
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "conversation.created" and not session_sent:
                    session_sent = True
                    ws.send(json.dumps({"type": "session.update", "session": session}))
                    continue
                if event_type == "session.updated" and not response_sent:
                    response_sent = True
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": input.text}
                                    ],
                                },
                            }
                        )
                    )
                    ws.send(json.dumps({"type": "response.create", "response": response}))
                    continue
                if event_type == "response.output_audio.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        audio_chunks.append(base64.b64decode(delta))
                    continue
                if event_type in ("response.output_audio.done", "response.done"):
                    payload = b"".join(audio_chunks)
                    return SpeechGenerateOutput(
                        mime=mime,
                        data=base64.b64encode(payload).decode("utf-8"),
                    )
                if event_type in ("error", "response.error"):
                    message = "xAI realtime error"
                    error_payload = event.get("error")
                    if isinstance(error_payload, dict):
                        value = error_payload.get("message")
                        if isinstance(value, str) and value:
                            message = value
                    raise AiKitError(
                        KitErrorPayload(
                            kind=ErrorKind.UNKNOWN,
                            message=message,
                            provider=self.provider,
                        )
                    )
        except WebSocketTimeoutException as exc:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.TIMEOUT,
                    message="xAI realtime speech request timed out",
                    provider=self.provider,
                    cause=exc,
                )
            ) from exc
        finally:
            if ws is not None:
                ws.close()

    def _resolve_speech_mode(self, metadata: Optional[Dict[str, str]]) -> str:
        if metadata:
            override = metadata.get("xai:speech-mode")
            if isinstance(override, str) and override.strip():
                return override.lower()
        return self.speech_mode


def _resolve_realtime_url(base_url: str) -> str:
    parsed = urlparse(base_url or "https://api.x.ai")
    scheme = "wss" if parsed.scheme != "http" else "ws"
    netloc = parsed.netloc or parsed.path
    return urlunparse((scheme, netloc, "/v1/realtime", "", "", ""))


def _resolve_speech_options(
    input: SpeechGenerateInput,
) -> tuple[str, int, str, Dict[str, Any], Dict[str, Any]]:
    response_format = (input.responseFormat or input.format or "").strip().lower()
    format_type = "audio/pcm"
    mime = "audio/pcm"
    if response_format and response_format != "pcm":
        if response_format == "pcmu":
            format_type = "audio/pcmu"
            mime = "audio/pcmu"
        elif response_format == "pcma":
            format_type = "audio/pcma"
            mime = "audio/pcma"
        else:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"xAI realtime speech only supports pcm/pcmu/pcma output (received {response_format})",
                    provider="xai",
                )
            )
    params = input.parameters if isinstance(input.parameters, dict) else {}
    sample_rate = _XAI_DEFAULT_SAMPLE_RATE
    if format_type != "audio/pcm":
        sample_rate = 8000
    else:
        raw_rate = params.get("sampleRate")
        if isinstance(raw_rate, (int, float)) and raw_rate > 0:
            sample_rate = int(raw_rate)
    session_overrides = params.get("session") if isinstance(params.get("session"), dict) else {}
    response_overrides = params.get("response") if isinstance(params.get("response"), dict) else {}
    return format_type, sample_rate, mime, session_overrides, response_overrides
