from __future__ import annotations

import base64
import io
import json
import logging
import os
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from websocket import WebSocketTimeoutException, create_connection

from .openai import OpenAIAdapter, OpenAIConfig
from ..errors import ErrorKind, KitErrorPayload, AiKitError
from ..types import (
    ImageGenerateInput,
    ImageGenerateOutput,
    SpeechGenerateInput,
    SpeechGenerateOutput,
    ToolCall,
    VoiceAgentInput,
    VoiceAgentOutput,
)
from ..types import Provider

_XAI_SPEECH_PROTOCOLS = ["realtime", "openai-beta.realtime-v1"]
_XAI_DEFAULT_VOICE = "Ara"
_XAI_DEFAULT_SAMPLE_RATE = 24000

# xAI Voice Catalog - https://docs.x.ai/docs/guides/voice
XAI_VOICES = {
    "Ara": {"type": "female", "tone": "warm, friendly", "description": "Default voice, balanced and conversational"},
    "Rex": {"type": "male", "tone": "confident, clear", "description": "Professional, ideal for business"},
    "Sal": {"type": "neutral", "tone": "smooth, balanced", "description": "Versatile for various contexts"},
    "Eve": {"type": "female", "tone": "energetic, upbeat", "description": "Engaging, great for interactive experiences"},
    "Leo": {"type": "male", "tone": "authoritative, strong", "description": "Decisive, suitable for instructional content"},
}

logger = logging.getLogger(__name__)


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

    def generate_voice_agent(self, input: VoiceAgentInput) -> VoiceAgentOutput:
        if not self.config.api_key:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.PROVIDER_AUTH,
                    message="xAI api key is required for realtime voice agent",
                    provider=self.provider,
                )
            )
        if not input.userText:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message="Voice agent requires userText",
                    provider=self.provider,
                )
            )
        audio_config, mime = _resolve_voice_agent_audio(input)
        session: Dict[str, Any] = {
            "voice": input.voice or _XAI_DEFAULT_VOICE,
            "turn_detection": {"type": input.turnDetection},
            "audio": audio_config,
        }
        if input.instructions:
            session["instructions"] = input.instructions
        if input.tools:
            session["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in input.tools
            ]
        params = input.parameters if isinstance(input.parameters, dict) else {}
        session_overrides = params.get("session") if isinstance(params.get("session"), dict) else {}
        if session_overrides:
            session.update(session_overrides)
        response: Dict[str, Any] = {
            "modalities": input.responseModalities or ["audio", "text"],
        }
        response_overrides = params.get("response") if isinstance(params.get("response"), dict) else {}
        if response_overrides:
            response.update(response_overrides)

        url = _resolve_realtime_url(self._xai_base_url)
        headers = [f"Authorization: Bearer {self.config.api_key}"]
        ws = None
        audio_chunks: List[bytes] = []
        transcript_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        # Tool-call turns can emit response.done before the follow-up response with audio.
        pending_tool_response = False
        session_sent = False
        response_sent = False
        timeout = self.config.timeout
        if input.timeoutMs and input.timeoutMs > 0:
            timeout = input.timeoutMs / 1000.0
        debug_events = os.environ.get("AI_KIT_XAI_DEBUG_EVENTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        def _debug_event(event_type: Optional[str]) -> None:
            if not debug_events:
                return
            print(f"xai.realtime event={event_type or 'unknown'}", flush=True)
        try:
            ws = create_connection(
                url,
                header=headers,
                subprotocols=_XAI_SPEECH_PROTOCOLS,
                timeout=timeout,
            )
            while True:
                raw = ws.recv()
                event = json.loads(raw)
                event_type = event.get("type")
                _debug_event(event_type if isinstance(event_type, str) else None)
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
                                        {"type": "input_text", "text": input.userText}
                                    ],
                                },
                            }
                        )
                    )
                    ws.send(json.dumps({"type": "response.create", "response": response}))
                    continue
                if event_type == "response.function_call_arguments.done":
                    name = event.get("name") if isinstance(event.get("name"), str) else ""
                    call_id = event.get("call_id") if isinstance(event.get("call_id"), str) else ""
                    arguments = event.get("arguments") if isinstance(event.get("arguments"), str) else "{}"
                    call = ToolCall(id=call_id or f"call_{len(tool_calls) + 1}", name=name, argumentsJson=arguments)
                    tool_calls.append(call)
                    output = {"ok": True}
                    if callable(input.toolHandler):
                        try:
                            output = input.toolHandler(call)
                        except Exception as exc:
                            output = {"ok": False, "error": str(exc)}
                    output_json = output if isinstance(output, str) else json.dumps(output)
                    pending_tool_response = True
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call.id,
                                    "output": output_json,
                                },
                            }
                        )
                    )
                    ws.send(json.dumps({"type": "response.create"}))
                    continue
                if event_type == "response.created" and pending_tool_response:
                    pending_tool_response = False
                if event_type == "response.output_audio_transcript.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        transcript_parts.append(delta)
                    continue
                if event_type == "response.output_audio.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        audio_chunks.append(base64.b64decode(delta))
                    continue
                if event_type in ("response.output_audio.done", "response.done"):
                    if pending_tool_response:
                        continue
                    audio = None
                    if audio_chunks:
                        payload = b"".join(audio_chunks)
                        audio = SpeechGenerateOutput(
                            mime=mime,
                            data=base64.b64encode(payload).decode("utf-8"),
                        )
                    return VoiceAgentOutput(
                        transcript="".join(transcript_parts).strip() or None,
                        audio=audio,
                        toolCalls=tool_calls or None,
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
                    message="xAI realtime voice agent request timed out",
                    provider=self.provider,
                    cause=exc,
                )
            ) from exc
        finally:
            if ws is not None:
                ws.close()

    def _generate_speech_realtime(self, input: SpeechGenerateInput) -> SpeechGenerateOutput:
        if not self.config.api_key:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.PROVIDER_AUTH,
                    message="xAI api key is required for realtime speech",
                    provider=self.provider,
                )
            )
        format_type, sample_rate, mime, session_overrides, response_overrides, wrap_wav = _resolve_speech_options(input)
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
                    if wrap_wav:
                        payload = _encode_wav(payload, sample_rate)
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
) -> tuple[str, int, str, Dict[str, Any], Dict[str, Any], bool]:
    response_format = (input.responseFormat or input.format or "").strip().lower()
    wrap_wav = False
    if response_format in {"wav", "wave"}:
        wrap_wav = True
        response_format = "pcm"
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
    instructions = params.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        if "instructions" not in response_overrides:
            response_overrides = dict(response_overrides)
            response_overrides["instructions"] = instructions
    if wrap_wav:
        mime = "audio/wav"
    return format_type, sample_rate, mime, session_overrides, response_overrides, wrap_wav


def _encode_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    if not pcm_bytes:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()


def _resolve_voice_agent_audio(input: VoiceAgentInput) -> tuple[Dict[str, Any], str]:
    params = input.parameters if isinstance(input.parameters, dict) else {}
    output_format = None
    if input.audio and isinstance(input.audio.output, dict):
        output_format = input.audio.output.get("format")
    format_type = "audio/pcm"
    if isinstance(output_format, dict):
        raw_type = output_format.get("type")
        if isinstance(raw_type, str) and raw_type:
            format_type = raw_type
    sample_rate = _XAI_DEFAULT_SAMPLE_RATE
    if format_type != "audio/pcm":
        sample_rate = 8000
    else:
        raw_rate = None
        if isinstance(output_format, dict):
            raw_rate = output_format.get("rate")
        if raw_rate is None:
            raw_rate = params.get("sampleRate")
        if isinstance(raw_rate, (int, float)) and raw_rate > 0:
            sample_rate = int(raw_rate)
    input_format = None
    if input.audio and isinstance(input.audio.input, dict):
        input_format = input.audio.input.get("format")
    audio = {
        "input": {"format": input_format or {"type": format_type, "rate": sample_rate}},
        "output": {"format": output_format or {"type": format_type, "rate": sample_rate}},
    }
    return audio, format_type
