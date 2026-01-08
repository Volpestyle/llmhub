from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional


Provider = str


@dataclass
class ModelCapabilities:
    text: bool
    vision: bool
    image: bool
    tool_use: bool
    structured_output: bool
    reasoning: bool
    audio_in: Optional[bool] = None
    audio_out: Optional[bool] = None
    video: Optional[bool] = None
    video_in: Optional[bool] = None


@dataclass
class TokenPrices:
    input: Optional[float] = None
    output: Optional[float] = None


@dataclass
class ModelMetadata:
    id: str
    displayName: str
    provider: Provider
    capabilities: ModelCapabilities
    family: Optional[str] = None
    contextWindow: Optional[int] = None
    tokenPrices: Optional[TokenPrices] = None
    videoPrices: Optional[Dict[str, float]] = None
    deprecated: Optional[bool] = None
    inPreview: Optional[bool] = None
    inputs: Optional[List[Dict[str, Any]]] = None


@dataclass
class EntitlementContext:
    provider: Optional[Provider] = None
    apiKey: Optional[str] = None
    apiKeyFingerprint: Optional[str] = None
    accountId: Optional[str] = None
    region: Optional[str] = None
    environment: Optional[str] = None
    tenantId: Optional[str] = None
    userId: Optional[str] = None


@dataclass
class ModelModalities:
    text: bool
    vision: Optional[bool] = None
    audioIn: Optional[bool] = None
    audioOut: Optional[bool] = None
    imageOut: Optional[bool] = None
    videoIn: Optional[bool] = None
    videoOut: Optional[bool] = None


@dataclass
class ModelFeatures:
    tools: Optional[bool] = None
    jsonMode: Optional[bool] = None
    jsonSchema: Optional[bool] = None
    streaming: Optional[bool] = None
    batch: Optional[bool] = None


@dataclass
class ModelLimits:
    contextTokens: Optional[int] = None
    maxOutputTokens: Optional[int] = None


@dataclass
class ModelPricing:
    currency: str = "USD"
    inputPer1M: Optional[float] = None
    cachedInputPer1M: Optional[float] = None
    outputPer1M: Optional[float] = None
    extras: Optional[Dict[str, float]] = None
    effectiveAsOf: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ModelAvailability:
    entitled: bool
    lastVerifiedAt: Optional[str] = None
    confidence: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ModelRecord:
    id: str
    provider: Provider
    providerModelId: str
    displayName: Optional[str]
    modalities: ModelModalities
    features: ModelFeatures
    limits: Optional[ModelLimits]
    tags: Optional[List[str]]
    pricing: Optional[ModelPricing]
    availability: ModelAvailability


@dataclass
class ModelConstraints:
    requireTools: Optional[bool] = None
    requireJson: Optional[bool] = None
    requireVision: Optional[bool] = None
    requireVideo: Optional[bool] = None
    maxCostUsd: Optional[float] = None
    latencyClass: Optional[str] = None
    allowPreview: Optional[bool] = None


@dataclass
class ModelResolutionRequest:
    constraints: Optional[ModelConstraints] = None
    preferredModels: Optional[List[str]] = None


@dataclass
class ResolvedModel:
    primary: ModelRecord
    fallback: Optional[List[ModelRecord]] = None


@dataclass
class ContentPart:
    type: str
    text: Optional[str] = None
    image: Optional[Dict[str, Any]] = None


@dataclass
class ImageInput:
    url: Optional[str] = None
    base64: Optional[str] = None
    mediaType: Optional[str] = None


@dataclass
class AudioInput:
    url: Optional[str] = None
    base64: Optional[str] = None
    mediaType: Optional[str] = None
    fileName: Optional[str] = None
    path: Optional[str] = None


@dataclass
class Message:
    role: str
    content: List[ContentPart]
    toolCallId: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ToolDefinition:
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ToolChoice:
    type: str
    name: Optional[str] = None


@dataclass
class JsonSchemaFormat:
    name: str
    schema: Dict[str, Any]
    strict: Optional[bool] = None


@dataclass
class ResponseFormat:
    type: str
    jsonSchema: Optional[JsonSchemaFormat] = None


@dataclass
class GenerateInput:
    provider: Provider
    model: str
    messages: List[Message]
    tools: Optional[List[ToolDefinition]] = None
    toolChoice: Optional[ToolChoice] = None
    responseFormat: Optional[ResponseFormat] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    maxTokens: Optional[int] = None
    stream: Optional[bool] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class ImageGenerateInput:
    provider: Provider
    model: str
    prompt: str
    size: Optional[str] = None
    inputImages: Optional[List[ImageInput]] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ImageGenerateOutput:
    mime: str
    data: str
    images: Optional[List[Dict[str, str]]] = None
    raw: Optional[Any] = None


@dataclass
class MeshGenerateInput:
    provider: Provider
    model: str
    prompt: str
    inputImages: Optional[List[ImageInput]] = None
    format: Optional[str] = None


@dataclass
class MeshGenerateOutput:
    data: str
    format: Optional[str] = None
    raw: Optional[Any] = None


@dataclass
class LipsyncGenerateInput:
    provider: Provider
    model: str
    videoUrl: Optional[str] = None
    videoBase64: Optional[str] = None
    audioUrl: Optional[str] = None
    audioBase64: Optional[str] = None
    text: Optional[str] = None
    voiceId: Optional[str] = None
    voiceSpeed: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class LipsyncGenerateOutput:
    mime: str
    data: str
    raw: Optional[Any] = None


@dataclass
class SpeechGenerateInput:
    provider: Provider
    model: str
    text: str
    voice: Optional[str] = None
    responseFormat: Optional[str] = None
    format: Optional[str] = None
    speed: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class SpeechGenerateOutput:
    mime: str
    data: str
    raw: Optional[Any] = None


@dataclass
class VideoGenerateInput:
    provider: Provider
    model: str
    prompt: str
    startImage: Optional[str] = None
    inputImages: Optional[List[ImageInput]] = None
    audioUrl: Optional[str] = None
    audioBase64: Optional[str] = None
    duration: Optional[float] = None
    aspectRatio: Optional[str] = None
    negativePrompt: Optional[str] = None
    generateAudio: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class VideoGenerateOutput:
    mime: str
    data: str
    duration: Optional[float] = None
    raw: Optional[Any] = None


@dataclass
class VoiceAgentAudioConfig:
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None


@dataclass
class VoiceAgentInput:
    provider: Provider
    model: str
    instructions: Optional[str] = None
    voice: Optional[str] = None
    userText: Optional[str] = None
    audio: Optional[VoiceAgentAudioConfig] = None
    turnDetection: Optional[str] = None
    tools: Optional[List["ToolDefinition"]] = None
    responseModalities: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None
    timeoutMs: Optional[int] = None
    toolHandler: Optional[Callable[["ToolCall"], Any]] = None


@dataclass
class VoiceAgentOutput:
    transcript: Optional[str] = None
    audio: Optional[SpeechGenerateOutput] = None
    toolCalls: Optional[List["ToolCall"]] = None
    raw: Optional[Any] = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptWord:
    start: float
    end: float
    word: str


@dataclass
class TranscribeInput:
    provider: Provider
    model: str
    audio: AudioInput
    language: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    responseFormat: Optional[str] = None
    timestampGranularities: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class TranscribeOutput:
    text: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[TranscriptSegment]] = None
    words: Optional[List[TranscriptWord]] = None
    raw: Optional[Any] = None
    cost: Optional[CostBreakdown] = None


@dataclass
class ToolCall:
    id: str
    name: str
    argumentsJson: str


@dataclass
class Usage:
    inputTokens: Optional[int] = None
    outputTokens: Optional[int] = None
    totalTokens: Optional[int] = None


@dataclass
class CostBreakdown:
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    pricing_per_million: Optional[TokenPrices] = None


@dataclass
class GenerateOutput:
    text: Optional[str] = None
    toolCalls: Optional[List[ToolCall]] = None
    usage: Optional[Usage] = None
    finishReason: Optional[str] = None
    cost: Optional[CostBreakdown] = None
    raw: Optional[Any] = None


@dataclass
class StreamChunk:
    type: str
    textDelta: Optional[str] = None
    call: Optional[ToolCall] = None
    delta: Optional[str] = None
    usage: Optional[Usage] = None
    finishReason: Optional[str] = None
    cost: Optional[CostBreakdown] = None
    error: Optional[Dict[str, Any]] = None


def as_json_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        data = asdict(obj)
        return {k: v for k, v in data.items() if v is not None}
    if isinstance(obj, list):
        return [as_json_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: as_json_dict(v) for k, v in obj.items() if v is not None}
    return obj


def ensure_messages(messages: Iterable[Message | Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for msg in messages:
        if isinstance(msg, Message):
            normalized.append(as_json_dict(msg))
        else:
            normalized.append(msg)
    return normalized
