from importlib import import_module
from pkgutil import extend_path

from .allowlists import list_task_models, list_transcribe_models
from .catalog import load_catalog_models
from .errors import AiKitError, ErrorKind
from .registry import ModelRegistry
from .types import (
    CostBreakdown,
    EntitlementContext,
    GenerateInput,
    GenerateOutput,
    ImageGenerateInput,
    ImageGenerateOutput,
    Message,
    MeshGenerateInput,
    MeshGenerateOutput,
    AudioInput,
    TranscribeInput,
    TranscribeOutput,
    TranscriptSegment,
    TranscriptWord,
    ModelConstraints,
    ModelMetadata,
    ModelRecord,
    ModelResolutionRequest,
    ResolvedModel,
    StreamChunk,
    ToolDefinition,
    ToolChoice,
    Usage,
)

__path__ = extend_path(__path__, __name__)

__all__ = [
    "list_task_models",
    "list_transcribe_models",
    "load_catalog_models",
    "ModelRegistry",
    "CostBreakdown",
    "EntitlementContext",
    "GenerateInput",
    "GenerateOutput",
    "ImageGenerateInput",
    "ImageGenerateOutput",
    "Message",
    "MeshGenerateInput",
    "MeshGenerateOutput",
    "AudioInput",
    "TranscribeInput",
    "TranscribeOutput",
    "TranscriptSegment",
    "TranscriptWord",
    "ModelConstraints",
    "ModelMetadata",
    "ModelRecord",
    "ModelResolutionRequest",
    "ResolvedModel",
    "StreamChunk",
    "ToolDefinition",
    "ToolChoice",
    "Usage",
    "AiKitError",
    "ErrorKind",
]


def _optional(module_path: str, names: list[str]) -> None:
    try:
        module = import_module(module_path, __name__)
    except Exception:
        return
    for name in names:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)
            __all__.append(name)


_optional(".hub", ["Kit", "KitConfig"])
_optional(".router", ["ModelRouter"])
_optional(".kit_cache", ["get_cached_kit", "list_provider_models"])
_optional(
    ".providers",
    [
        "OpenAIAdapter",
        "OpenAIConfig",
        "AnthropicAdapter",
        "AnthropicConfig",
        "GeminiAdapter",
        "GeminiConfig",
        "XAIAdapter",
        "XAIConfig",
        "OllamaAdapter",
        "OllamaConfig",
        "BedrockAdapter",
        "BedrockConfig",
    ],
)
_optional(".http_asgi", ["create_asgi_app"])
_optional(
    ".testing",
    [
        "FixtureAdapter",
        "FixtureEntry",
        "FixtureKeyInput",
        "build_stream_chunks",
        "fixture_key",
    ],
)
_optional(".clients", ["MeshyClient", "MeshyError", "MeshyTask", "ReplicateClient", "FalClient"])
