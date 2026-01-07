from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict

from .errors import ErrorKind, KitErrorPayload, AiKitError, to_kit_error
from .entitlements import fingerprint_api_key
from .pricing import estimate_cost, estimate_transcribe_cost
from .registry import ModelRegistry
from .types import (
    EntitlementContext,
    GenerateInput,
    GenerateOutput,
    ImageGenerateInput,
    ImageGenerateOutput,
    MeshGenerateInput,
    MeshGenerateOutput,
    SpeechGenerateInput,
    SpeechGenerateOutput,
    TranscribeInput,
    TranscribeOutput,
    Provider,
)
from .providers import (
    AnthropicAdapter,
    AnthropicConfig,
    GeminiAdapter,
    GeminiConfig,
    OpenAIAdapter,
    OpenAIConfig,
    XAIAdapter,
    XAIConfig,
    OllamaAdapter,
    OllamaConfig,
    BedrockAdapter,
    BedrockConfig,
    ReplicateAdapter,
    ReplicateConfig,
)


@dataclass
class KitConfig:
    providers: Dict[Provider, object]
    registry_ttl_seconds: int = 1800
    adapters: Dict[Provider, object] | None = None
    adapter_factory: Callable[[Provider, EntitlementContext | None], object] | None = None


class _KeyPool:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self._index = 0

    def next(self) -> str:
        if not self._keys:
            return ""
        key = self._keys[self._index % len(self._keys)]
        self._index = (self._index + 1) % len(self._keys)
        return key


class Kit:
    def __init__(self, config: KitConfig) -> None:
        if (
            not config.providers
            and not config.adapters
            and not config.adapter_factory
        ):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message="At least one provider configuration or adapter is required",
                )
            )
        self._providers, self._key_pools = self._prepare_providers(config.providers)
        self._adapters = self._build_adapters(self._providers)
        if config.adapters:
            self._adapters.update(config.adapters)
        self._external_adapter_factory = config.adapter_factory
        self._registry = ModelRegistry(
            self._adapters,
            ttl_seconds=config.registry_ttl_seconds,
            adapter_factory=self._adapter_factory,
        )

    def list_models(self, providers=None, refresh=False, entitlement=None):
        return self._registry.list_models(providers, refresh, entitlement)

    def list_model_records(self, providers=None, refresh=False, entitlement=None):
        return self._registry.list_model_records(providers, refresh, entitlement)

    def generate(self, input: GenerateInput) -> GenerateOutput:
        entitlement = self._entitlement_for_provider(input.provider)
        if entitlement:
            return self.generate_with_context(entitlement, input)
        adapter = self._require_adapter(input.provider)
        try:
            output = adapter.generate(input)
            return _attach_cost(input, output)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(None, input.provider, input.model, kit_err)
            raise kit_err

    def generate_with_context(
        self, entitlement: EntitlementContext | None, input: GenerateInput
    ) -> GenerateOutput:
        adapter = self._require_adapter(input.provider, entitlement)
        try:
            output = adapter.generate(input)
            return _attach_cost(input, output)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(entitlement, input.provider, input.model, kit_err)
            raise kit_err

    def generate_image(self, input: ImageGenerateInput) -> ImageGenerateOutput:
        entitlement = self._entitlement_for_provider(input.provider)
        if entitlement:
            return self.generate_image_with_context(entitlement, input)
        adapter = self._require_adapter(input.provider)
        if not hasattr(adapter, "generate_image"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support image generation",
                )
            )
        try:
            return adapter.generate_image(input)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(None, input.provider, input.model, kit_err)
            raise kit_err

    def generate_image_with_context(
        self, entitlement: EntitlementContext | None, input: ImageGenerateInput
    ) -> ImageGenerateOutput:
        adapter = self._require_adapter(input.provider, entitlement)
        if not hasattr(adapter, "generate_image"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support image generation",
                )
            )
        try:
            return adapter.generate_image(input)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(entitlement, input.provider, input.model, kit_err)
            raise kit_err

    def generate_mesh(self, input: MeshGenerateInput) -> MeshGenerateOutput:
        entitlement = self._entitlement_for_provider(input.provider)
        if entitlement:
            return self.generate_mesh_with_context(entitlement, input)
        adapter = self._require_adapter(input.provider)
        if not hasattr(adapter, "generate_mesh"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support mesh generation",
                )
            )
        try:
            return adapter.generate_mesh(input)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(None, input.provider, input.model, kit_err)
            raise kit_err

    def generate_mesh_with_context(
        self, entitlement: EntitlementContext | None, input: MeshGenerateInput
    ) -> MeshGenerateOutput:
        adapter = self._require_adapter(input.provider, entitlement)
        if not hasattr(adapter, "generate_mesh"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support mesh generation",
                )
            )
        try:
            return adapter.generate_mesh(input)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(entitlement, input.provider, input.model, kit_err)
            raise kit_err

    def generate_speech(self, input: SpeechGenerateInput) -> SpeechGenerateOutput:
        entitlement = self._entitlement_for_provider(input.provider)
        if entitlement:
            return self.generate_speech_with_context(entitlement, input)
        adapter = self._require_adapter(input.provider)
        if not hasattr(adapter, "generate_speech"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support speech generation",
                )
            )
        try:
            return adapter.generate_speech(input)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(None, input.provider, input.model, kit_err)
            raise kit_err

    def generate_speech_with_context(
        self, entitlement: EntitlementContext | None, input: SpeechGenerateInput
    ) -> SpeechGenerateOutput:
        adapter = self._require_adapter(input.provider, entitlement)
        if not hasattr(adapter, "generate_speech"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support speech generation",
                )
            )
        try:
            return adapter.generate_speech(input)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(entitlement, input.provider, input.model, kit_err)
            raise kit_err

    def transcribe(self, input: TranscribeInput) -> TranscribeOutput:
        entitlement = self._entitlement_for_provider(input.provider)
        if entitlement:
            return self.transcribe_with_context(entitlement, input)
        adapter = self._require_adapter(input.provider)
        if not hasattr(adapter, "transcribe"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support transcription",
                )
            )
        try:
            output = adapter.transcribe(input)
            return _attach_transcribe_cost(input, output)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(None, input.provider, input.model, kit_err)
            raise kit_err

    def transcribe_with_context(
        self, entitlement: EntitlementContext | None, input: TranscribeInput
    ) -> TranscribeOutput:
        adapter = self._require_adapter(input.provider, entitlement)
        if not hasattr(adapter, "transcribe"):
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.UNSUPPORTED,
                    message=f"Provider {input.provider} does not support transcription",
                )
            )
        try:
            output = adapter.transcribe(input)
            return _attach_transcribe_cost(input, output)
        except Exception as err:
            kit_err = to_kit_error(err)
            self._registry.learn_model_unavailable(entitlement, input.provider, input.model, kit_err)
            raise kit_err

    def stream_generate(self, input: GenerateInput):
        entitlement = self._entitlement_for_provider(input.provider)
        if entitlement:
            return self.stream_generate_with_context(entitlement, input)
        adapter = self._require_adapter(input.provider)
        return _attach_cost_stream(adapter.stream_generate(input), input.provider, input.model)

    def stream_generate_with_context(
        self, entitlement: EntitlementContext | None, input: GenerateInput
    ):
        adapter = self._require_adapter(input.provider, entitlement)
        return _attach_cost_stream(adapter.stream_generate(input), input.provider, input.model)

    def _build_adapters(self, providers: Dict[Provider, object]):
        adapters = {}
        if "openai" in providers:
            adapters["openai"] = OpenAIAdapter(providers["openai"])
        if "anthropic" in providers:
            adapters["anthropic"] = AnthropicAdapter(providers["anthropic"])
        if "google" in providers:
            adapters["google"] = GeminiAdapter(providers["google"])
        if "xai" in providers:
            adapters["xai"] = XAIAdapter(providers["xai"])
        if "bedrock" in providers:
            adapters["bedrock"] = BedrockAdapter(providers["bedrock"])
        if "ollama" in providers:
            adapters["ollama"] = OllamaAdapter(providers["ollama"])
        if "replicate" in providers:
            adapters["replicate"] = ReplicateAdapter(providers["replicate"])
        return adapters

    def _prepare_providers(self, providers: Dict[Provider, object]):
        key_pools: Dict[Provider, _KeyPool] = {}
        normalized: Dict[Provider, object] = {}
        for provider, cfg in providers.items():
            keys = self._collect_keys(cfg)
            if provider in ("ollama", "bedrock"):
                if keys:
                    key_pools[provider] = _KeyPool(keys)
                    normalized[provider] = self._with_api_key(cfg, keys[0])
                else:
                    normalized[provider] = cfg
                continue
            if not keys:
                raise AiKitError(
                    KitErrorPayload(
                        kind=ErrorKind.VALIDATION,
                        message=f"Provider {provider} api key is required",
                    )
                )
            key_pools[provider] = _KeyPool(keys)
            normalized[provider] = self._with_api_key(cfg, keys[0])
        return normalized, key_pools

    def _collect_keys(self, cfg: object) -> list[str]:
        api_key = getattr(cfg, "api_key", "") or ""
        api_keys = getattr(cfg, "api_keys", None) or []
        keys = []
        seen = set()
        for raw in [api_key, *api_keys]:
            trimmed = raw.strip() if isinstance(raw, str) else ""
            if not trimmed or trimmed in seen:
                continue
            seen.add(trimmed)
            keys.append(trimmed)
        return keys

    def _with_api_key(self, cfg: object, api_key: str) -> object:
        if hasattr(cfg, "__dataclass_fields__"):
            return replace(cfg, api_key=api_key)
        if hasattr(cfg, "api_key"):
            setattr(cfg, "api_key", api_key)
            return cfg
        return cfg

    def _entitlement_for_provider(self, provider: Provider) -> EntitlementContext | None:
        pool = self._key_pools.get(provider)
        if not pool:
            return None
        api_key = pool.next()
        if not api_key:
            return None
        return EntitlementContext(
            provider=provider,
            apiKey=api_key,
            apiKeyFingerprint=fingerprint_api_key(api_key),
        )

    def _adapter_factory(
        self, provider: Provider, entitlement: EntitlementContext | None
    ):
        if self._external_adapter_factory:
            adapter = self._external_adapter_factory(provider, entitlement)
            if adapter is not None:
                return adapter
        if not entitlement or not entitlement.apiKey:
            return self._adapters.get(provider)
        base_config = self._providers.get(provider)
        if base_config is None:
            return None
        if provider == "openai":
            config = OpenAIConfig(
                api_key=entitlement.apiKey,
                base_url=getattr(base_config, "base_url", "https://api.openai.com"),
                organization=getattr(base_config, "organization", None),
                default_use_responses=getattr(base_config, "default_use_responses", True),
                timeout=getattr(base_config, "timeout", None),
            )
            return OpenAIAdapter(config)
        if provider == "anthropic":
            config = AnthropicConfig(
                api_key=entitlement.apiKey,
                base_url=getattr(base_config, "base_url", "https://api.anthropic.com"),
                version=getattr(base_config, "version", None) or "2023-06-01",
                timeout=getattr(base_config, "timeout", None),
            )
            return AnthropicAdapter(config)
        if provider == "google":
            config = GeminiConfig(
                api_key=entitlement.apiKey,
                base_url=getattr(base_config, "base_url", "https://generativelanguage.googleapis.com"),
                timeout=getattr(base_config, "timeout", None),
            )
            return GeminiAdapter(config)
        if provider == "xai":
            config = XAIConfig(
                api_key=entitlement.apiKey,
                base_url=getattr(base_config, "base_url", "https://api.x.ai"),
                compatibility_mode=getattr(base_config, "compatibility_mode", "openai"),
                speech_mode=getattr(base_config, "speech_mode", "realtime"),
                timeout=getattr(base_config, "timeout", None),
            )
            return XAIAdapter(config)
        if provider == "bedrock":
            config = BedrockConfig(
                region=getattr(base_config, "region", ""),
                access_key_id=getattr(base_config, "access_key_id", ""),
                secret_access_key=getattr(base_config, "secret_access_key", ""),
                session_token=getattr(base_config, "session_token", None),
                endpoint=getattr(base_config, "endpoint", ""),
                runtime_endpoint=getattr(base_config, "runtime_endpoint", ""),
                control_plane_service=getattr(base_config, "control_plane_service", "bedrock"),
                runtime_service=getattr(base_config, "runtime_service", "bedrock-runtime"),
                timeout=getattr(base_config, "timeout", None),
            )
            return BedrockAdapter(config)
        if provider == "ollama":
            config = OllamaConfig(
                api_key=entitlement.apiKey,
                base_url=getattr(base_config, "base_url", "http://localhost:11434"),
                default_use_responses=getattr(base_config, "default_use_responses", False),
                timeout=getattr(base_config, "timeout", None),
            )
            return OllamaAdapter(config)
        if provider == "replicate":
            config = ReplicateConfig(
                api_key=entitlement.apiKey,
                api_keys=getattr(base_config, "api_keys", None),
            )
            return ReplicateAdapter(config)
        return None

    def _require_adapter(self, provider: Provider, entitlement: EntitlementContext | None = None):
        adapter = self._adapter_factory(provider, entitlement)
        if not adapter:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message=f"Provider {provider} is not configured",
                )
            )
        return adapter


def _attach_cost(input: GenerateInput, output: GenerateOutput) -> GenerateOutput:
    cost = estimate_cost(input.provider, input.model, output.usage)
    if cost is None:
        return output
    return replace(output, cost=cost)


def _transcribe_duration_seconds(output: TranscribeOutput) -> float | None:
    duration = getattr(output, "duration", None)
    if isinstance(duration, (int, float)):
        if duration >= 0:
            return float(duration)
    max_end: float | None = None
    for segments in (getattr(output, "segments", None), getattr(output, "words", None)):
        if not segments:
            continue
        for segment in segments:
            end = getattr(segment, "end", None)
            if isinstance(end, (int, float)):
                value = float(end)
                if max_end is None or value > max_end:
                    max_end = value
    return max_end


def _attach_transcribe_cost(input: TranscribeInput, output: TranscribeOutput) -> TranscribeOutput:
    if getattr(output, "cost", None) is not None:
        return output
    duration = _transcribe_duration_seconds(output)
    cost = estimate_transcribe_cost(input.provider, input.model, duration)
    if cost is None:
        return output
    return replace(output, cost=cost)


def _attach_cost_stream(stream, provider: Provider, model: str):
    for chunk in stream:
        if getattr(chunk, "type", None) == "message_end":
            cost = estimate_cost(provider, model, getattr(chunk, "usage", None))
            if cost is not None:
                yield replace(chunk, cost=cost)
                continue
        yield chunk
