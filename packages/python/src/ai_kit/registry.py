from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .entitlements import fingerprint_api_key
from .errors import ErrorKind, KitErrorPayload, AiKitError
from .pricing import apply_curated_metadata
from .types import (
    EntitlementContext,
    ModelAvailability,
    ModelFeatures,
    ModelLimits,
    ModelMetadata,
    ModelModalities,
    ModelPricing,
    ModelRecord,
    Provider,
)


@dataclass
class CacheEntry:
    data: List[ModelMetadata]
    fetched_at: float
    expires_at: float


class ModelRegistry:
    def __init__(
        self,
        adapters: Dict[Provider, object],
        ttl_seconds: int = 1800,
        learned_ttl_seconds: int = 1200,
        adapter_factory=None,
    ) -> None:
        self._adapters = adapters
        self._adapter_factory = adapter_factory
        self._ttl_seconds = ttl_seconds
        self._learned_ttl_seconds = learned_ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._learned: Dict[str, Tuple[float, str]] = {}

    def list_models(
        self,
        providers: Optional[List[Provider]] = None,
        refresh: bool = False,
        entitlement: Optional[EntitlementContext] = None,
    ) -> List[ModelMetadata]:
        entries = self._entries_for_providers(providers, refresh, entitlement)
        models = []
        for entry in entries.values():
            models.extend(entry.data)
        return sorted(models, key=lambda m: (m.provider, m.displayName))

    def list_model_records(
        self,
        providers: Optional[List[Provider]] = None,
        refresh: bool = False,
        entitlement: Optional[EntitlementContext] = None,
    ) -> List[ModelRecord]:
        entries = self._entries_for_providers(providers, refresh, entitlement)
        records: List[ModelRecord] = []
        for provider, entry in entries.items():
            for model in entry.data:
                records.append(
                    self._to_record(model, provider, entry.fetched_at, entitlement)
                )
        return sorted(records, key=lambda r: (r.provider, r.displayName or ""))

    def learn_model_unavailable(
        self,
        entitlement: Optional[EntitlementContext],
        provider: Provider,
        model_id: str,
        err: Exception,
    ) -> None:
        reason = _learn_reason(err)
        if not reason:
            return
        key = self._learned_key(provider, entitlement, model_id)
        self._learned[key] = (
            _now_timestamp() + self._learned_ttl_seconds,
            reason,
        )

    def _entries_for_providers(
        self,
        providers: Optional[List[Provider]],
        refresh: bool,
        entitlement: Optional[EntitlementContext],
    ) -> Dict[Provider, CacheEntry]:
        resolved = self._resolve_providers(providers, entitlement)
        if not resolved:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message="No providers configured",
                )
            )
        results: Dict[Provider, CacheEntry] = {}
        for provider in resolved:
            results[provider] = self._for_provider(provider, refresh, entitlement)
        return results

    def _resolve_providers(
        self,
        providers: Optional[List[Provider]],
        entitlement: Optional[EntitlementContext],
    ) -> List[Provider]:
        if providers:
            return providers
        if entitlement and entitlement.provider:
            return [entitlement.provider]
        return list(self._adapters.keys())

    def _for_provider(
        self,
        provider: Provider,
        refresh: bool,
        entitlement: Optional[EntitlementContext],
    ) -> CacheEntry:
        key = self._cache_key(provider, entitlement)
        if not refresh:
            cached = self._cache.get(key)
            if cached and cached.expires_at > _now_timestamp():
                return cached
        try:
            entry = self._fetch_and_cache(provider, entitlement, key)
            return entry
        except Exception as err:
            cached = self._cache.get(key)
            if cached and cached.expires_at > _now_timestamp():
                return cached
            raise err

    def _fetch_and_cache(
        self,
        provider: Provider,
        entitlement: Optional[EntitlementContext],
        key: str,
    ) -> CacheEntry:
        adapter = self._adapter_for(provider, entitlement)
        if not adapter:
            raise AiKitError(
                KitErrorPayload(
                    kind=ErrorKind.VALIDATION,
                    message=f"Provider {provider} is not configured",
                )
            )
        models = adapter.list_models()
        models = [apply_curated_metadata(model) for model in models]
        now = _now_timestamp()
        entry = CacheEntry(
            data=models,
            fetched_at=now,
            expires_at=now + self._ttl_seconds,
        )
        self._cache[key] = entry
        return entry

    def _adapter_for(self, provider: Provider, entitlement: Optional[EntitlementContext]):
        if self._adapter_factory:
            return self._adapter_factory(provider, entitlement)
        return self._adapters.get(provider)

    def _cache_key(self, provider: Provider, entitlement: Optional[EntitlementContext]) -> str:
        fingerprint = "default"
        if entitlement:
            fingerprint = entitlement.apiKeyFingerprint or fingerprint_api_key(entitlement.apiKey) or "default"
        return "|".join(
            [
                provider,
                fingerprint,
                (entitlement.accountId if entitlement else "") or "",
                (entitlement.region if entitlement else "") or "",
                (entitlement.environment if entitlement else "") or "",
                (entitlement.tenantId if entitlement else "") or "",
                (entitlement.userId if entitlement else "") or "",
            ]
        )

    def _learned_key(
        self,
        provider: Provider,
        entitlement: Optional[EntitlementContext],
        model_id: str,
    ) -> str:
        return f"{self._cache_key(provider, entitlement)}|{model_id}"

    def _learned_status(
        self,
        provider: Provider,
        entitlement: Optional[EntitlementContext],
        model_id: str,
    ) -> Optional[str]:
        key = self._learned_key(provider, entitlement, model_id)
        entry = self._learned.get(key)
        if not entry:
            return None
        expires_at, reason = entry
        if expires_at < _now_timestamp():
            self._learned.pop(key, None)
            return None
        return reason

    def _to_record(
        self,
        model: ModelMetadata,
        provider: Provider,
        fetched_at: float,
        entitlement: Optional[EntitlementContext],
    ) -> ModelRecord:
        modalities = ModelModalities(
            text=model.capabilities.text,
            vision=model.capabilities.vision,
            audioIn=getattr(model.capabilities, "audio_in", None),
            audioOut=getattr(model.capabilities, "audio_out", None),
            imageOut=getattr(model.capabilities, "image", None),
        )
        features = ModelFeatures(
            tools=model.capabilities.tool_use,
            jsonMode=model.capabilities.structured_output,
            jsonSchema=model.capabilities.structured_output,
            streaming=True,
        )
        limits = (
            ModelLimits(contextTokens=model.contextWindow)
            if model.contextWindow
            else None
        )
        pricing = (
            ModelPricing(
                currency="USD",
                inputPer1M=model.tokenPrices.input if model.tokenPrices else None,
                outputPer1M=model.tokenPrices.output if model.tokenPrices else None,
                source="config",
            )
            if model.tokenPrices
            else None
        )
        tags = []
        if model.inPreview:
            tags.append("preview")
        if model.deprecated:
            tags.append("deprecated")
        availability = ModelAvailability(
            entitled=True,
            confidence="listed",
            lastVerifiedAt=_to_iso(fetched_at),
        )
        learned_reason = self._learned_status(provider, entitlement, model.id)
        if learned_reason:
            availability.entitled = False
            availability.confidence = "learned"
            availability.reason = learned_reason
        return ModelRecord(
            id=f"{provider}:{model.id}",
            provider=provider,
            providerModelId=model.id,
            displayName=model.displayName,
            modalities=modalities,
            features=features,
            limits=limits,
            tags=tags or None,
            pricing=pricing,
            availability=availability,
        )


def _now_timestamp() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def _to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _learn_reason(err: Exception) -> Optional[str]:
    if isinstance(err, AiKitError):
        if err.kind in (ErrorKind.PROVIDER_NOT_FOUND, ErrorKind.VALIDATION):
            return str(err)
        if err.upstreamStatus in (400, 403, 404):
            return str(err)
    return None
