from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import CostBreakdown, ModelCapabilities, ModelMetadata, Provider, TokenPrices, Usage

_scraped_cache: Optional[List[Dict[str, Any]]] = None


def _shared_models_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parents[4] / "models"


def _local_models_dir() -> Path:
    return Path(__file__).with_name("models")


def _load_scraped_from_dir(base_dir: Path) -> List[Dict[str, Any]]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    models: List[Dict[str, Any]] = []
    for provider_dir in base_dir.iterdir():
        if not provider_dir.is_dir():
            continue
        path = provider_dir / "scraped_models.json"
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        provider = provider_dir.name
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if "provider" not in entry:
                entry = {**entry, "provider": provider}
            models.append(entry)
    return models


def load_scraped_models() -> List[Dict[str, Any]]:
    global _scraped_cache
    if _scraped_cache is not None:
        return _scraped_cache
    models: List[Dict[str, Any]] = []
    for base_dir in (_shared_models_dir(), _local_models_dir()):
        models.extend(_load_scraped_from_dir(base_dir))
    _scraped_cache = models
    return _scraped_cache


def load_curated_models() -> List[Dict[str, Any]]:
    # Backwards-compatible alias.
    return load_scraped_models()


def _normalize_model_id(provider: Provider, model_id: str) -> str:
    prefix = f"{provider}/"
    if model_id.startswith(prefix):
        return model_id[len(prefix) :]
    return model_id


def find_curated_model(provider: Provider, model_id: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize_model_id(provider, model_id)
    best: Optional[Dict[str, Any]] = None
    for entry in load_scraped_models():
        if entry.get("provider") != provider:
            continue
        entry_id = entry.get("id") or ""
        if entry_id == normalized:
            return entry
        if normalized.startswith(entry_id) and (not best or len(entry_id) > len(best.get("id") or "")):
            best = entry
    return best


def apply_curated_metadata(model: ModelMetadata) -> ModelMetadata:
    curated = find_curated_model(model.provider, model.id)
    if not curated:
        return model
    capabilities = model.capabilities
    curated_caps = curated.get("capabilities")
    if isinstance(curated_caps, dict):
        capabilities = ModelCapabilities(
            text=bool(curated_caps.get("text", capabilities.text)),
            vision=bool(curated_caps.get("vision", capabilities.vision)),
            image=bool(curated_caps.get("image", getattr(capabilities, "image", False))),
            tool_use=bool(curated_caps.get("tool_use", capabilities.tool_use)),
            structured_output=bool(curated_caps.get("structured_output", capabilities.structured_output)),
            reasoning=bool(curated_caps.get("reasoning", capabilities.reasoning)),
            audio_in=bool(curated_caps.get("audio_in", getattr(capabilities, "audio_in", False))),
            audio_out=bool(curated_caps.get("audio_out", getattr(capabilities, "audio_out", False))),
            video=bool(curated_caps.get("video", getattr(capabilities, "video", False))),
            video_in=bool(curated_caps.get("video_in", getattr(capabilities, "video_in", False))),
        )
    token_prices = model.tokenPrices
    curated_prices = curated.get("tokenPrices")
    if isinstance(curated_prices, dict):
        token_prices = TokenPrices(
            input=curated_prices.get("input", token_prices.input if token_prices else None),
            output=curated_prices.get("output", token_prices.output if token_prices else None),
        )
    video_prices = curated.get("videoPrices")
    if not isinstance(video_prices, dict):
        video_prices = model.videoPrices
    context_window = curated.get("contextWindow")
    if not isinstance(context_window, int):
        context_window = model.contextWindow
    return replace(
        model,
        displayName=curated.get("displayName") or model.displayName,
        family=curated.get("family") or model.family,
        capabilities=capabilities,
        contextWindow=context_window,
        tokenPrices=token_prices,
        videoPrices=video_prices,
        deprecated=bool(curated.get("deprecated", model.deprecated)),
        inPreview=bool(curated.get("inPreview", model.inPreview)),
    )


def lookup_token_prices(provider: Provider, model_id: str) -> Optional[TokenPrices]:
    curated = find_curated_model(provider, model_id)
    if not curated:
        return None
    prices = curated.get("tokenPrices")
    if not isinstance(prices, dict):
        return None
    return TokenPrices(input=prices.get("input"), output=prices.get("output"))

def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _audio_price_per_minute(curated: Dict[str, Any]) -> Optional[float]:
    raw = curated.get("audioPrices") or curated.get("transcribePrices") or curated.get("audio_prices")
    if isinstance(raw, dict):
        per_minute = _coerce_float(raw.get("perMinute") or raw.get("per_minute") or raw.get("perMinuteUsd"))
        if per_minute is not None:
            return per_minute
        per_second = _coerce_float(raw.get("perSecond") or raw.get("per_second") or raw.get("perSecondUsd"))
        if per_second is not None:
            return per_second * 60.0
        return None
    return _coerce_float(raw)


def estimate_cost(provider: Provider, model_id: str, usage: Optional[Usage]) -> Optional[CostBreakdown]:
    if usage is None:
        return None
    if usage.inputTokens is None and usage.outputTokens is None:
        return None
    pricing = lookup_token_prices(provider, model_id)
    if pricing is None:
        return None
    input_rate = pricing.input or 0.0
    output_rate = pricing.output or 0.0
    if input_rate == 0 and output_rate == 0:
        return None
    input_tokens = usage.inputTokens or 0
    output_tokens = usage.outputTokens or 0
    input_cost = input_tokens * input_rate / 1_000_000
    output_cost = output_tokens * output_rate / 1_000_000
    return CostBreakdown(
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(input_cost + output_cost, 6),
        pricing_per_million=pricing,
    )


def estimate_transcribe_cost(
    provider: Provider,
    model_id: str,
    duration_seconds: Optional[float],
) -> Optional[CostBreakdown]:
    if duration_seconds is None:
        return None
    try:
        duration_value = float(duration_seconds)
    except (TypeError, ValueError):
        return None
    if duration_value <= 0:
        return None
    curated = find_curated_model(provider, model_id)
    if not curated:
        return None
    rate_per_minute = _audio_price_per_minute(curated)
    if rate_per_minute is None or rate_per_minute <= 0:
        return None
    total = round((duration_value / 60.0) * rate_per_minute, 6)
    return CostBreakdown(
        input_cost_usd=total,
        output_cost_usd=0.0,
        total_cost_usd=total,
    )
