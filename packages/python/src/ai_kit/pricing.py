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
    for item in base_dir.iterdir():
        if item.is_dir():
            path = item / "scraped_models.json"
            provider = item.name
        elif item.is_file() and item.name.endswith("_models.json"):
            path = item
            provider = item.name.removesuffix("_models.json")
        else:
            continue
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
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


def _normalize_resolution_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = "".join(ch for ch in value.lower() if ch.isalnum())
    return cleaned or None


def _first_price(raw: Dict[str, Any], keys: list[str]) -> Optional[float]:
    for key in keys:
        if key in raw:
            value = _coerce_float(raw.get(key))
            if value is not None:
                return value
    return None


def _video_price_per_second(
    curated: Dict[str, Any],
    *,
    resolution: Optional[str] = None,
    with_audio: Optional[bool] = None,
) -> Optional[float]:
    raw = curated.get("videoPrices") or curated.get("video_prices")
    if not isinstance(raw, dict):
        return None

    resolution_key = _normalize_resolution_key(resolution)
    if resolution_key:
        if with_audio:
            price = _first_price(
                raw,
                [
                    f"per_second_usd_{resolution_key}_with_audio",
                    f"per_second_{resolution_key}_with_audio",
                    f"perSecondUsd{resolution_key}WithAudio",
                ],
            )
            if price is not None:
                return price
        price = _first_price(
            raw,
            [
                f"per_second_usd_{resolution_key}",
                f"per_second_{resolution_key}",
                f"perSecondUsd{resolution_key}",
            ],
        )
        if price is not None:
            return price

    if with_audio:
        price = _first_price(
            raw,
            [
                "per_second_usd_with_audio",
                "per_second_with_audio",
                "perSecondUsdWithAudio",
            ],
        )
        if price is not None:
            return price

    price = _first_price(
        raw,
        [
            "per_second_usd",
            "perSecond",
            "per_second",
            "perSecondUsd",
        ],
    )
    if price is not None:
        return price

    max_price = _first_price(raw, ["per_second_usd_max", "perSecondMax", "per_second_max"])
    if max_price is not None:
        return max_price
    min_price = _first_price(raw, ["per_second_usd_min", "perSecondMin", "per_second_min"])
    if min_price is not None:
        return min_price

    resolution_prices: list[float] = []
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if not key.startswith("per_second_usd_") and not key.startswith("per_second_"):
            continue
        if key.endswith("_min") or key.endswith("_max") or key.endswith("_with_audio"):
            continue
        coerced = _coerce_float(value)
        if coerced is not None:
            resolution_prices.append(coerced)
    if resolution_prices:
        return max(resolution_prices)

    per_minute = _first_price(
        raw,
        [
            "per_minute_usd",
            "perMinute",
            "per_minute",
        ],
    )
    if per_minute is not None:
        return per_minute / 60.0
    return None


def _video_price_per_request(
    curated: Dict[str, Any],
    *,
    resolution: Optional[str] = None,
    with_audio: Optional[bool] = None,
) -> Optional[float]:
    raw = curated.get("videoPrices") or curated.get("video_prices")
    if not isinstance(raw, dict):
        return None
    resolution_key = _normalize_resolution_key(resolution)
    if resolution_key:
        if with_audio:
            price = _first_price(
                raw,
                [
                    f"per_request_usd_{resolution_key}_with_audio",
                    f"per_request_{resolution_key}_with_audio",
                    f"perRequestUsd{resolution_key}WithAudio",
                ],
            )
            if price is not None:
                return price
        price = _first_price(
            raw,
            [
                f"per_request_usd_{resolution_key}",
                f"per_request_{resolution_key}",
                f"perRequestUsd{resolution_key}",
            ],
        )
        if price is not None:
            return price
    if with_audio:
        price = _first_price(
            raw,
            [
                "per_request_usd_with_audio",
                "per_request_with_audio",
                "perRequestUsdWithAudio",
            ],
        )
        if price is not None:
            return price
    return _first_price(raw, ["per_request_usd", "perRequest", "per_request"])


def estimate_video_cost(
    provider: Provider,
    model_id: str,
    duration_seconds: Optional[float],
    *,
    resolution: Optional[str] = None,
    with_audio: Optional[bool] = None,
) -> Optional[CostBreakdown]:
    curated = find_curated_model(provider, model_id)
    if not curated:
        return None
    per_request = _video_price_per_request(curated, resolution=resolution, with_audio=with_audio)
    if per_request is not None:
        return CostBreakdown(
            input_cost_usd=per_request,
            output_cost_usd=0.0,
            total_cost_usd=per_request,
        )
    if duration_seconds is None:
        return None
    try:
        duration_value = float(duration_seconds)
    except (TypeError, ValueError):
        return None
    if duration_value <= 0:
        return None
    rate_per_second = _video_price_per_second(curated, resolution=resolution, with_audio=with_audio)
    if rate_per_second is None or rate_per_second <= 0:
        return None
    total = round(duration_value * rate_per_second, 6)
    return CostBreakdown(
        input_cost_usd=total,
        output_cost_usd=0.0,
        total_cost_usd=total,
    )
