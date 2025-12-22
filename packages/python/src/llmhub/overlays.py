from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .types import ModelCapabilities, ModelMetadata, Provider, TokenPrices


_curated_cache: Optional[Dict[str, Dict[str, ModelMetadata]]] = None


def _load_curated() -> Dict[str, Dict[str, ModelMetadata]]:
    global _curated_cache
    if _curated_cache is not None:
        return _curated_cache
    path = os.path.join(os.path.dirname(__file__), "curated_models.json")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    curated: Dict[str, Dict[str, ModelMetadata]] = {}
    for entry in raw:
        provider = entry.get("provider")
        if not provider:
            continue
        caps = entry.get("capabilities") or {}
        model = ModelMetadata(
            id=entry.get("id"),
            displayName=entry.get("displayName") or entry.get("id"),
            provider=provider,
            family=entry.get("family"),
            capabilities=ModelCapabilities(
                text=bool(caps.get("text")),
                vision=bool(caps.get("vision")),
                tool_use=bool(caps.get("tool_use")),
                structured_output=bool(caps.get("structured_output")),
                reasoning=bool(caps.get("reasoning")),
            ),
            contextWindow=entry.get("contextWindow"),
            tokenPrices=TokenPrices(
                input=(entry.get("tokenPrices") or {}).get("input"),
                output=(entry.get("tokenPrices") or {}).get("output"),
            )
            if entry.get("tokenPrices")
            else None,
            deprecated=entry.get("deprecated"),
            inPreview=entry.get("inPreview"),
        )
        curated.setdefault(provider, {})[model.id] = model
    _curated_cache = curated
    return curated


def lookup_curated(provider: Provider, model_id: str) -> Optional[ModelMetadata]:
    curated = _load_curated()
    return curated.get(provider, {}).get(model_id)
