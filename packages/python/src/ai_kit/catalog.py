from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ErrorKind, KitErrorPayload, InferenceKitError
from .types import ModelCapabilities, ModelMetadata, TokenPrices

_catalog_cache: Optional[List[ModelMetadata]] = None


def _shared_catalog_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[4] / "models" / "catalog_models.json"


def _local_catalog_path() -> Path:
    return Path(__file__).with_name("catalog_models.json")


def _parse_catalog_model(entry: Dict[str, Any]) -> Optional[ModelMetadata]:
    model_id = entry.get("id")
    if not model_id:
        return None
    capabilities = entry.get("capabilities") or {}
    token_prices = entry.get("tokenPrices")
    return ModelMetadata(
        id=str(model_id),
        displayName=str(entry.get("displayName") or model_id),
        provider=str(entry.get("provider") or "catalog"),
        capabilities=ModelCapabilities(
            text=bool(capabilities.get("text")),
            vision=bool(capabilities.get("vision")),
            tool_use=bool(capabilities.get("tool_use")),
            structured_output=bool(capabilities.get("structured_output")),
            reasoning=bool(capabilities.get("reasoning")),
        ),
        family=str(entry.get("family")) if entry.get("family") else None,
        contextWindow=entry.get("contextWindow") if isinstance(entry.get("contextWindow"), int) else None,
        tokenPrices=TokenPrices(
            input=token_prices.get("input"),
            output=token_prices.get("output"),
        )
        if isinstance(token_prices, dict)
        else None,
        deprecated=entry.get("deprecated") if "deprecated" in entry else None,
        inPreview=entry.get("inPreview") if "inPreview" in entry else None,
    )


def load_catalog_models() -> List[ModelMetadata]:
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    for path in (_shared_catalog_path(), _local_catalog_path()):
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                models: List[ModelMetadata] = []
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    model = _parse_catalog_model(entry)
                    if model is not None:
                        models.append(model)
                _catalog_cache = models
                return _catalog_cache
        except Exception:
            continue
    _catalog_cache = []
    return _catalog_cache


def has_catalog_models() -> bool:
    return len(load_catalog_models()) > 0


class CatalogAdapter:
    def __init__(self, models: List[ModelMetadata]) -> None:
        self._models = list(models)

    def list_models(self) -> List[ModelMetadata]:
        return list(self._models)

    def generate(self, *_args, **_kwargs):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Catalog provider does not support generate",
                provider="catalog",
            )
        )

    def generate_image(self, *_args, **_kwargs):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Catalog provider does not support image generation",
                provider="catalog",
            )
        )

    def generate_mesh(self, *_args, **_kwargs):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Catalog provider does not support mesh generation",
                provider="catalog",
            )
        )

    def stream_generate(self, *_args, **_kwargs):
        raise InferenceKitError(
            KitErrorPayload(
                kind=ErrorKind.UNSUPPORTED,
                message="Catalog provider does not support streaming",
                provider="catalog",
            )
        )
