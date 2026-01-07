from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import ModelCapabilities, ModelMetadata, TokenPrices

_TASK_FAMILY = {
    "image-segmentation": "cutout",
    "depth-estimation": "depth",
    "novel-view": "views",
}


def _models_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[4] / "models"


def _local_models_root() -> Path:
    return Path(__file__).with_name("models")


def _optional_bool(caps: Dict[str, Any], key: str) -> Optional[bool]:
    if key not in caps:
        return None
    return bool(caps.get(key))


def _parse_capabilities(raw: Any) -> ModelCapabilities:
    caps = raw if isinstance(raw, dict) else {}
    return ModelCapabilities(
        text=bool(caps.get("text")),
        vision=bool(caps.get("vision")),
        image=bool(caps.get("image")),
        tool_use=bool(caps.get("tool_use")),
        structured_output=bool(caps.get("structured_output")),
        reasoning=bool(caps.get("reasoning")),
        audio_in=_optional_bool(caps, "audio_in"),
        audio_out=_optional_bool(caps, "audio_out"),
        video=_optional_bool(caps, "video"),
        video_in=_optional_bool(caps, "video_in"),
    )


def _parse_token_prices(raw: Any) -> Optional[TokenPrices]:
    if not isinstance(raw, dict):
        return None
    return TokenPrices(input=raw.get("input"), output=raw.get("output"))


def _parse_inputs(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(raw, list):
        return None
    return [item for item in raw if isinstance(item, dict)]


def _load_models_file(path: Path, provider_override: Optional[str] = None) -> List[ModelMetadata]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    models: List[ModelMetadata] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        provider = provider_override or entry.get("provider")
        model_id = entry.get("id")
        if not provider or not model_id:
            continue
        capabilities = _parse_capabilities(entry.get("capabilities"))
        token_prices = _parse_token_prices(entry.get("tokenPrices"))
        video_prices = entry.get("videoPrices")
        if not isinstance(video_prices, dict):
            video_prices = None
        inputs = _parse_inputs(entry.get("inputs"))
        models.append(
            ModelMetadata(
                id=str(model_id),
                displayName=str(entry.get("displayName") or model_id),
                provider=str(provider),
                capabilities=capabilities,
                family=str(entry.get("family")) if entry.get("family") else None,
                contextWindow=entry.get("contextWindow")
                if isinstance(entry.get("contextWindow"), int)
                else None,
                tokenPrices=token_prices,
                videoPrices=video_prices,
                deprecated=entry.get("deprecated") if "deprecated" in entry else None,
                inPreview=entry.get("inPreview") if "inPreview" in entry else None,
                inputs=inputs,
            )
        )
    return models


def _resolve_models_file(filename: str) -> Optional[Path]:
    for root in (_models_root(), _local_models_root()):
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def _load_local_catalog_models() -> List[ModelMetadata]:
    try:
        from .local.registry import REGISTRY  # type: ignore
        from .local import models as _local_models  # type: ignore  # noqa: F401
    except Exception:
        return []
    models: List[ModelMetadata] = []
    for spec in REGISTRY.list():
        family = _TASK_FAMILY.get(spec.task)
        if not family:
            continue
        models.append(
            ModelMetadata(
                id=spec.id,
                displayName=spec.id,
                provider="catalog",
                family=family,
                capabilities=ModelCapabilities(
                    text=False,
                    vision=False,
                    image=True,
                    tool_use=False,
                    structured_output=False,
                    reasoning=False,
                ),
            )
        )
    return models


def load_catalog_models() -> List[ModelMetadata]:
    """
    Load models that we explicitly catalog for UI selection and pipelines.

    - Local pipeline models are mapped to provider="catalog".
    - Manual provider catalogs (replicate/meshy) are loaded from JSON files.
    """
    models = _load_local_catalog_models()
    for provider, filename in (
        ("replicate", "replicate_models.json"),
        ("meshy", "meshy_models.json"),
        ("fal", "fal_models.json"),
    ):
        path = _resolve_models_file(filename)
        if not path:
            continue
        models.extend(_load_models_file(path, provider_override=provider))
    return models
