from __future__ import annotations

from typing import List

from .types import ModelRecord, ModelResolutionRequest, ResolvedModel


class ModelRouter:
    def resolve(self, models: List[ModelRecord], request: ModelResolutionRequest) -> ResolvedModel:
        candidates = _filter_models(models, request)
        if not candidates:
            raise ValueError("router: no models match constraints")
        return ResolvedModel(
            primary=candidates[0],
            fallback=candidates[1:] if len(candidates) > 1 else None,
        )


def _filter_models(models: List[ModelRecord], request: ModelResolutionRequest) -> List[ModelRecord]:
    constraints = request.constraints
    preferred = [entry.strip() for entry in (request.preferredModels or []) if entry.strip()]
    allow_preview = constraints.allowPreview if constraints else True

    def matches(model: ModelRecord) -> bool:
        if not model.availability.entitled:
            return False
        if constraints:
            if constraints.requireTools and not model.features.tools:
                return False
            if constraints.requireJson and not (
                model.features.jsonMode or model.features.jsonSchema
            ):
                return False
            if constraints.requireVision and not model.modalities.vision:
                return False
            if constraints.maxCostUsd and not _within_cost(model, constraints.maxCostUsd):
                return False
        if allow_preview is False and model.tags and "preview" in model.tags:
            return False
        if preferred and not _matches_preferred(model, preferred):
            return False
        return True

    candidates = [model for model in models if matches(model)]
    candidates.sort(key=lambda m: (
        _preferred_rank(m, preferred),
        _price_score(m),
        m.displayName or "",
    ))
    return candidates


def _matches_preferred(model: ModelRecord, preferred: List[str]) -> bool:
    return any(entry in (model.id, model.providerModelId) for entry in preferred)


def _preferred_rank(model: ModelRecord, preferred: List[str]) -> int:
    if not preferred:
        return len(preferred) + 1
    for idx, entry in enumerate(preferred):
        if entry in (model.id, model.providerModelId):
            return idx
    return len(preferred) + 1


def _within_cost(model: ModelRecord, max_cost: float) -> bool:
    if not model.pricing:
        return True
    if model.pricing.inputPer1M and model.pricing.inputPer1M > max_cost:
        return False
    if model.pricing.outputPer1M and model.pricing.outputPer1M > max_cost:
        return False
    return True


def _price_score(model: ModelRecord) -> float:
    if not model.pricing:
        return 0.0
    inputs = model.pricing.inputPer1M or 0.0
    outputs = model.pricing.outputPer1M or 0.0
    if inputs and outputs:
        return min(inputs, outputs)
    return inputs or outputs
