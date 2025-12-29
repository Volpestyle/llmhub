from __future__ import annotations

import os
from functools import lru_cache

from .device import resolve_device


def get_pipeline(task: str, model: str, device: object | None = None):
    resolved = resolve_device(device)
    return _get_pipeline_cached(task, model, str(resolved))


@lru_cache(maxsize=4)
def _get_pipeline_cached(task: str, model: str, device_str: str):
    if task == "novel-view":
        from .novel_view import get_novel_view_pipeline
        return get_novel_view_pipeline(model, device_str)

    from transformers import pipeline as hf_pipeline

    trust_remote_code = _env_value(
        "AI_KIT_TRUST_REMOTE_CODE",
        "INFERENCE_KIT_TRUST_REMOTE_CODE",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    pipe = hf_pipeline(task, model=model, trust_remote_code=trust_remote_code)
    _move_pipeline_to_device(pipe, device_str)
    return pipe


def _move_pipeline_to_device(pipe, device_str: str) -> None:
    import torch

    device = torch.device(device_str)
    if hasattr(pipe, "model"):
        pipe.model.to(device)
    if hasattr(pipe, "device"):
        pipe.device = device


def _env_value(primary: str, legacy: str) -> str:
    value = os.getenv(primary, "")
    if value:
        return value
    return os.getenv(legacy, "")
