from __future__ import annotations

import os


def resolve_device(preferred: object | None = None):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for local model runners") from exc
    if isinstance(preferred, torch.device):
        return preferred
    if _env_bool("INFERENCE_KIT_LOCAL_DISABLE_GPU", False):
        return torch.device("cpu")
    raw = str(preferred or os.getenv("INFERENCE_KIT_LOCAL_DEVICE", "")).strip().lower()
    if raw and raw != "auto":
        return torch.device(raw)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")
