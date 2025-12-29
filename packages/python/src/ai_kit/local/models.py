from __future__ import annotations

from .registry import REGISTRY


REGISTRY.register(
    "image-segmentation",
    "rmbg-1.4",
    "briaai/RMBG-1.4",
    default=True,
)
REGISTRY.register(
    "depth-estimation",
    "depth-anything-v2-small",
    "depth-anything/Depth-Anything-V2-Small-hf",
    default=True,
)
REGISTRY.register(
    "depth-estimation",
    "depth-anything-v2-large",
    "depth-anything/Depth-Anything-V2-Large-hf",
)

REGISTRY.register(
    "novel-view",
    "zero123-plus",
    "sudo-ai/zero123plus-v1.2",
)
REGISTRY.register(
    "novel-view",
    "stable-zero123",
    "ashawkey/stable-zero123-diffusers",
    default=True,
)
REGISTRY.register(
    "novel-view",
    "zero123",
    "ashawkey/zero123",
)
