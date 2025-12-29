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
    "stable-zero123",
    "ashawkey/stable-zero123-diffusers",
    default=True,
)
REGISTRY.register(
    "novel-view",
    "zero123-xl",
    "ashawkey/zero123-xl-diffusers",
)
REGISTRY.register(
    "novel-view",
    "zero1to3-105000",
    "kxic/zero123-105000",
)
REGISTRY.register(
    "novel-view",
    "zero1to3-165000",
    "kxic/zero123-165000",
)
REGISTRY.register(
    "novel-view",
    "zero1to3-xl",
    "kxic/zero123-xl",
)
