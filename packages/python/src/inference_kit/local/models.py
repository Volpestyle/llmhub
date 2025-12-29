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
