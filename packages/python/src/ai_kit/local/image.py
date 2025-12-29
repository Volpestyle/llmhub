from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def apply_mask_to_rgba(
    image: Image.Image,
    mask: Image.Image,
    *,
    feather_px: int = 0,
) -> Image.Image:
    mask_l = mask.convert("L")
    if feather_px > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(radius=feather_px))
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_l)
    return rgba


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    min_val = float(depth.min())
    max_val = float(depth.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    scaled = (depth - min_val) / (max_val - min_val)
    return (scaled * 255).clip(0, 255).astype(np.uint8)
