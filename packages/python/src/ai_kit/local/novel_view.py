from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from PIL import Image


@dataclass(frozen=True)
class NovelViewParams:
    steps: int = 28
    guidance_scale: float = 3.0


class NovelViewPipeline:
    def __init__(self, pipe, device_str: str) -> None:
        self._pipe = pipe
        self._device_str = device_str

    def generate(
        self,
        image: Image.Image,
        *,
        azimuth_deg: float,
        elevation_deg: float,
        seed: int,
        steps: int,
        guidance_scale: float,
        width: int | None = None,
        height: int | None = None,
    ) -> Image.Image:
        import torch

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        call_kwargs = _build_call_kwargs(
            self._pipe,
            image=image,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
            device_str=self._device_str,
        )
        result = self._pipe(**call_kwargs)
        images = getattr(result, "images", None)
        if isinstance(images, list) and images:
            return images[0]
        if isinstance(result, list) and result:
            return result[0]
        if isinstance(result, Image.Image):
            return result
        raise RuntimeError("Novel-view pipeline returned no images")


def get_novel_view_pipeline(model: str, device_str: str) -> NovelViewPipeline:
    import torch
    from diffusers import DiffusionPipeline

    device = torch.device(device_str)
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    trust_remote_code = _env_value(
        "AI_KIT_TRUST_REMOTE_CODE",
        "INFERENCE_KIT_TRUST_REMOTE_CODE",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return NovelViewPipeline(pipe, device_str)


def _build_call_kwargs(
    pipe: Any,
    *,
    image: Image.Image,
    azimuth_deg: float,
    elevation_deg: float,
    steps: int,
    guidance_scale: float,
    generator: Any,
    width: int | None,
    height: int | None,
    device_str: str,
) -> Dict[str, Any]:
    params = inspect.signature(pipe.__call__).parameters
    kwargs: Dict[str, Any] = {}

    image_param = _first_param(params, ("image", "conditioning_image", "input_image"))
    if not image_param:
        raise RuntimeError("Novel-view pipeline does not accept an image parameter")
    kwargs[image_param] = image

    az_set = _set_if_present(params, ("azimuth", "azimuth_deg", "yaw", "theta"), azimuth_deg, kwargs)
    el_set = _set_if_present(params, ("elevation", "elevation_deg", "pitch", "phi"), elevation_deg, kwargs)
    if not (az_set and el_set) and "camera" in params:
        import torch

        camera = torch.tensor([[elevation_deg, azimuth_deg, 0.0]], dtype=torch.float32)
        if device_str:
            camera = camera.to(device_str)
        kwargs["camera"] = camera
        az_set = True
        el_set = True
    if not (az_set and el_set):
        raise RuntimeError("Novel-view pipeline does not accept azimuth/elevation parameters")

    if "roll" in params:
        kwargs["roll"] = 0.0
    if "distance" in params:
        kwargs["distance"] = 1.0
    if "radius" in params:
        kwargs["radius"] = 1.0
    if "camera_distance" in params:
        kwargs["camera_distance"] = 1.0
    if "num_inference_steps" in params:
        kwargs["num_inference_steps"] = int(steps)
    if "guidance_scale" in params:
        kwargs["guidance_scale"] = float(guidance_scale)
    if "generator" in params:
        kwargs["generator"] = generator
    if width is not None and "width" in params:
        kwargs["width"] = int(width)
    if height is not None and "height" in params:
        kwargs["height"] = int(height)
    if "output_type" in params:
        kwargs["output_type"] = "pil"
    if "num_images_per_prompt" in params:
        kwargs["num_images_per_prompt"] = 1

    return kwargs


def _env_value(primary: str, legacy: str) -> str:
    value = os.getenv(primary, "")
    if value:
        return value
    return os.getenv(legacy, "")


def _set_if_present(
    params: Dict[str, inspect.Parameter],
    keys: Iterable[str],
    value: float,
    kwargs: Dict[str, Any],
) -> bool:
    for key in keys:
        if key in params:
            kwargs[key] = float(value)
            return True
    return False


def _first_param(params: Dict[str, inspect.Parameter], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in params:
            return key
    return None
