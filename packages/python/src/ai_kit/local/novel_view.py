from __future__ import annotations

import inspect
import os
from pathlib import Path
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
    from huggingface_hub import snapshot_download

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

    snapshot_dir = Path(snapshot_download(model))
    zero123_pipeline = snapshot_dir / "clip_camera_projection" / "zero123.py"
    if zero123_pipeline.exists():
        _ensure_torch_xpu_stub(torch)
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            str(snapshot_dir),
            torch_dtype=dtype,
            custom_pipeline="clip_camera_projection/zero123",
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            use_safetensors=True,
        )
    else:
        _ensure_torch_xpu_stub(torch)
        from .zero1to3_pipeline import Zero1to3StableDiffusionPipeline

        model_path = _ensure_zero1to3_components(
            model,
            snapshot_download,
            snapshot_dir=snapshot_dir,
        )
        pipe = Zero1to3StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
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

    image_param = _first_param(params, ("image", "conditioning_image", "input_image", "input_imgs"))
    if not image_param:
        raise RuntimeError("Novel-view pipeline does not accept an image parameter")
    kwargs[image_param] = image
    if image_param == "input_imgs" and "prompt_imgs" in params:
        kwargs["prompt_imgs"] = image

    az_set = _set_if_present(params, ("azimuth", "azimuth_deg", "yaw", "theta"), azimuth_deg, kwargs)
    el_set = _set_if_present(params, ("elevation", "elevation_deg", "pitch", "phi"), elevation_deg, kwargs)
    if "poses" in params:
        kwargs["poses"] = [float(elevation_deg), float(azimuth_deg), 0.0]
        az_set = True
        el_set = True
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


def _ensure_torch_xpu_stub(torch_module) -> None:
    if hasattr(torch_module, "xpu"):
        return

    class _XPU:
        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def manual_seed(seed: int):
            return torch_module.manual_seed(seed)

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def is_available() -> bool:
            return False

    torch_module.xpu = _XPU()


def _ensure_zero1to3_components(
    model: str,
    snapshot_download,
    *,
    snapshot_dir: Path | None = None,
) -> str:
    snapshot_dir = Path(snapshot_dir or snapshot_download(model))
    component_dir = Path(snapshot_dir) / "cc_projection"
    component_dir.mkdir(parents=True, exist_ok=True)
    component_file = component_dir / "pipeline_zero1to3.py"
    if not component_file.exists():
        component_file.write_text(
            "from ai_kit.local.zero1to3_pipeline import CCProjection\n\n"
            "__all__ = [\"CCProjection\"]\n"
        )
    return str(snapshot_dir)


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
