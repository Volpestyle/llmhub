from __future__ import annotations

import io
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import replicate
import requests
from PIL import Image


def _is_file_output(obj: Any) -> bool:
    return hasattr(obj, "read") and callable(getattr(obj, "read"))


def _read_file_output(obj: Any) -> bytes:
    # Replicate FileOutput supports .read() (sync)
    return obj.read()  # type: ignore[no-any-return]


def _download_url(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    return r.content


class ReplicateClient:
    """
    Thin wrapper around the Replicate Python SDK.

    Replicate auth is resolved from env var REPLICATE_API_TOKEN by default.
    """

    def __init__(
        self,
        *,
        use_file_output: bool = True,
        max_retries: Optional[int] = None,
        base_delay_s: Optional[float] = None,
        max_delay_s: Optional[float] = None,
        min_interval_s: Optional[float] = None,
    ):
        self.use_file_output = use_file_output
        if max_retries is None:
            max_retries = int(os.getenv("AI_KIT_REPLICATE_MAX_RETRIES", "5"))
        if base_delay_s is None:
            base_delay_s = float(os.getenv("AI_KIT_REPLICATE_BASE_DELAY_S", "2"))
        if max_delay_s is None:
            max_delay_s = float(os.getenv("AI_KIT_REPLICATE_MAX_DELAY_S", "60"))
        if min_interval_s is None:
            min_interval_s = float(os.getenv("AI_KIT_REPLICATE_MIN_INTERVAL_S", "0"))
        self.max_retries = max(0, max_retries)
        self.base_delay_s = max(0.1, base_delay_s)
        self.max_delay_s = max(self.base_delay_s, max_delay_s)
        self.min_interval_s = max(0.0, min_interval_s)
        self._next_allowed_ts = 0.0

    def run(self, model: str, *, inputs: Dict[str, Any]) -> Any:
        # replicate.run(..., use_file_output=...) is supported in replicate>=1.0.0.
        attempt = 0
        saw_throttle = False
        while True:
            try:
                self._throttle_start()
                return replicate.run(model, input=inputs, use_file_output=self.use_file_output)
            except Exception as exc:
                if self._is_throttle(exc):
                    saw_throttle = True
                if not self._should_retry(exc, attempt, saw_throttle):
                    raise
                delay = self._retry_delay(exc, attempt)
                time.sleep(delay)
                attempt += 1

    def remove_background(
        self,
        *,
        model: str,
        image_path: Path,
        preserve_partial_alpha: bool = True,
        content_moderation: bool = False,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        with image_path.open("rb") as f:
            inputs: Dict[str, Any] = {
                "preserve_partial_alpha": preserve_partial_alpha,
                "content_moderation": content_moderation,
            }
            if parameters:
                inputs.update(parameters)
            inputs["image"] = f
            out = self.run(
                model,
                inputs=inputs,
            )
        return self._coerce_single_file(out)

    def multiview_zero123plusplus(
        self,
        *,
        model: str,
        image_path: Path,
        remove_background: bool = False,
        return_intermediate_images: bool = False,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[List[bytes], bytes]:
        with image_path.open("rb") as f:
            inputs: Dict[str, Any] = {
                "remove_background": remove_background,
                "return_intermediate_images": return_intermediate_images,
            }
            if parameters:
                inputs.update(parameters)
            inputs["image"] = f
            out = self.run(
                model,
                inputs=inputs,
            )
        # Common outputs: list[FileOutput] or list[url] or single FileOutput
        if isinstance(out, (list, tuple)):
            return [self._coerce_single_file(x) for x in out]
        return self._coerce_single_file(out)

    def depth_anything_v2(
        self,
        *,
        model: str,
        image_path: Path,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bytes]:
        with image_path.open("rb") as f:
            inputs: Dict[str, Any] = {}
            if parameters:
                inputs.update(parameters)
            inputs["image"] = f
            out = self.run(model, inputs=inputs)
        # Expected dict with keys like "grey_depth" and "color_depth"
        if not isinstance(out, dict):
            # Some model variants might return a single output; normalize.
            return {"grey_depth": self._coerce_single_file(out)}
        result: Dict[str, bytes] = {}
        for k, v in out.items():
            result[str(k)] = self._coerce_single_file(v)
        return result

    def nano_banana(
        self,
        *,
        model: str = "google/nano-banana",
        prompt: str,
        image_input: Optional[List[Union[str, Path]]] = None,
        aspect_ratio: Optional[str] = None,
        output_format: str = "png",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Google Nano Banana (Gemini 2.5 Flash Image) - image generation and editing.

        Supports:
        - Character and style consistency across multiple images
        - Multi-image fusion (up to 3 reference images)
        - Conversational editing with natural language
        - Style transfer and aesthetic inspiration

        Args:
            model: Replicate model ID (default: google/nano-banana)
            prompt: Text description of the image to generate or edit to make
            image_input: Optional list of input images (URLs or file paths)
            aspect_ratio: Optional aspect ratio for the output
            output_format: Output format (png or jpg)
            parameters: Additional parameters to pass to the model

        Returns:
            Generated image as bytes
        """
        inputs: Dict[str, Any] = {
            "prompt": prompt,
            "output_format": output_format,
        }

        if aspect_ratio:
            inputs["aspect_ratio"] = aspect_ratio

        if parameters:
            inputs.update(parameters)

        # Handle image inputs - can be URLs or file paths
        if image_input:
            processed_images: List[Any] = []
            for img in image_input:
                if isinstance(img, Path):
                    # Open file and add file handle
                    processed_images.append(img.open("rb"))
                elif isinstance(img, str):
                    if img.startswith("http"):
                        # URL - pass directly
                        processed_images.append(img)
                    else:
                        # Assume file path string
                        processed_images.append(Path(img).open("rb"))
                else:
                    processed_images.append(img)
            inputs["image_input"] = processed_images

        try:
            out = self.run(model, inputs=inputs)
        finally:
            # Close any file handles we opened
            if image_input:
                for img in inputs.get("image_input", []):
                    if hasattr(img, "close"):
                        img.close()

        return self._coerce_single_file(out)

    def nano_banana_batch(
        self,
        *,
        model: str = "google/nano-banana",
        prompt: str,
        image_inputs: List[List[Union[str, Path]]],
        aspect_ratio: Optional[str] = None,
        output_format: str = "png",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[bytes]:
        """
        Batch process multiple images with the same prompt.

        Useful for editing all anchor images in a PersonaPack with consistent
        customizations (e.g., change hair color across all poses).

        Args:
            model: Replicate model ID
            prompt: Text description of the edit to apply
            image_inputs: List of image input lists (one per output)
            aspect_ratio: Optional aspect ratio
            output_format: Output format
            parameters: Additional parameters

        Returns:
            List of generated images as bytes
        """
        results: List[bytes] = []
        for img_list in image_inputs:
            result = self.nano_banana(
                model=model,
                prompt=prompt,
                image_input=img_list,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                parameters=parameters,
            )
            results.append(result)
        return results

    def _coerce_single_file(self, out: Any) -> bytes:
        if out is None:
            raise RuntimeError("Replicate returned None output")
        if isinstance(out, bytes):
            return out
        if _is_file_output(out):
            return _read_file_output(out)
        if isinstance(out, str) and out.startswith("http"):
            return _download_url(out)
        # Some outputs can be dicts with 'url'
        if isinstance(out, dict) and "url" in out and isinstance(out["url"], str):
            return _download_url(out["url"])
        raise TypeError(f"Unsupported Replicate output type: {type(out)}")

    def _should_retry(self, exc: Exception, attempt: int, saw_throttle: bool) -> bool:
        if attempt >= self.max_retries:
            return False
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        if status == 429:
            return True
        if status == 404 and saw_throttle:
            return True
        message = self._error_message(exc).lower()
        return "429" in message or "throttl" in message or "rate limit" in message

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        message = self._error_message(exc)
        match = re.search(r"reset(?:s)? in ~?(\\d+)s", message, re.IGNORECASE)
        if match:
            return min(self.max_delay_s, float(match.group(1)) + 1.0)
        base = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        jitter = random.uniform(0, min(1.0, base * 0.1))
        return base + jitter

    def _is_throttle(self, exc: Exception) -> bool:
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        if status == 429:
            return True
        message = self._error_message(exc).lower()
        return "throttl" in message or "rate limit" in message

    def _error_message(self, exc: Exception) -> str:
        detail = getattr(exc, "detail", None)
        if isinstance(detail, str) and detail:
            return detail
        return str(exc)

    def _throttle_start(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = time.monotonic()
        if self._next_allowed_ts > now:
            time.sleep(self._next_allowed_ts - now)
        self._next_allowed_ts = time.monotonic() + self.min_interval_s

    @staticmethod
    def split_grid_image(
        *,
        grid_png: bytes,
        rows: int = 2,
        cols: int = 3,
        padding: int = 0,
    ) -> List[bytes]:
        """
        Best-effort helper for the case where a model returns a single PNG
        containing multiple views in a grid.

        Assumes all cells are equal size.
        """
        img = Image.open(io.BytesIO(grid_png)).convert("RGBA")
        w, h = img.size
        cell_w = (w - padding * (cols - 1)) // cols
        cell_h = (h - padding * (rows - 1)) // rows
        views: List[bytes] = []
        for r in range(rows):
            for c in range(cols):
                left = c * (cell_w + padding)
                top = r * (cell_h + padding)
                crop = img.crop((left, top, left + cell_w, top + cell_h))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                views.append(buf.getvalue())
        return views
