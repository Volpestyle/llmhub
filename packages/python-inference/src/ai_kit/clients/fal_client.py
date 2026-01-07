from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import fal_client
import requests


class FalClient:
    """
    Thin wrapper around fal-client for file uploads + multiview to 3D requests.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        key = api_key or os.getenv("AI_KIT_FAL_API_KEY") or os.getenv("FAL_API_KEY") or os.getenv("FAL_KEY")
        if not key:
            raise RuntimeError("Missing FAL API key (AI_KIT_FAL_API_KEY, FAL_API_KEY, or FAL_KEY)")
        self._client = fal_client.SyncClient(key=key)
        if timeout_s is not None:
            self._client.default_timeout = float(timeout_s)

    def upload_file(self, path: Path) -> str:
        return self._client.upload_file(path)

    def subscribe(
        self,
        model: str,
        *,
        arguments: Dict[str, Any],
        with_logs: bool = False,
        on_queue_update: Optional[Callable[[Any], None]] = None,
    ) -> Any:
        return self._client.subscribe(
            model,
            arguments=arguments,
            with_logs=with_logs,
            on_queue_update=on_queue_update,
        )

    def multiview_to_3d(
        self,
        *,
        model: str,
        front_image_url: str,
        left_image_url: Optional[str] = None,
        back_image_url: Optional[str] = None,
        right_image_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"front_image_url": front_image_url}
        if left_image_url:
            payload["left_image_url"] = left_image_url
        if back_image_url:
            payload["back_image_url"] = back_image_url
        if right_image_url:
            payload["right_image_url"] = right_image_url
        if parameters:
            payload.update(parameters)

        if on_log:
            def _on_queue_update(update: Any) -> None:
                logs = getattr(update, "logs", None)
                if not logs:
                    return
                for entry in logs:
                    message = entry.get("message") if isinstance(entry, dict) else None
                    if message:
                        on_log(str(message))

            return self._client.subscribe(
                model,
                arguments=payload,
                with_logs=True,
                on_queue_update=_on_queue_update,
            )

        return self._client.subscribe(model, arguments=payload)

    def download_url(self, url: str) -> bytes:
        res = requests.get(url, timeout=120)
        res.raise_for_status()
        return res.content
