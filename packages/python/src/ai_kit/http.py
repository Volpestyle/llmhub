from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .errors import HubErrorPayload, InferenceKitError, classify_status


def request_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    response = requests.request(
        method,
        url,
        headers=headers,
        json=json_body,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise InferenceKitError(
            HubErrorPayload(
                kind=classify_status(response.status_code),
                message=response.text,
                upstreamStatus=response.status_code,
            )
        )
    return response.json()


def request_stream(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
):
    response = requests.request(
        method,
        url,
        headers=headers,
        json=json_body,
        timeout=timeout,
        stream=True,
    )
    if response.status_code >= 400:
        raise InferenceKitError(
            HubErrorPayload(
                kind=classify_status(response.status_code),
                message=response.text,
                upstreamStatus=response.status_code,
            )
        )
    return response
