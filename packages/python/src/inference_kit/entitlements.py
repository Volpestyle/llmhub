from __future__ import annotations

import hashlib


def fingerprint_api_key(api_key: str | None) -> str:
    if not api_key or not api_key.strip():
        return ""
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()
