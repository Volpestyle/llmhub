from __future__ import annotations

import base64
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional


def decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        payload = value.split(",", 1)[1]
        return base64.b64decode(payload)
    return base64.b64decode(value)


def write_temp_file(data: str, suffix: str) -> Path:
    raw = decode_base64(data)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp.write(raw)
        temp.flush()
    finally:
        temp.close()
    return Path(temp.name)


def data_url_media_type(value: str) -> Optional[str]:
    if not value.startswith("data:"):
        return None
    header = value.split(",", 1)[0]
    if ";" in header:
        header = header.split(";", 1)[0]
    return header[5:] if header.startswith("data:") else None


def guess_extension(media_type: Optional[str], default: str = ".png") -> str:
    if not media_type:
        return default
    suffix = mimetypes.guess_extension(media_type)
    if suffix:
        return suffix
    if media_type == "image/jpg":
        return ".jpg"
    if media_type in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if media_type in {"audio/wav", "audio/x-wav"}:
        return ".wav"
    return default
