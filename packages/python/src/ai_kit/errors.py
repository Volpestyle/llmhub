from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorKind(str, Enum):
    UNKNOWN = "unknown_error"
    PROVIDER_AUTH = "provider_auth_error"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_NOT_FOUND = "provider_not_found"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    VALIDATION = "validation_error"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"


@dataclass
class KitErrorPayload:
    kind: ErrorKind
    message: str
    provider: Optional[str] = None
    upstreamStatus: Optional[int] = None
    upstreamCode: Optional[str] = None
    requestId: Optional[str] = None
    cause: Optional[Exception] = None


class InferenceKitError(Exception):
    def __init__(self, payload: KitErrorPayload):
        super().__init__(payload.message)
        self.kind = payload.kind
        self.provider = payload.provider
        self.upstreamStatus = payload.upstreamStatus
        self.upstreamCode = payload.upstreamCode
        self.requestId = payload.requestId
        self.cause = payload.cause


def classify_status(status: Optional[int]) -> ErrorKind:
    if status is None:
        return ErrorKind.UNKNOWN
    if status in (401, 403):
        return ErrorKind.PROVIDER_AUTH
    if status == 404:
        return ErrorKind.PROVIDER_NOT_FOUND
    if status == 429:
        return ErrorKind.PROVIDER_RATE_LIMIT
    if status >= 500:
        return ErrorKind.PROVIDER_UNAVAILABLE
    return ErrorKind.UNKNOWN


def to_kit_error(err: Exception) -> InferenceKitError:
    if isinstance(err, InferenceKitError):
        return err
    return InferenceKitError(
        KitErrorPayload(kind=ErrorKind.UNKNOWN, message=str(err), cause=err)
    )
