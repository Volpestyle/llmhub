from __future__ import annotations

from typing import Dict, List

_TASK_ALLOWLISTS: Dict[str, Dict[str, List[str]]] = {
    "transcribe": {
        "openai": [
            "whisper-1",
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
        ],
        "local": [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ],
    }
}


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def list_task_models(task: str, provider: str) -> list[str]:
    task_key = _normalize(task)
    provider_key = _normalize(provider)
    models = _TASK_ALLOWLISTS.get(task_key, {}).get(provider_key, [])
    return list(models)


def list_transcribe_models(provider: str) -> list[str]:
    return list_task_models("transcribe", provider)
