from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LocalModelSpec:
    task: str
    id: str
    hf_repo: str
    default: bool = False


class LocalModelRegistry:
    def __init__(self) -> None:
        self._models: Dict[str, Dict[str, LocalModelSpec]] = {}
        self._defaults: Dict[str, str] = {}

    def register(
        self,
        task: str,
        model_id: str,
        hf_repo: str,
        *,
        default: bool = False,
        replace: bool = False,
    ) -> LocalModelSpec:
        task_models = self._models.setdefault(task, {})
        if model_id in task_models and not replace:
            raise ValueError(f"Model '{model_id}' already registered for task '{task}'")
        spec = LocalModelSpec(task=task, id=model_id, hf_repo=hf_repo, default=default)
        task_models[model_id] = spec
        if default or task not in self._defaults:
            self._defaults[task] = model_id
        return spec

    def resolve(self, task: str, model_id: Optional[str]) -> LocalModelSpec:
        task_models = self._models.get(task, {})
        if model_id:
            spec = task_models.get(model_id)
            if spec:
                return spec
            return LocalModelSpec(task=task, id=model_id, hf_repo=model_id, default=False)
        default_id = self._defaults.get(task)
        if not default_id or default_id not in task_models:
            raise ValueError(f"No default model registered for task '{task}'")
        return task_models[default_id]

    def list(self, task: Optional[str] = None) -> List[LocalModelSpec]:
        if task is not None:
            return list(self._models.get(task, {}).values())
        specs: List[LocalModelSpec] = []
        for models in self._models.values():
            specs.extend(models.values())
        return specs

    def tasks(self) -> Iterable[str]:
        return self._models.keys()


REGISTRY = LocalModelRegistry()
