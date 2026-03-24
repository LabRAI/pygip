from typing import Any, Callable, Dict, Optional

import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(name: str, builder: Callable[..., nn.Module], defaults: Optional[Dict[str, Any]] = None) -> None:
    if name in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' already registered.")
    _MODEL_REGISTRY[name] = {"builder": builder, "defaults": defaults or {}}


def available_models():
    return sorted(_MODEL_REGISTRY.keys())


def get_model_config(name: str) -> Optional[Dict[str, Any]]:
    return _MODEL_REGISTRY.get(name)


__all__ = ["register_model", "available_models", "get_model_config"]
