from typing import Any, Callable, Dict

from .base import Dataset

_DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str, builder: Callable[..., Dataset]) -> None:
    if name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' already registered.")
    _DATASET_REGISTRY[name] = builder


def available_datasets():
    return sorted(_DATASET_REGISTRY.keys())


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' is not registered. Known: {available_datasets()}")
    return _DATASET_REGISTRY[name](**kwargs)
