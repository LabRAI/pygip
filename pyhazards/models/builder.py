from __future__ import annotations

import inspect
from typing import Any, Dict

import torch.nn as nn

from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .heads import ClassificationHead, RegressionHead, SegmentationHead
from .registry import get_model_config


def build_model(name: str, task: str, **kwargs: Any) -> nn.Module:
    """
    Build a model by name and task.
    This delegates to registry metadata to keep a consistent interface.
    """
    cfg = get_model_config(name)
    if cfg is None:
        raise KeyError(f"Model '{name}' is not registered.")

    task = task.lower()
    builder = cfg["builder"]
    defaults: Dict[str, Any] = cfg.get("defaults", {})
    merged = {**defaults, **kwargs, "task": task}

    # Some builders (e.g., default_builder) require `name`, while others don't.
    # Pass `name` only when the callable can accept it.
    sig = inspect.signature(builder)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if "name" in params or accepts_kwargs:
        return builder(**{**merged, "name": name})
    return builder(**merged)


def default_builder(name: str, task: str, **kwargs: Any) -> nn.Module:
    """
    Generic builder for standard backbones + heads.
    """
    task = task.lower()
    if name == "mlp":
        backbone = MLPBackbone(kwargs["in_dim"], hidden_dim=kwargs.get("hidden_dim", 256), depth=kwargs.get("depth", 2))
        head = _make_head(task, kwargs)
        return _combine(backbone, head)
    if name == "cnn":
        backbone = CNNPatchEncoder(kwargs.get("in_channels", 3), hidden_dim=kwargs.get("hidden_dim", 64))
        head = _make_head(task, kwargs, backbone_out_dim=kwargs.get("hidden_dim", 64))
        return _combine(backbone, head)
    if name == "temporal":
        backbone = TemporalEncoder(kwargs["in_dim"], hidden_dim=kwargs.get("hidden_dim", 128), num_layers=kwargs.get("num_layers", 1))
        head = _make_head(task, kwargs)
        return _combine(backbone, head)
    raise ValueError(f"Unknown backbone '{name}'.")


def _make_head(task: str, kwargs: Dict[str, Any], backbone_out_dim: int | None = None) -> nn.Module:
    if task == "classification":
        in_dim = backbone_out_dim or kwargs.get("hidden_dim") or kwargs["in_dim"]
        return ClassificationHead(in_dim=in_dim, num_classes=kwargs["out_dim"])
    if task == "regression":
        in_dim = backbone_out_dim or kwargs.get("hidden_dim") or kwargs["in_dim"]
        return RegressionHead(in_dim=in_dim, out_dim=kwargs.get("out_dim", 1))
    if task == "segmentation":
        in_channels = kwargs.get("hidden_dim") or backbone_out_dim or kwargs.get("in_channels", 1)
        return SegmentationHead(in_channels=in_channels, num_classes=kwargs["out_dim"])
    raise ValueError(f"Unsupported task '{task}'.")


def _combine(backbone: nn.Module, head: nn.Module) -> nn.Module:
    return nn.Sequential(backbone, head)


__all__ = ["build_model", "default_builder"]
