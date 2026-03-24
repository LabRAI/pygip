from __future__ import annotations

import os
from typing import Optional

import torch

_DEFAULT_DEVICE_STR = os.getenv("PYHAZARDS_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
_default_device = torch.device(_DEFAULT_DEVICE_STR)


def auto_device(prefer: str | None = None) -> torch.device:
    """
    Choose a device automatically. Respects PYHAZARDS_DEVICE and prefer flag.
    """
    if prefer:
        return torch.device(prefer)
    return _default_device


def num_devices() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_device() -> torch.device:
    return _default_device


def set_device(device_str: str | torch.device) -> None:
    global _default_device
    _default_device = torch.device(device_str)
