from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Strategy = Literal["auto", "ddp", "dp", "none"]


@dataclass
class DistributedConfig:
    strategy: Strategy = "auto"
    devices: int | None = None


def select_strategy(prefer: Strategy = "auto") -> Strategy:
    if prefer == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return "ddp"
        if torch.cuda.is_available():
            return "none"
        return "none"
    return prefer
