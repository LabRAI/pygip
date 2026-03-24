from __future__ import annotations

from typing import Any, Callable, Iterable, List

import torch


class SlidingWindowInference:
    """
    Placeholder for sliding-window inference over large rasters or grids.
    Implement windowing logic and stitching as needed.
    """

    def __init__(self, model: torch.nn.Module, window_fn: Callable[..., Iterable[Any]] | None = None):
        self.model = model
        self.window_fn = window_fn

    def __call__(self, inputs: Any) -> List[torch.Tensor]:
        if self.window_fn is None:
            raise NotImplementedError("Provide a window_fn to generate windows from inputs.")
        outputs: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for window in self.window_fn(inputs):
                outputs.append(self.model(window))
        return outputs
