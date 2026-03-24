"""
Reusable transforms for preprocessing hazard datasets.
Currently placeholders; implement normalization, index computation, temporal windowing, etc.
"""

from typing import Callable

from ..base import DataBundle

TransformFn = Callable[[DataBundle], DataBundle]

__all__ = ["TransformFn"]
