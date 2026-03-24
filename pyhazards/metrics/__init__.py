from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class MetricBase(ABC):
    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        ...

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class ClassificationMetrics(MetricBase):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        pred_labels = preds.argmax(dim=-1)
        acc = (pred_labels == targets).float().mean().item()
        return {"Acc": acc}


class RegressionMetrics(MetricBase):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        mae = F.l1_loss(preds, targets).item()
        rmse = torch.sqrt(F.mse_loss(preds, targets)).item()
        return {"MAE": mae, "RMSE": rmse}


class SegmentationMetrics(MetricBase):
    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        pred_labels = preds.argmax(dim=1)
        # simple pixel accuracy; extend to IoU/Dice as needed
        acc = (pred_labels == targets).float().mean().item()
        return {"PixelAcc": acc}


__all__ = ["MetricBase", "ClassificationMetrics", "RegressionMetrics", "SegmentationMetrics"]
