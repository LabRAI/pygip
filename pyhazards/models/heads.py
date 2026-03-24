import torch.nn as nn


class ClassificationHead(nn.Module):
    """Simple classification head."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class RegressionHead(nn.Module):
    """Regression head for scalar or multi-target outputs."""

    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class SegmentationHead(nn.Module):
    """Segmentation head for raster masks."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


__all__ = ["ClassificationHead", "RegressionHead", "SegmentationHead"]
