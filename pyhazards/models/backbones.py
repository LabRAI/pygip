import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    """Simple MLP for tabular features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNPatchEncoder(nn.Module):
    """Lightweight CNN encoder for raster patches."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


class TemporalEncoder(nn.Module):
    """GRU-based encoder for time-series signals."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.rnn(x)
        return out[:, -1, :]


__all__ = ["MLPBackbone", "CNNPatchEncoder", "TemporalEncoder"]
