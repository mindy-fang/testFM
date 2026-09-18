"""Small Transformer encoder for the same 76 x 150 sensor windows."""
import math

import torch
from torch import nn


class SensorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(76, 32)
        position = torch.arange(150, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, 32, 2, dtype=torch.float32) * (-math.log(10000.0) / 32))
        encoding = torch.zeros(150, 32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer('position_encoding', encoding.unsqueeze(0))
        # Construct layers independently rather than cloning an identical initialization.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                dropout=0.15, activation='relu', batch_first=True, norm_first=False)
            for _ in range(4)
        ])
    def forward_tokens(self, x, visible_indices=None):
        z = self.projection(x.transpose(1, 2))
        z = z + self.position_encoding[:, :z.shape[1]]
        if visible_indices is not None:
            z = z.gather(1, visible_indices.unsqueeze(-1).expand(-1, -1, z.shape[-1]))
        for layer in self.layers:
            z = layer(z)
        return z

    def forward(self, x):
        return self.forward_tokens(x).mean(dim=1)


class SensorTransformer(SensorEncoder):
    """Retains legacy state-dict keys and initialization order."""
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 9)

    def forward(self, x):
        return self.head(super().forward(x))
