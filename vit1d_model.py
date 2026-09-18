"""Time-patch ViT for [batch, channels, 150] sensor windows."""
import torch
from torch import nn


class ViT1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels not in (12, 64):
            raise ValueError('Expected 12 IMU or 64 pressure channels')
        self.channels = channels
        self.projection = nn.Linear(5 * channels, 32)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 32))
        self.position = nn.Parameter(torch.empty(1, 31, 32))
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.position, std=.02)
        self.dropout = nn.Dropout(.15)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(32, 4, 64, .15, activation='gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(4)])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 9)

    def patchify(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.channels, 150):
            raise ValueError('Expected [B, %d, 150]' % self.channels)
        return x.transpose(1, 2).reshape(x.shape[0], 30, 5 * self.channels)

    def forward(self, x):
        z = self.projection(self.patchify(x))
        z = self.dropout(torch.cat([self.cls_token.expand(len(x), -1, -1), z], 1) + self.position)
        for layer in self.layers:
            z = layer(z)
        return self.head(self.norm(z)[:, 0])
