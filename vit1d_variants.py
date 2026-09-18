"""Fusion and raw-pressure variants used in the Twente experiments."""
from torch import nn
from vit1d_model import ViT1D


class FusionViT1D(ViT1D):
    def __init__(self, channels=76):
        if channels != 76:
            raise ValueError('Fusion input requires 76 channels')
        super().__init__(64)
        self.channels = 76
        self.projection = nn.Linear(5 * 76, 32)


class NoRatioViT1D(ViT1D):
    def __init__(self, channels):
        if channels not in (32, 44):
            raise ValueError('Expected 32 pressure or 44 pressure+IMU channels')
        super().__init__(64)
        self.channels = channels
        self.projection = nn.Linear(5 * channels, 32)
