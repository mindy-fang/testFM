"""Temporal MAE adaptation; not the original TransfHAR implementation."""
import torch
from torch import nn

from transformer_model import SensorEncoder


def sample_mask(batch, generator, mode=None):
    modes = torch.randint(0, 2, (batch,), generator=generator) if mode is None else torch.full((batch,), mode)
    random_order = torch.rand(batch, 150, generator=generator).argsort(1)
    mask = torch.zeros(batch, 150, dtype=torch.bool)
    mask.scatter_(1, random_order[:, :98], True)
    starts = torch.randint(0, 53, (batch,), generator=generator)
    contiguous = (torch.arange(150)[None, :] >= starts[:, None]) & (torch.arange(150)[None, :] < starts[:, None]+98)
    mask = torch.where((modes == 1)[:, None], contiguous, mask)
    visible = torch.arange(150).expand(batch, -1)[~mask].reshape(batch, 52)
    return mask, visible


def masked_mse(reconstruction, target, mask):
    return (reconstruction-target).square()[mask].mean()


class TemporalMAE(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = SensorEncoder() if encoder is None else encoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 32))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.15,
            activation='relu', batch_first=True, norm_first=False)
        self.reconstruct = nn.Linear(32, 76)

    def forward(self, x, visible):
        tokens = self.encoder.forward_tokens(x, visible)
        restored = self.mask_token.expand(x.shape[0], 150, 32).clone()
        restored.scatter_(1, visible.unsqueeze(-1).expand(-1, -1, 32), tokens)
        restored = restored + self.encoder.position_encoding
        return self.reconstruct(self.decoder(restored))


class ActivityClassifier(nn.Module):
    def __init__(self, encoder, head, frozen=False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.frozen = frozen
        if frozen:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        return self.head(self.encoder(x))
