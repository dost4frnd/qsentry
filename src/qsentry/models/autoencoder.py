"""One-class sequence Autoencoder used as an open-set anomaly scorer.

Trained on *normal* telemetry only.  The per-window reconstruction error
``s(x) = ||x - x_hat||_2^2`` (Eq. 2) is used for ranking; a 95th-percentile
threshold on the normal-training error provides a thresholded decision.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SequenceAutoencoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, latent_dim: int = 16,
                 hidden: int = 128, dropout: float = 0.0, **_):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        in_dim = n_features * seq_len
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        z = self.encoder(flat)
        rec = self.decoder(z)
        return rec.reshape(b, self.seq_len, self.n_features)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        rec = self.forward(x)
        return ((x - rec) ** 2).reshape(x.shape[0], -1).sum(dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def percentile_threshold(scores: np.ndarray, q: float = 0.95) -> float:
    """Reconstruction threshold tau = Q_q of the normal-training errors."""
    return float(np.quantile(scores, q))
