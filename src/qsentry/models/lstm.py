"""LSTM recurrent classifier (classical recurrent baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.1,
                 bidirectional: bool = False, **_):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        out, (h, _) = self.lstm(x)
        emb = out[:, -1, :]               # last timestep representation
        logits = self.head(self.dropout(emb))
        if return_embedding:
            return logits, emb
        return logits

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, emb = self.forward(x, return_embedding=True)
        return emb

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
