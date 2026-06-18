"""Transformer encoder classifier for telemetry sequences.

A compact encoder is implemented explicitly (rather than via
``nn.TransformerEncoder``) so that per-layer attention maps and the pooled
latent embedding can be extracted for the interpretability figures
(attention heatmap, t-SNE).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, attn_w = self.attn(x, x, x, need_weights=True,
                                     average_attn_weights=True)
        self.last_attn = attn_w.detach()
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_ff: int = 128,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_ff, dropout)
             for _ in range(num_layers)])
        self.head = nn.Linear(d_model, n_classes)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        h = self.pos(self.input_proj(x))
        for layer in self.layers:
            h = layer(h)
        emb = h.mean(dim=1)                # global temporal pooling
        logits = self.head(emb)
        if return_embedding:
            return logits, emb
        return logits

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean attention map ``(T, T)`` averaged over the batch
        and the last encoder layer."""
        self.eval()
        _ = self.forward(x)
        attn = self.layers[-1].last_attn      # (B, T, T)
        return attn.mean(dim=0)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, emb = self.forward(x, return_embedding=True)
        return emb

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
