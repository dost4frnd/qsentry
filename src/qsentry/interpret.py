"""Interpretability utilities: attention maps and latent-space projection."""
from __future__ import annotations

import numpy as np

from .datasets import Preprocessor, get_xy
from .train import extract_embeddings


def attention_profile(model, df, pre: Preprocessor, split: str = "test",
                      max_samples: int = 512, device: str = "cpu") -> dict:
    """Mean attention map (T x T) and per-timestep attention mass.

    Only valid for models exposing ``attention_maps`` (the Transformer).
    """
    import torch

    if not hasattr(model, "attention_maps"):
        raise AttributeError("model does not expose attention maps")
    x, _, _ = get_xy(df, split, pre, known_only=True)
    x = x[:max_samples]
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    model.to(device)
    attn = model.attention_maps(xt).cpu().numpy()       # (T, T)
    incoming = attn.mean(axis=0)                          # attention received
    return {"attention_map": attn.tolist(),
            "per_timestep_mass": incoming.tolist(),
            "seq_len": int(attn.shape[0])}


def tsne_embeddings(model, df, pre: Preprocessor, split: str = "test",
                    max_samples: int = 1500, perplexity: float = 30.0,
                    seed: int = 0, device: str = "cpu") -> dict:
    """2-D t-SNE of latent embeddings coloured by true class."""
    from sklearn.manifold import TSNE

    x, y, labels = get_xy(df, split, pre, known_only=True)
    if len(x) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x, y = x[idx], y[idx]
        labels = [labels[i] for i in idx]
    emb = extract_embeddings(model, x, device=device)
    perp = float(min(perplexity, max(5.0, (len(emb) - 1) / 3.0)))
    proj = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed).fit_transform(emb)
    return {"xy": proj.tolist(), "labels": labels, "y": y.tolist()}
