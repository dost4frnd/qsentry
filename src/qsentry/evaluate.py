"""High-level evaluation routines used by the analysis scripts.

These functions operate on a *trained* model plus a fitted
:class:`~qsentry.datasets.Preprocessor` and a domain DataFrame, and return
plain dictionaries / arrays that the scripts serialise to ``results/``.
"""
from __future__ import annotations

import numpy as np

from .datasets import Preprocessor, get_xy, split_frame
from .metrics import (closed_set_metrics, confusion, open_set_metrics,
                      softmax)
from .physics import CHANNELS, N_CHANNELS
from .train import predict_logits, reconstruction_scores


# --------------------------------------------------------------------------- #
# Closed-set
# --------------------------------------------------------------------------- #

def closed_set_report(model, df, pre: Preprocessor, split: str = "test",
                      device: str = "cpu") -> dict:
    """Closed-set metrics + confusion matrix on known-label test samples."""
    x, y, _ = get_xy(df, split, pre, known_only=True)
    logits = predict_logits(model, x, device=device)
    n_classes = len(pre.classes_)
    metrics = closed_set_metrics(y, logits, n_classes)
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)
    cm = confusion(y, y_pred, n_classes)
    return {
        "metrics": metrics,
        "confusion": cm.tolist(),
        "classes": list(pre.classes_),
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "n": int(len(y)),
    }


def cross_domain_report(model, domain_frames: dict, pre: Preprocessor,
                        split: str = "test", device: str = "cpu") -> dict:
    """Evaluate one trained model on every domain's test split."""
    rows = {}
    for name, df in domain_frames.items():
        rep = closed_set_report(model, df, pre, split=split, device=device)
        rows[name] = rep["metrics"]
    return rows


# --------------------------------------------------------------------------- #
# Open-set / anomaly
# --------------------------------------------------------------------------- #

def anomaly_scores(ae, df, pre: Preprocessor, split: str,
                   device: str = "cpu"):
    """Reconstruction scores + binary anomaly labels (1 == not normal)."""
    sub = split_frame(df, split)
    x = pre.transform(sub, as_sequence=True)
    scores = reconstruction_scores(ae, x, device=device)
    is_anom = (sub["label"].to_numpy() != "normal").astype(int)
    return scores, is_anom, sub["label"].tolist()


def calibrate_threshold(ae, train_df, pre: Preprocessor, q: float = 0.95,
                        device: str = "cpu") -> float:
    """tau = Q_q of reconstruction error on normal training windows."""
    sub = split_frame(train_df, "train")
    sub = sub[sub["label"] == "normal"]
    x = pre.transform(sub, as_sequence=True)
    s = reconstruction_scores(ae, x, device=device)
    return float(np.quantile(s, q))


def open_set_report(ae, domain_frames: dict, pre: Preprocessor,
                    threshold: float, split: str = "test",
                    device: str = "cpu") -> dict:
    rows = {}
    for name, df in domain_frames.items():
        scores, is_anom, _ = anomaly_scores(ae, df, pre, split, device=device)
        rows[name] = open_set_metrics(is_anom, scores, threshold)
    return rows


# --------------------------------------------------------------------------- #
# Channel importance (permutation)
# --------------------------------------------------------------------------- #

def permutation_channel_importance(model, df, pre: Preprocessor,
                                   split: str = "test", n_repeats: int = 5,
                                   seed: int = 0, device: str = "cpu") -> dict:
    """Macro-F1 drop when each physical channel is permuted across samples.

    A larger drop means the model relies more on that channel -- an
    interpretable, model-agnostic importance measure.
    """
    from sklearn.metrics import f1_score

    x, y, _ = get_xy(df, split, pre, known_only=True)
    base_pred = softmax(predict_logits(model, x, device=device)).argmax(1)
    base_f1 = f1_score(y, base_pred, average="macro", zero_division=0)

    rng = np.random.default_rng(seed)
    importances, stds = {}, {}
    for f in range(N_CHANNELS):
        drops = []
        for _ in range(n_repeats):
            xp = x.copy()
            perm = rng.permutation(xp.shape[0])
            xp[:, :, f] = xp[perm, :, f]            # break channel<->label link
            pred = softmax(predict_logits(model, xp, device=device)).argmax(1)
            f1 = f1_score(y, pred, average="macro", zero_division=0)
            drops.append(base_f1 - f1)
        importances[CHANNELS[f]] = float(np.mean(drops))
        stds[CHANNELS[f]] = float(np.std(drops))
    return {"base_f1": float(base_f1), "importance": importances,
            "importance_std": stds}
