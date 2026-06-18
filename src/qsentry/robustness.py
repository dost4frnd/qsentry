"""Robustness severity sweeps.

For a trained closed-set detector we synthesise fresh test domains at a range
of shift severities (phase-drift amplitude, asymmetric-loss magnitude, attack
intensity) and record how detection quality degrades.  This produces the
robustness curves in the manuscript and quantifies graceful degradation.
"""
from __future__ import annotations

import numpy as np

from .datasets import (DomainSpec, Preprocessor, build_domain, feature_columns,
                       split_frame)
from .metrics import softmax
from .train import predict_logits


def _macro_f1(model, df, pre, device):
    from sklearn.metrics import f1_score
    sub = split_frame(df, "test")
    sub = sub[pre.is_known(sub["label"].tolist())].reset_index(drop=True)
    x = pre.transform(sub, as_sequence=True)
    y = pre.encode_labels(sub["label"].tolist())
    pred = softmax(predict_logits(model, x, device=device)).argmax(1)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def severity_sweep(model, pre: Preprocessor, axis: str, levels,
                   seq_len: int = 32, n_test: int = 480, seed: int = 7,
                   device: str = "cpu") -> dict:
    """Sweep one shift ``axis`` across ``levels`` and record macro-F1.

    axis in {"drift_strength", "asym_strength", "attack_intensity"}.
    """
    assert axis in {"drift_strength", "asym_strength", "attack_intensity"}
    results = []
    for i, level in enumerate(levels):
        kwargs = dict(drift_strength=0.0, asym_strength=0.0,
                      attack_intensity=1.0)
        kwargs[axis] = float(level)
        spec = DomainSpec(name=f"{axis}_{level}", seq_len=seq_len,
                          n_train=8, n_val=8, n_test=n_test,
                          seed=seed + i * 17, **kwargs)
        df = build_domain(spec)
        f1 = _macro_f1(model, df, pre, device)
        results.append({"level": float(level), "f1_macro": f1})
    return {"axis": axis, "points": results}


def run_all_sweeps(model, pre: Preprocessor, sweep_cfg: dict,
                   seq_len: int = 32, device: str = "cpu") -> dict:
    out = {}
    for axis, levels in sweep_cfg.items():
        out[axis] = severity_sweep(model, pre, axis, levels, seq_len=seq_len,
                                   device=device)
    return out
