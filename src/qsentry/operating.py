"""Deployment-oriented operating analysis.

Measures inference latency / throughput and parameter budgets for each model,
which feed the feasibility discussion (a security layer must keep up with the
control-plane telemetry rate).  DET / operating curves themselves are computed
from the open-set scores via :mod:`qsentry.metrics`.
"""
from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def benchmark_latency(model, x_sample: np.ndarray, batch_size: int = 1,
                      n_warmup: int = 5, n_iter: int = 50,
                      device: str = "cpu") -> dict:
    """Per-window inference latency (ms) and throughput (windows/s)."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    xb = torch.as_tensor(x_sample[:batch_size], dtype=torch.float32, device=dev)
    is_ae = hasattr(model, "reconstruction_error")

    def _run():
        if is_ae:
            return model.reconstruction_error(xb)
        return model(xb)

    for _ in range(n_warmup):
        _run()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _run()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    per_batch_ms = float(times.mean() * 1e3)
    per_window_ms = per_batch_ms / batch_size
    return {
        "batch_size": int(batch_size),
        "latency_ms_per_batch": per_batch_ms,
        "latency_ms_per_window": per_window_ms,
        "throughput_windows_per_s": float(1000.0 / per_window_ms),
        "latency_ms_std": float(times.std() * 1e3),
    }


def parameter_budget(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters_total": int(total),
            "parameters_trainable": int(trainable)}


def profile_model(model, x_sample: np.ndarray, batch_size: int = 1,
                  device: str = "cpu") -> dict:
    out = parameter_budget(model)
    out.update(benchmark_latency(model, x_sample, batch_size=batch_size,
                                 device=device))
    return out
