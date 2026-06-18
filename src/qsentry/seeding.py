"""Deterministic seeding across Python, NumPy and PyTorch."""
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def seed_everything(seed: int = 1234, deterministic: bool = True) -> int:
    """Seed all relevant RNGs and return the seed.

    Parameters
    ----------
    seed:
        The integer seed.
    deterministic:
        If ``True`` request deterministic CuDNN/torch behaviour. This makes
        results reproducible at a small performance cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def new_rng(seed: int) -> np.random.Generator:
    """Return a fresh, independent NumPy ``Generator``."""
    return np.random.default_rng(seed)
