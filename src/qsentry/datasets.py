"""Domain dataset construction, preprocessing and loading.

A *domain* is a self-contained dataset (``clean``, ``drift``, ``asym`` or
``unknown``) sharing the same label semantics but differing in channel
statistics / nuisance shift.  Each domain is materialised as a flattened CSV
with ``4 + F*T`` columns:

    sample_id, domain, split, label, <ch>__t00, <ch>__t01, ...

For ``T = 32`` and ``F = 12`` this is ``4 + 384 = 388`` columns, matching the
manuscript.  Splits are disjoint by ``sample_id`` (no leakage), the scaler is
fit on the *training split only*, and the closed-set label encoder excludes the
held-out novelty class so it can be masked for closed-set scoring while still
being available for open-set evaluation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .physics import (CHANNELS, CLOSED_SET_CLASSES, N_CHANNELS, UNKNOWN_CLASS,
                      TelemetryConfig, TelemetryGenerator)
from .seeding import new_rng

META_COLS = ("sample_id", "domain", "split", "label")


# --------------------------------------------------------------------------- #
# Column layout
# --------------------------------------------------------------------------- #

def feature_columns(seq_len: int) -> list[str]:
    """Canonical (time-major) feature column names."""
    return [f"{ch}__t{t:02d}" for t in range(seq_len) for ch in CHANNELS]


def flatten(x: np.ndarray) -> np.ndarray:
    """(N, T, F) -> (N, T*F) in time-major order."""
    n = x.shape[0]
    return x.reshape(n, -1)


def unflatten(flat: np.ndarray, seq_len: int) -> np.ndarray:
    """(N, T*F) -> (N, T, F)."""
    n = flat.shape[0]
    return flat.reshape(n, seq_len, N_CHANNELS)


# --------------------------------------------------------------------------- #
# Domain specification
# --------------------------------------------------------------------------- #

@dataclass
class DomainSpec:
    name: str
    seq_len: int = 32
    n_train: int = 4480          # total windows in the train split
    n_val: int = 960
    n_test: int = 960
    drift_strength: float = 0.0
    asym_strength: float = 0.0
    attack_intensity: float = 1.0
    mixed: bool = False          # randomise severities per window (unknown)
    include_unknown_in_test: bool = False
    n_test_normal: int = 120     # only used when include_unknown_in_test
    seed: int = 1234
    extra: dict = field(default_factory=dict)


def _generator_for(spec: DomainSpec, rng: np.random.Generator,
                   per_window_mix: bool = False) -> TelemetryGenerator:
    cfg = TelemetryConfig(
        seq_len=spec.seq_len,
        drift_strength=spec.drift_strength,
        asym_strength=spec.asym_strength,
        attack_intensity=spec.attack_intensity,
    )
    if per_window_mix:
        cfg.drift_strength = float(rng.uniform(0.0, max(spec.drift_strength, 1.0)))
        cfg.asym_strength = float(rng.uniform(0.0, 3.0))
    return TelemetryGenerator(cfg)


def _emit(spec: DomainSpec, classes: list[str], counts: list[int],
          split: str, rng: np.random.Generator, start_id: int) -> tuple:
    rows_x, rows_y, ids = [], [], []
    sid = start_id
    for cls, n in zip(classes, counts):
        for _ in range(n):
            gen = _generator_for(spec, rng, per_window_mix=spec.mixed)
            rows_x.append(gen.sample_window(cls, rng))
            rows_y.append(cls)
            ids.append(sid)
            sid += 1
    x = np.stack(rows_x, axis=0)
    return x, rows_y, ids, sid


def _balanced_counts(total: int, k: int) -> list[int]:
    base = total // k
    counts = [base] * k
    for i in range(total - base * k):
        counts[i] += 1
    return counts


def build_domain(spec: DomainSpec) -> pd.DataFrame:
    """Materialise a domain as a flattened DataFrame."""
    rng = new_rng(spec.seed)
    fcols = feature_columns(spec.seq_len)
    frames = []
    next_id = 0

    closed = list(CLOSED_SET_CLASSES)

    # ---- train / val : closed-set, balanced --------------------------- #
    for split, total in (("train", spec.n_train), ("val", spec.n_val)):
        counts = _balanced_counts(total, len(closed))
        x, y, ids, next_id = _emit(spec, closed, counts, split, rng, next_id)
        df = pd.DataFrame(flatten(x), columns=fcols)
        df.insert(0, "label", y)
        df.insert(0, "split", split)
        df.insert(0, "domain", spec.name)
        df.insert(0, "sample_id", ids)
        frames.append(df)

    # ---- test --------------------------------------------------------- #
    if spec.include_unknown_in_test:
        attack_types = [c for c in closed if c != "normal"] + [UNKNOWN_CLASS]
        n_attack_total = spec.n_test - spec.n_test_normal
        per_attack = _balanced_counts(n_attack_total, len(attack_types))
        classes = ["normal"] + attack_types
        counts = [spec.n_test_normal] + per_attack
    else:
        classes = closed
        counts = _balanced_counts(spec.n_test, len(closed))

    x, y, ids, next_id = _emit(spec, classes, counts, "test", rng, next_id)
    df = pd.DataFrame(flatten(x), columns=fcols)
    df.insert(0, "label", y)
    df.insert(0, "split", "test")
    df.insert(0, "domain", spec.name)
    df.insert(0, "sample_id", ids)
    frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    return out


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

@dataclass
class Preprocessor:
    """Train-only scaler + closed-set label encoder."""

    seq_len: int = 32
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    classes_: list[str] = field(default_factory=lambda: list(CLOSED_SET_CLASSES))

    @property
    def feature_cols(self) -> list[str]:
        return feature_columns(self.seq_len)

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        x = train_df[self.feature_cols].to_numpy(dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1e-8
        return self

    # -- label helpers -------------------------------------------------- #
    def encode_labels(self, labels) -> np.ndarray:
        """Map labels to indices; novelty / unknown -> -1."""
        lut = {c: i for i, c in enumerate(self.classes_)}
        return np.array([lut.get(l, -1) for l in labels], dtype=np.int64)

    def is_known(self, labels) -> np.ndarray:
        known = set(self.classes_)
        return np.array([l in known for l in labels], dtype=bool)

    def normal_index(self) -> int:
        return self.classes_.index("normal")

    # -- feature transform ---------------------------------------------- #
    def transform(self, df: pd.DataFrame, as_sequence: bool = True) -> np.ndarray:
        x = df[self.feature_cols].to_numpy(dtype=np.float64)
        x = (x - self.mean_) / self.std_
        x = x.astype(np.float32)
        if as_sequence:
            x = unflatten(x, self.seq_len)
        return x

    # -- persistence ---------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, std=self.std_,
                 classes=np.array(self.classes_, dtype=object),
                 seq_len=self.seq_len)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        d = np.load(path, allow_pickle=True)
        return cls(seq_len=int(d["seq_len"]), mean_=d["mean"], std_=d["std"],
                   classes_=list(d["classes"]))


# --------------------------------------------------------------------------- #
# IO + loaders
# --------------------------------------------------------------------------- #

def save_domain(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_domain(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def get_xy(df: pd.DataFrame, split: str, pre: Preprocessor,
           known_only: bool = False):
    """Return ``(X, y_idx, labels)`` for one split.

    ``X`` is ``(N, T, F)``, ``y_idx`` are closed-set indices (novelty == -1)
    and ``labels`` are the raw string labels.
    """
    sub = split_frame(df, split)
    if known_only:
        sub = sub[pre.is_known(sub["label"].tolist())].reset_index(drop=True)
    x = pre.transform(sub, as_sequence=True)
    labels = sub["label"].tolist()
    y = pre.encode_labels(labels)
    return x, y, labels


def audit(df: pd.DataFrame, pre: Preprocessor) -> dict:
    """Structural integrity report (duplicates / NaNs / Inf / leakage)."""
    fcols = pre.feature_cols
    x = df[fcols].to_numpy(dtype=np.float64)
    # leakage: a sample_id must belong to exactly one split
    leak = df.groupby("sample_id")["split"].nunique()
    rep = {
        "rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_feature_columns": int(len(fcols)),
        "seq_len": int(pre.seq_len),
        "n_channels": int(N_CHANNELS),
        "duplicate_rows": int(df.duplicated(subset=fcols).sum()),
        "missing_values": int(df[fcols].isna().sum().sum()),
        "infinite_values": int(np.isinf(x).sum()),
        "split_leakage_ids": int((leak > 1).sum()),
        "class_counts": {k: int(v) for k, v in
                         df["label"].value_counts().sort_index().items()},
        "split_counts": {k: int(v) for k, v in
                         df["split"].value_counts().items()},
    }
    return rep


def write_audit(rep: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
