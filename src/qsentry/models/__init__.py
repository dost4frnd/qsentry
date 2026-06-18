"""Model registry / factory."""
from __future__ import annotations

from .autoencoder import SequenceAutoencoder, percentile_threshold
from .lstm import LSTMClassifier
from .qlstm import QLSTMClassifier, to_qiskit_circuit
from .transformer import TransformerClassifier

SUPERVISED = {
    "transformer": TransformerClassifier,
    "lstm": LSTMClassifier,
    "qlstm": QLSTMClassifier,
}

ALL = dict(SUPERVISED)
ALL["autoencoder"] = SequenceAutoencoder


def build_model(name: str, *, n_features: int, seq_len: int, n_classes: int,
                params: dict | None = None):
    """Instantiate a model by registry ``name``."""
    params = dict(params or {})
    name = name.lower()
    if name not in ALL:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(ALL)}")
    cls = ALL[name]
    if name == "autoencoder":
        return cls(n_features=n_features, seq_len=seq_len, **params)
    return cls(n_features=n_features, n_classes=n_classes, **params)


__all__ = [
    "build_model", "SUPERVISED", "ALL",
    "TransformerClassifier", "LSTMClassifier", "QLSTMClassifier",
    "SequenceAutoencoder", "percentile_threshold", "to_qiskit_circuit",
]
