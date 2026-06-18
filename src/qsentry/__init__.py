"""QSentry: a telemetry-driven intelligent security layer for resilient
quantum communication networks.

The package bundles:
  * a physically-grounded TF-QKD telemetry generator (``physics``),
  * domain-shift dataset construction (``datasets``),
  * four sequence model families -- Transformer, LSTM, hybrid QLSTM and a
    one-class Autoencoder (``models``),
  * training, evaluation, calibration, robustness and operating-point
    analysis utilities,
  * publication-quality figure and LaTeX-table generation.

See ``docs/REPRODUCIBILITY.md`` for the end-to-end workflow.
"""

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:  # pragma: no cover - trivial
    __version__ = _metadata.version("qsentry")
except Exception:  # not installed as a distribution
    __version__ = "1.0.0"
