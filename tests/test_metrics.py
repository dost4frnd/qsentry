import numpy as np
from qsentry.metrics import (closed_set_metrics, expected_calibration_error,
                             open_set_metrics, softmax, temperature_scale)


def test_softmax_normalized():
    z = np.random.randn(5, 4)
    p = softmax(z)
    assert np.allclose(p.sum(1), 1.0)
    assert (p >= 0).all()


def test_perfect_closed_set():
    n_classes = 3
    y = np.array([0, 1, 2, 0, 1, 2])
    logits = np.eye(n_classes)[y] * 10.0
    m = closed_set_metrics(y, logits, n_classes)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_open_set_separable():
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])
    is_anom = np.array([0, 0, 0, 1, 1, 1])
    m = open_set_metrics(is_anom, scores, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["f1_thresholded"] == 1.0


def test_ece_perfectly_calibrated_low():
    # confident & correct -> low ECE
    y = np.array([0, 1, 2, 0, 1, 2])
    logits = np.eye(3)[y] * 8.0
    cal = expected_calibration_error(y, softmax(logits), n_bins=10)
    assert cal["ece"] < 0.2
    assert len(cal["bin_accuracy"]) == 10


def test_temperature_scale_returns_positive():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((50, 4))
    y = rng.integers(0, 4, 50)
    t = temperature_scale(logits, y)
    assert t > 0
