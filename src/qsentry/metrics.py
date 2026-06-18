"""Evaluation metrics for the two-task (closed-set / open-set) protocol.

Closed-set: accuracy, macro precision/recall/F1, one-vs-rest ROC-AUC.
Open-set : ROC-AUC, average precision, thresholded F1 at tau.
Trust    : expected calibration error (ECE) + reliability bins, Brier score.
Operating: ROC / DET / PR curves, alarm-rate vs detection-rate.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Closed-set
# --------------------------------------------------------------------------- #

def closed_set_metrics(y_true: np.ndarray, logits: np.ndarray,
                       n_classes: int) -> dict:
    """Closed-set classification metrics on known-label samples."""
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred,
                                                 average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
    }
    # OVR macro ROC-AUC (guard against missing classes in y_true)
    try:
        labels = list(range(n_classes))
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                          labels=labels))
    except ValueError:
        out["roc_auc_ovr_macro"] = float("nan")
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------- #
# Open-set / anomaly
# --------------------------------------------------------------------------- #

def open_set_metrics(is_anomaly: np.ndarray, scores: np.ndarray,
                     threshold: float) -> dict:
    """Anomaly metrics given binary labels (1 == attack/novel) and scores."""
    y_pred = (scores > threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(is_anomaly, scores)),
        "average_precision": float(average_precision_score(is_anomaly, scores)),
        "f1_thresholded": float(f1_score(is_anomaly, y_pred, zero_division=0)),
        "precision_thresholded": float(precision_score(is_anomaly, y_pred,
                                                       zero_division=0)),
        "recall_thresholded": float(recall_score(is_anomaly, y_pred,
                                                 zero_division=0)),
        "threshold": float(threshold),
    }
    return out


def roc_points(is_anomaly: np.ndarray, scores: np.ndarray):
    fpr, tpr, thr = roc_curve(is_anomaly, scores)
    return fpr, tpr, thr


def pr_points(is_anomaly: np.ndarray, scores: np.ndarray):
    prec, rec, thr = precision_recall_curve(is_anomaly, scores)
    return prec, rec, thr


def det_points(is_anomaly: np.ndarray, scores: np.ndarray):
    """Detection-error-tradeoff points: (false-alarm-rate, miss-rate)."""
    fpr, tpr, thr = roc_curve(is_anomaly, scores)
    return fpr, 1.0 - tpr, thr


def alarm_detection_curve(is_anomaly: np.ndarray, scores: np.ndarray):
    """Return (threshold, alarm_rate, detection_rate) swept over scores."""
    order = np.argsort(scores)
    thr = np.unique(scores[order])
    alarm, detect = [], []
    n = len(scores)
    pos = is_anomaly.sum()
    for t in thr:
        pred = scores > t
        alarm.append(pred.sum() / n)
        detect.append((pred & (is_anomaly == 1)).sum() / max(pos, 1))
    return thr, np.array(alarm), np.array(detect)


# --------------------------------------------------------------------------- #
# Calibration / trust
# --------------------------------------------------------------------------- #

def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                               n_bins: int = 15) -> dict:
    """Top-label ECE with reliability-diagram bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    bin_conf, bin_acc, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        bin_count.append(cnt)
        if cnt == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        bin_conf.append(avg_conf)
        bin_acc.append(avg_acc)
        ece += (cnt / n) * abs(avg_acc - avg_conf)
    return {
        "ece": float(ece),
        "bin_edges": bins.tolist(),
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_count": bin_count,
    }


def brier_multiclass(y_true: np.ndarray, probs: np.ndarray,
                     n_classes: int) -> float:
    """Mean one-vs-rest Brier score."""
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def temperature_scale(logits: np.ndarray, y_true: np.ndarray,
                      grid: np.ndarray | None = None) -> float:
    """Fit a single temperature ``T`` minimising NLL on a held-out set."""
    if grid is None:
        grid = np.linspace(0.5, 5.0, 91)
    best_t, best_nll = 1.0, np.inf
    onehot = None
    for t in grid:
        probs = softmax(logits / t)
        idx = (np.arange(len(y_true)), y_true)
        nll = -np.mean(np.log(np.clip(probs[idx], 1e-12, 1.0)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t
