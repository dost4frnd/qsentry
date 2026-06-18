"""Publication-quality figure generation (matplotlib).

Every function writes a vector PDF into the figures directory and returns the
path.  A restrained, consistent house style is applied globally.  Figures are
grouped into: system/data schematics, closed-set results, cross-domain,
open-set/anomaly, calibration, robustness, operating points and
interpretability.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .physics import CHANNELS

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

PALETTE = ["#1b6ca8", "#d1495b", "#2e8b57", "#e0a800", "#6a4c93",
           "#117a8b", "#bc5090", "#5a5a5a", "#ff7f0e"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


def _save(fig, figdir, name) -> str:
    Path(figdir).mkdir(parents=True, exist_ok=True)
    path = str(Path(figdir) / f"{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def _short(label: str) -> str:
    return (label.replace("_attack", "").replace("_", " ")
            .replace("synchronization", "sync").title())


# --------------------------------------------------------------------------- #
# 1. System architecture schematic
# --------------------------------------------------------------------------- #

def architecture_diagram(figdir: str, name: str = "fig_architecture") -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text, color):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.3, edgecolor=color, facecolor=color + "22")
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.2, color="#444"))

    box(0.2, 2.0, 2.2, 1.1, "Quantum link\n(TF-QKD nodes)", PALETTE[0])
    box(2.9, 2.0, 2.1, 1.1, "Control-plane\ntelemetry\n(12 channels)", PALETTE[5])
    box(5.5, 3.1, 2.5, 1.0, "Stage 1:\nclosed-set detector\n(Transformer)", PALETTE[2])
    box(5.5, 0.9, 2.5, 1.0, "Stage 2:\nopen-set novelty\n(autoencoder)", PALETTE[1])
    box(8.6, 2.0, 3.0, 1.1, "Trust layer:\ncalibrated scores +\nalarm / triage", PALETTE[4])

    arrow(2.4, 2.55, 2.9, 2.55)
    arrow(5.0, 2.75, 5.5, 3.5)
    arrow(5.0, 2.35, 5.5, 1.4)
    arrow(8.0, 3.5, 8.6, 2.8)
    arrow(8.0, 1.4, 8.6, 2.3)
    ax.text(6.0, 4.55, "QSentry security layer", fontsize=11, fontweight="bold")
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 2. Telemetry signatures
# --------------------------------------------------------------------------- #

def telemetry_signatures(samples: dict, channels, figdir: str,
                         name: str = "fig_telemetry_signatures") -> str:
    """``samples``: {class_name: (T, F) array}; plot a few key channels."""
    set_style()
    show = [c for c in channels if c in
            ("phase_lock_error_rad", "qber_phase", "reference_power",
             "detector_count_rate")]
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 2.8),
                             sharex=True)
    ch_index = {c: i for i, c in enumerate(channels)}
    for ax, ch in zip(axes, show):
        for k, (cls, arr) in enumerate(samples.items()):
            ax.plot(arr[:, ch_index[ch]], color=PALETTE[k % len(PALETTE)],
                    label=_short(cls), alpha=0.9)
        ax.set_title(ch.replace("_", " "))
        ax.set_xlabel("timestep")
    axes[0].set_ylabel("value (scaled)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 3. Confusion matrix
# --------------------------------------------------------------------------- #

def confusion_matrix(cm, classes, figdir: str, name: str,
                     title: str | None = None) -> str:
    set_style()
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([_short(c) for c in classes], rotation=45, ha="right",
                       fontsize=7.5)
    ax.set_yticklabels([_short(c) for c in classes], fontsize=7.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title)
    thr = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if norm[i, j] > thr else "#333")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 4. Drift traces
# --------------------------------------------------------------------------- #

def drift_traces(clean_arr, drift_arr, channel: str, channels, figdir: str,
                 name: str) -> str:
    set_style()
    idx = list(channels).index(channel)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for a in clean_arr:
        ax.plot(a[:, idx], color=PALETTE[0], alpha=0.25, lw=1.0)
    for a in drift_arr:
        ax.plot(a[:, idx], color=PALETTE[1], alpha=0.25, lw=1.0)
    ax.plot(np.mean(clean_arr, axis=0)[:, idx], color=PALETTE[0], lw=2.4,
            label="clean")
    ax.plot(np.mean(drift_arr, axis=0)[:, idx], color=PALETTE[1], lw=2.4,
            label="drift")
    ax.set_xlabel("timestep")
    ax.set_ylabel(channel.replace("_", " "))
    ax.set_title(f"{channel.replace('_', ' ')}: clean vs drift")
    ax.legend()
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 5. Cross-domain bars
# --------------------------------------------------------------------------- #

def cross_domain_bars(table: dict, metric: str, figdir: str, name: str,
                      ylabel: str | None = None) -> str:
    """``table``: {model: {domain: {metric: value}}}."""
    set_style()
    models = list(table.keys())
    domains = list(next(iter(table.values())).keys())
    x = np.arange(len(domains))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for k, m in enumerate(models):
        vals = [table[m][d].get(metric, np.nan) for d in domains]
        ax.bar(x + k * width, vals, width, label=m, color=PALETTE[k % len(PALETTE)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(domains)
    ax.set_ylabel(ylabel or metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{ylabel or metric} across domains")
    ax.legend(ncol=len(models))
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 6. Anomaly summary bars
# --------------------------------------------------------------------------- #

def anomaly_summary(report: dict, figdir: str, name: str) -> str:
    """``report``: {domain: {roc_auc, average_precision, f1_thresholded}}."""
    set_style()
    domains = list(report.keys())
    metrics = ["roc_auc", "average_precision", "f1_thresholded"]
    labels = {"roc_auc": "ROC-AUC", "average_precision": "Avg. precision",
              "f1_thresholded": "F1 @ tau"}
    x = np.arange(len(domains))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for k, m in enumerate(metrics):
        vals = [report[d][m] for d in domains]
        ax.bar(x + k * width, vals, width, label=labels[m],
               color=PALETTE[k % len(PALETTE)])
    ax.set_xticks(x + width)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Open-set anomaly detection (autoencoder)")
    ax.legend(ncol=3)
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 7-8. ROC / PR curves
# --------------------------------------------------------------------------- #

def roc_curves(curves: dict, figdir: str, name: str) -> str:
    """``curves``: {domain: (fpr, tpr, auc)}."""
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for k, (dom, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=PALETTE[k % len(PALETTE)],
                label=f"{dom} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.0)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("Anomaly ROC")
    ax.legend()
    return _save(fig, figdir, name)


def pr_curves(curves: dict, figdir: str, name: str) -> str:
    """``curves``: {domain: (recall, precision, ap)}."""
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for k, (dom, (rec, prec, ap)) in enumerate(curves.items()):
        ax.plot(rec, prec, color=PALETTE[k % len(PALETTE)],
                label=f"{dom} (AP={ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Anomaly precision-recall")
    ax.legend()
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 9. Calibration / reliability
# --------------------------------------------------------------------------- #

def reliability_diagram(calib: dict, figdir: str, name: str,
                        title: str | None = None) -> str:
    """``calib``: output of metrics.expected_calibration_error."""
    set_style()
    edges = np.asarray(calib["bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.asarray(calib["bin_accuracy"])
    conf = np.asarray(calib["bin_confidence"])
    counts = np.asarray(calib["bin_count"])
    width = (edges[1] - edges[0]) * 0.9
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.bar(centers, acc, width=width, color=PALETTE[0], alpha=0.85,
           label="accuracy", edgecolor="white")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1.0, label="perfect")
    mask = counts > 0
    ax.plot(centers[mask], conf[mask], "o-", color=PALETTE[1], ms=4,
            label="confidence")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Reliability (ECE={calib['ece']:.3f})")
    ax.legend(loc="upper left")
    return _save(fig, figdir, name)


def ece_by_domain(ece_table: dict, figdir: str, name: str) -> str:
    """``ece_table``: {model: {domain: ece}}."""
    set_style()
    models = list(ece_table.keys())
    domains = list(next(iter(ece_table.values())).keys())
    x = np.arange(len(domains))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for k, m in enumerate(models):
        vals = [ece_table[m][d] for d in domains]
        ax.bar(x + k * width, vals, width, label=m, color=PALETTE[k % len(PALETTE)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(domains)
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Calibration error across domains")
    ax.legend(ncol=len(models))
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 10. Robustness curves
# --------------------------------------------------------------------------- #

def robustness_curves(sweeps: dict, figdir: str, name: str) -> str:
    """``sweeps``: {axis: {"points": [{"level":, "f1_macro":}, ...]}}."""
    set_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    nice = {"drift_strength": "phase-drift amplitude",
            "asym_strength": "asymmetric-loss magnitude",
            "attack_intensity": "attack intensity"}
    for k, (axis, data) in enumerate(sweeps.items()):
        pts = data["points"]
        xs = [p["level"] for p in pts]
        ys = [p["f1_macro"] for p in pts]
        ax.plot(xs, ys, "o-", color=PALETTE[k % len(PALETTE)],
                label=nice.get(axis, axis))
    ax.set_xlabel("shift severity")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness to increasing domain shift")
    ax.legend()
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 11. DET / operating curves
# --------------------------------------------------------------------------- #

def det_curves(curves: dict, figdir: str, name: str) -> str:
    """``curves``: {domain: (false_alarm, miss_rate)}."""
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for k, (dom, (fa, miss)) in enumerate(curves.items()):
        ax.plot(fa, miss, color=PALETTE[k % len(PALETTE)], label=dom)
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("miss rate")
    ax.set_title("Detection error tradeoff")
    ax.legend()
    return _save(fig, figdir, name)


def operating_curve(thr, alarm, detect, figdir: str, name: str) -> str:
    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(alarm, detect, color=PALETTE[2])
    ax.set_xlabel("alarm rate (fraction flagged)")
    ax.set_ylabel("detection rate")
    ax.set_title("Operating characteristic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 12. Attention heatmap
# --------------------------------------------------------------------------- #

def attention_heatmap(attn, figdir: str, name: str) -> str:
    set_style()
    attn = np.asarray(attn)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(attn, cmap="magma", aspect="auto")
    ax.set_xlabel("key timestep")
    ax.set_ylabel("query timestep")
    ax.set_title("Transformer attention (last layer)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, figdir, name)


def attention_mass(mass, figdir: str, name: str) -> str:
    set_style()
    mass = np.asarray(mass)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar(np.arange(len(mass)), mass, color=PALETTE[4])
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean attention received")
    ax.set_title("Temporal attention concentration")
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 13. t-SNE
# --------------------------------------------------------------------------- #

def tsne_scatter(proj: dict, figdir: str, name: str) -> str:
    set_style()
    xy = np.asarray(proj["xy"])
    labels = proj["labels"]
    uniq = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for k, cls in enumerate(uniq):
        m = np.array([l == cls for l in labels])
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.7,
                   color=PALETTE[k % len(PALETTE)], label=_short(cls))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Latent embedding (t-SNE)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(False)
    return _save(fig, figdir, name)


# --------------------------------------------------------------------------- #
# 14. Channel importance
# --------------------------------------------------------------------------- #

def channel_importance(imp: dict, figdir: str, name: str) -> str:
    """``imp``: output of evaluate.permutation_channel_importance."""
    set_style()
    items = sorted(imp["importance"].items(), key=lambda kv: kv[1])
    names = [k.replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    errs = [imp.get("importance_std", {}).get(k, 0.0) for k, _ in items]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.barh(np.arange(len(names)), vals, xerr=errs, color=PALETTE[0],
            ecolor="#888", capsize=2)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("macro-F1 drop when permuted")
    ax.set_title("Channel importance (permutation)")
    return _save(fig, figdir, name)
