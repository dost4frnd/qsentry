"""Generate LaTeX table floats (booktabs) for the manuscript.

Each function writes a complete ``table`` float (caption + label) into the
tables directory so the manuscript can ``\\input`` it directly.  Tables for the
*new* experiments (calibration, efficiency, channel importance, dataset
dimensions) are produced here from pipeline outputs; the preserved
closed-set / cross-domain / open-set numbers live in ``main.tex`` but can also
be regenerated here for comparison with a fresh run.
"""
from __future__ import annotations

from pathlib import Path

MODEL_NICE = {"transformer": "Transformer", "lstm": "LSTM", "qlstm": "QLSTM",
              "autoencoder": "Autoencoder"}


def _w(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return str(path)


def _f(x, nd=4):
    try:
        if x != x:  # NaN
            return "--"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _nice(name):
    return MODEL_NICE.get(name, name)


# --------------------------------------------------------------------------- #

def write_dataset_table(specs: dict, tabdir: str,
                        name: str = "tab_datasets") -> str:
    """``specs``: {domain: {n_train, n_val, n_test, shift}}."""
    rows = []
    for dom, s in specs.items():
        rows.append(
            f"{dom} & {s.get('shift','--')} & {s['n_train']} & {s['n_val']} "
            f"& {s['n_test']} & {s['n_train']+s['n_val']+s['n_test']} \\\\")
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Domain datasets. All windows share the $T{\\times}F = 32{\\times}12$ "
        "telemetry layout (384 sequence features). The held-out novelty class appears "
        "only in the \\texttt{unknown} test split.}\n"
        "\\label{tab:datasets}\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Domain & Shift & Train & Val & Test & Total \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return _w(Path(tabdir) / f"{name}.tex", tex)


def write_closed_set_table(metrics: dict, tabdir: str,
                           name: str = "tab_closed_set") -> str:
    """``metrics``: {model: {accuracy, f1_macro, roc_auc_ovr_macro}}."""
    rows = []
    for m, vals in metrics.items():
        rows.append(f"{_nice(m)} & {_f(vals['accuracy'])} & "
                    f"{_f(vals['f1_macro'])} & {_f(vals['roc_auc_ovr_macro'])} \\\\")
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Same-domain (\\texttt{clean}) closed-set detection.}\n"
        "\\label{tab:closed-set}\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Model & Accuracy & Macro-F1 & AUC-OVR \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return _w(Path(tabdir) / f"{name}.tex", tex)


def write_cross_domain_table(table: dict, tabdir: str,
                             name: str = "tab_cross_domain") -> str:
    """``table``: {model: {domain: {accuracy, f1_macro, roc_auc_ovr_macro}}}."""
    models = list(table.keys())
    domains = list(next(iter(table.values())).keys())
    header = "Model & " + " & ".join(domains) + " \\\\"
    rows = []
    for m in models:
        cells = " & ".join(_f(table[m][d]["f1_macro"]) for d in domains)
        rows.append(f"{_nice(m)} & {cells} \\\\")
    body = "\n".join(rows)
    col = "l" + "c" * len(domains)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Cross-domain generalisation (macro-F1). Models are trained on "
        "\\texttt{clean} and evaluated on each domain's test split.}\n"
        "\\label{tab:cross-domain}\n"
        f"\\begin{{tabular}}{{{col}}}\n\\toprule\n"
        f"{header}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return _w(Path(tabdir) / f"{name}.tex", tex)


def write_open_set_table(report: dict, tabdir: str,
                         name: str = "tab_open_set") -> str:
    """``report``: {domain: {average_precision, f1_thresholded, roc_auc, threshold}}."""
    rows = []
    for dom, v in report.items():
        rows.append(f"{dom} & {_f(v['average_precision'])} & "
                    f"{_f(v['f1_thresholded'])} & {_f(v['roc_auc'])} & "
                    f"{_f(v['threshold'])} \\\\")
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Open-set anomaly detection (autoencoder, one-class). The "
        "threshold $\\tau$ is the 95th percentile of normal-train reconstruction "
        "error.}\n"
        "\\label{tab:open-set}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Domain & Avg.\\ Prec.\\ & F1\\,@\\,$\\tau$ & ROC-AUC & $\\tau$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return _w(Path(tabdir) / f"{name}.tex", tex)


def write_calibration_table(ece_table: dict, tabdir: str,
                            name: str = "tab_calibration") -> str:
    """``ece_table``: {model: {domain: ece}}."""
    models = list(ece_table.keys())
    domains = list(next(iter(ece_table.values())).keys())
    header = "Model & " + " & ".join(domains) + " \\\\"
    rows = []
    for m in models:
        cells = " & ".join(_f(ece_table[m][d], 4) for d in domains)
        rows.append(f"{_nice(m)} & {cells} \\\\")
    body = "\n".join(rows)
    col = "l" + "c" * len(domains)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Expected calibration error (ECE, lower is better) of the "
        "closed-set detectors across domains. Calibration degrades under shift, "
        "motivating the trust layer.}\n"
        "\\label{tab:calibration}\n"
        f"\\begin{{tabular}}{{{col}}}\n\\toprule\n"
        f"{header}\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return _w(Path(tabdir) / f"{name}.tex", tex)


def write_efficiency_table(profiles: dict, tabdir: str,
                           name: str = "tab_efficiency") -> str:
    """``profiles``: {model: {parameters_trainable, latency_ms_per_window,
    throughput_windows_per_s}}."""
    rows = []
    for m, p in profiles.items():
        rows.append(
            f"{_nice(m)} & {p['parameters_trainable']:,} & "
            f"{_f(p['latency_ms_per_window'], 3)} & "
            f"{p['throughput_windows_per_s']:.0f} \\\\")
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Parameter budget and CPU inference cost (batch size 1). "
        "Latency is per telemetry window.}\n"
        "\\label{tab:efficiency}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Model & Params & Latency (ms) & Throughput (win/s) \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return _w(Path(tabdir) / f"{name}.tex", tex)


def write_channel_importance_table(imp: dict, tabdir: str, top_k: int = 6,
                                   name: str = "tab_channel_importance") -> str:
    """``imp``: output of evaluate.permutation_channel_importance."""
    items = sorted(imp["importance"].items(), key=lambda kv: kv[1], reverse=True)
    items = items[:top_k]
    rows = []
    for ch, v in items:
        std = imp.get("importance_std", {}).get(ch, 0.0)
        rows.append(f"{ch.replace('_',' ')} & {_f(v,4)} & {_f(std,4)} \\\\")
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Most important telemetry channels for closed-set detection, "
        "by permutation importance (macro-F1 drop). Physically-meaningful "
        "channels dominate.}\n"
        "\\label{tab:channel-importance}\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Channel & F1 drop & std \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return _w(Path(tabdir) / f"{name}.tex", tex)
