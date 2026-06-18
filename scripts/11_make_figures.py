"""Render every manuscript figure from the serialized results."""
import numpy as np

from _common import (banner, fig_dir, load_experiment, load_json, metrics_dir,
                     resolve_domains)
from qsentry import viz
from qsentry.physics import CHANNELS, CLOSED_SET_CLASSES, make_generator
from qsentry.seeding import new_rng


def _example_windows():
    rng = new_rng(0)
    gen = make_generator(seq_len=32)
    return {c: gen.sample_window(c, rng) for c in CLOSED_SET_CLASSES}


def _drift_examples(n=20):
    rng = new_rng(1)
    clean = make_generator(seq_len=32, drift_strength=0.0)
    drift = make_generator(seq_len=32, drift_strength=1.5)
    ca = np.stack([clean.sample_window("normal", rng) for _ in range(n)])
    da = np.stack([drift.sample_window("normal", rng) for _ in range(n)])
    return ca, da


def main():
    cfg, root = load_experiment()
    fd = str(fig_dir(cfg))
    md = metrics_dir(cfg)
    specs = resolve_domains(cfg, root)
    banner(f"Rendering figures -> {fd}")
    made = []

    made.append(viz.architecture_diagram(fd))
    made.append(viz.telemetry_signatures(_example_windows(), CHANNELS, fd))
    ca, da = _drift_examples()
    made.append(viz.drift_traces(ca, da, "phase_lock_error_rad", CHANNELS, fd,
                                 "fig_drift_phase_lock"))
    made.append(viz.drift_traces(ca, da, "qber_phase", CHANNELS, fd,
                                 "fig_drift_qber_phase"))

    # closed-set confusion matrices
    try:
        cs = load_json(md / "closed_set.json")
        for name, rep in cs.items():
            made.append(viz.confusion_matrix(
                rep["confusion"], rep["classes"], fd,
                f"fig_confusion_{name}", title=f"{name} (clean)"))
    except FileNotFoundError:
        pass

    # cross-domain bars
    try:
        cd = load_json(md / "cross_domain.json")
        made.append(viz.cross_domain_bars(cd, "f1_macro", fd,
                    "fig_cross_domain_f1", ylabel="macro-F1"))
        made.append(viz.cross_domain_bars(cd, "accuracy", fd,
                    "fig_cross_domain_acc", ylabel="accuracy"))
        made.append(viz.cross_domain_bars(cd, "roc_auc_ovr_macro", fd,
                    "fig_cross_domain_auc", ylabel="AUC-OVR"))
    except FileNotFoundError:
        pass

    # open-set
    try:
        os_ = load_json(md / "open_set.json")
        made.append(viz.anomaly_summary(os_["report"], fd, "fig_anomaly_summary"))
        roc = {d: (np.array(v["fpr"]), np.array(v["tpr"]), v["auc"])
               for d, v in os_["curves"]["roc"].items()}
        pr = {d: (np.array(v["recall"]), np.array(v["precision"]), v["ap"])
              for d, v in os_["curves"]["pr"].items()}
        det = {d: (np.array(v["fa"]), np.array(v["miss"]))
               for d, v in os_["curves"]["det"].items()}
        made.append(viz.roc_curves(roc, fd, "fig_anomaly_roc"))
        made.append(viz.pr_curves(pr, fd, "fig_anomaly_pr"))
        made.append(viz.det_curves(det, fd, "fig_det"))
    except FileNotFoundError:
        pass

    # calibration
    try:
        cal = load_json(md / "calibration.json")
        made.append(viz.ece_by_domain(cal["ece"], fd, "fig_ece_by_domain"))
        ref = cfg.get("interpret", {}).get("reference_model", "transformer")
        if ref in cal["reliability"]:
            for dom in ("clean", "unknown"):
                if dom in cal["reliability"][ref]:
                    made.append(viz.reliability_diagram(
                        cal["reliability"][ref][dom], fd,
                        f"fig_reliability_{ref}_{dom}",
                        title=f"{ref} reliability ({dom})"))
    except FileNotFoundError:
        pass

    # robustness
    try:
        rob = load_json(md / "robustness.json")
        made.append(viz.robustness_curves(rob["sweeps"], fd, "fig_robustness"))
    except FileNotFoundError:
        pass

    # interpretability
    try:
        ip = load_json(md / "interpret.json")
        if "attention" in ip:
            made.append(viz.attention_heatmap(ip["attention"]["attention_map"],
                        fd, "fig_attention_heatmap"))
            made.append(viz.attention_mass(ip["attention"]["per_timestep_mass"],
                        fd, "fig_attention_mass"))
        made.append(viz.tsne_scatter(ip["tsne"], fd, "fig_tsne"))
    except FileNotFoundError:
        pass

    # channel importance
    try:
        imp = load_json(md / "channel_importance.json")
        made.append(viz.channel_importance(imp, fd, "fig_channel_importance"))
    except FileNotFoundError:
        pass

    for p in made:
        print(f"  + {p}")
    print(f"{len(made)} figures written.")


if __name__ == "__main__":
    main()
