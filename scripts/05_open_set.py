"""Open-set novelty detection with the one-class autoencoder."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     load_model_from_ckpt, metrics_dir, resolve_domains,
                     resolve_models, save_json)
from qsentry.datasets import Preprocessor, load_domain
from qsentry.evaluate import (anomaly_scores, calibrate_threshold,
                              open_set_report)
from qsentry.metrics import det_points, pr_points, roc_points


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    models = resolve_models(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    q = cfg.get("eval", {}).get("ae_quantile", 0.95)

    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    train_df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")
    frames = {n: load_domain(data_dir(cfg) / f"tfqkd_{n}.csv") for n in specs}

    ae_name = next((n for n in models if models[n]["model"] == "autoencoder"),
                   None)
    if ae_name is None:
        raise SystemExit("no autoencoder configured")
    ae, _ = load_model_from_ckpt(cfg, ae_name, len(pre.classes_), seq_len)

    tau = calibrate_threshold(ae, train_df, pre, q=q, device=device)
    banner(f"Open-set anomaly detection (tau={tau:.4f})")
    report = open_set_report(ae, frames, pre, threshold=tau, split="test",
                             device=device)

    # curve points for figures
    curves = {"roc": {}, "pr": {}, "det": {}}
    for name, df in frames.items():
        s, is_anom, _ = anomaly_scores(ae, df, pre, "test", device=device)
        fpr, tpr, _ = roc_points(is_anom, s)
        prec, rec, _ = pr_points(is_anom, s)
        fa, miss, _ = det_points(is_anom, s)
        curves["roc"][name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                               "auc": report[name]["roc_auc"]}
        curves["pr"][name] = {"recall": rec.tolist(), "precision": prec.tolist(),
                              "ap": report[name]["average_precision"]}
        curves["det"][name] = {"fa": fa.tolist(), "miss": miss.tolist()}
        v = report[name]
        print(f"  {name:8s} AP={v['average_precision']:.4f} "
              f"AUC={v['roc_auc']:.4f} F1@tau={v['f1_thresholded']:.4f}")

    save_json({"threshold": tau, "report": report, "curves": curves},
              metrics_dir(cfg) / "open_set.json")


if __name__ == "__main__":
    main()
