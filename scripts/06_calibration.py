"""Calibration / trust analysis: ECE + reliability bins across domains."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     load_model_from_ckpt, metrics_dir, resolve_domains,
                     resolve_models, save_json)
from qsentry.datasets import Preprocessor, get_xy, load_domain
from qsentry.metrics import (brier_multiclass, expected_calibration_error,
                             softmax, temperature_scale)
from qsentry.train import predict_logits


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    models = resolve_models(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    n_bins = cfg.get("eval", {}).get("calibration_bins", 15)
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    frames = {n: load_domain(data_dir(cfg) / f"tfqkd_{n}.csv") for n in specs}
    n_classes = len(pre.classes_)

    banner("Calibration analysis")
    ece_table, reliability, brier, temps = {}, {}, {}, {}
    for name in models:
        if models[name]["model"] == "autoencoder":
            continue
        model, _ = load_model_from_ckpt(cfg, name, n_classes, seq_len)
        ece_table[name], reliability[name], brier[name] = {}, {}, {}
        # temperature fit on clean validation split
        xv, yv, _ = get_xy(frames[train_domain], "val", pre, known_only=True)
        temps[name] = temperature_scale(predict_logits(model, xv, device=device),
                                        yv)
        for dom, df in frames.items():
            x, y, _ = get_xy(df, "test", pre, known_only=True)
            logits = predict_logits(model, x, device=device)
            probs = softmax(logits)
            cal = expected_calibration_error(y, probs, n_bins=n_bins)
            ece_table[name][dom] = cal["ece"]
            reliability[name][dom] = cal
            brier[name][dom] = brier_multiclass(y, probs, n_classes)
        cells = " ".join(f"{d}={ece_table[name][d]:.3f}" for d in specs)
        print(f"  {name:12s} ECE: {cells} | T={temps[name]:.2f}")
    save_json({"ece": ece_table, "reliability": reliability, "brier": brier,
               "temperature": temps}, metrics_dir(cfg) / "calibration.json")


if __name__ == "__main__":
    main()
