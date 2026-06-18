"""Same-domain closed-set evaluation for all supervised detectors."""
from _common import (banner, data_dir, load_experiment, load_model_from_ckpt,
                     metrics_dir, resolve_domains, resolve_models, save_json,
                     ckpt_dir)
from qsentry.datasets import Preprocessor, load_domain
from qsentry.evaluate import closed_set_report


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    models = resolve_models(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")

    banner(f"Closed-set evaluation on '{train_domain}' test split")
    out = {}
    for name in models:
        if models[name]["model"] == "autoencoder":
            continue
        model, _ = load_model_from_ckpt(cfg, name, len(pre.classes_), seq_len)
        rep = closed_set_report(model, df, pre, split="test", device=device)
        out[name] = rep
        m = rep["metrics"]
        print(f"  {name:12s} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
              f"auc={m['roc_auc_ovr_macro']:.4f}")
    save_json(out, metrics_dir(cfg) / "closed_set.json")


if __name__ == "__main__":
    main()
