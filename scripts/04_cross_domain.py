"""Cross-domain generalisation: train-on-clean, evaluate on every domain."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     load_model_from_ckpt, metrics_dir, resolve_domains,
                     resolve_models, save_json)
from qsentry.datasets import Preprocessor, load_domain
from qsentry.evaluate import cross_domain_report


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    models = resolve_models(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    frames = {n: load_domain(data_dir(cfg) / f"tfqkd_{n}.csv") for n in specs}

    banner("Cross-domain evaluation")
    table = {}
    for name in models:
        if models[name]["model"] == "autoencoder":
            continue
        model, _ = load_model_from_ckpt(cfg, name, len(pre.classes_), seq_len)
        table[name] = cross_domain_report(model, frames, pre, split="test",
                                          device=device)
        cells = " ".join(f"{d}={table[name][d]['f1_macro']:.3f}" for d in specs)
        print(f"  {name:12s} {cells}")
    save_json(table, metrics_dir(cfg) / "cross_domain.json")


if __name__ == "__main__":
    main()
