"""Permutation channel importance for the reference detector."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     load_model_from_ckpt, metrics_dir, resolve_domains,
                     save_json)
from qsentry.datasets import Preprocessor, load_domain
from qsentry.evaluate import permutation_channel_importance


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    ref = cfg.get("interpret", {}).get("reference_model", "transformer")
    n_rep = cfg.get("eval", {}).get("channel_importance_repeats", 5)
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")
    model, _ = load_model_from_ckpt(cfg, ref, len(pre.classes_), seq_len)

    banner(f"Channel importance (model={ref}, repeats={n_rep})")
    imp = permutation_channel_importance(model, df, pre, split="test",
                                         n_repeats=n_rep, device=device)
    ordered = sorted(imp["importance"].items(), key=lambda kv: kv[1],
                     reverse=True)
    for ch, v in ordered[:6]:
        print(f"  {ch:24s} {v:+.4f}")
    save_json(imp, metrics_dir(cfg) / "channel_importance.json")


if __name__ == "__main__":
    main()
