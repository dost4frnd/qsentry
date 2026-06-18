"""Fit the preprocessor (train-only scaler + closed-set encoder)."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     resolve_domains)
from qsentry.datasets import Preprocessor, load_domain, split_frame


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    train_domain = cfg.get("train_domain", "clean")
    banner(f"Fitting preprocessor on '{train_domain}' train split")
    df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")
    pre = Preprocessor(seq_len=specs[train_domain].seq_len)
    pre.fit(split_frame(df, "train"))
    out = ckpt_dir(cfg) / "preprocessor.npz"
    pre.save(out)
    print(f"  scaler over {len(pre.feature_cols)} features; "
          f"classes={len(pre.classes_)} -> {out}")


if __name__ == "__main__":
    main()
