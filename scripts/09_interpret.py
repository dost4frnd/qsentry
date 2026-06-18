"""Interpretability: attention maps + latent t-SNE for the reference model."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     load_model_from_ckpt, metrics_dir, resolve_domains,
                     save_json)
from qsentry.datasets import Preprocessor, load_domain
from qsentry.interpret import attention_profile, tsne_embeddings


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    ic = cfg.get("interpret", {})
    ref = ic.get("reference_model", "transformer")
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")
    model, _ = load_model_from_ckpt(cfg, ref, len(pre.classes_), seq_len)

    banner(f"Interpretability (model={ref})")
    out = {}
    try:
        out["attention"] = attention_profile(model, df, pre, split="test",
                                             device=device)
        print("  attention map extracted")
    except AttributeError:
        print(f"  {ref} exposes no attention; skipping attention map")
    out["tsne"] = tsne_embeddings(model, df, pre, split="test",
                                  max_samples=ic.get("tsne_max_samples", 1500),
                                  device=device)
    print("  t-SNE projection computed")
    save_json(out, metrics_dir(cfg) / "interpret.json")


if __name__ == "__main__":
    main()
