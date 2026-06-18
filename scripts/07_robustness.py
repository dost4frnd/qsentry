"""Robustness severity sweeps with the reference detector."""
from _common import (banner, ckpt_dir, load_experiment, load_model_from_ckpt,
                     metrics_dir, resolve_domains, resolve_robustness,
                     save_json)
from qsentry.datasets import Preprocessor
from qsentry.robustness import severity_sweep


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    rob = resolve_robustness(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    ref = rob.get("reference_model", "transformer")
    conf = rob["config"]
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    model, _ = load_model_from_ckpt(cfg, ref, len(pre.classes_), seq_len)

    banner(f"Robustness sweeps (model={ref})")
    sweeps = {}
    for axis, levels in conf["axes"].items():
        sweeps[axis] = severity_sweep(model, pre, axis, levels, seq_len=seq_len,
                                      n_test=conf.get("n_test", 480),
                                      seed=conf.get("seed", 7), device=device)
        pts = " ".join(f"{p['level']}:{p['f1_macro']:.3f}"
                       for p in sweeps[axis]["points"])
        print(f"  {axis:18s} {pts}")
    save_json({"reference_model": ref, "sweeps": sweeps},
              metrics_dir(cfg) / "robustness.json")


if __name__ == "__main__":
    main()
