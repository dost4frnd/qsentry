"""Operating-point profiling: parameter budget + CPU inference latency."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     load_model_from_ckpt, metrics_dir, resolve_domains,
                     resolve_models, save_json)
from qsentry.datasets import Preprocessor, get_xy, load_domain
from qsentry.operating import profile_model


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    models = resolve_models(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")
    x, _, _ = get_xy(df, "test", pre, known_only=True)

    banner("Operating-point profiling (batch size 1, CPU)")
    profiles = {}
    for name in models:
        model, _ = load_model_from_ckpt(cfg, name, len(pre.classes_), seq_len)
        profiles[name] = profile_model(model, x, batch_size=1, device=device)
        p = profiles[name]
        print(f"  {name:12s} params={p['parameters_trainable']:>8,} "
              f"lat={p['latency_ms_per_window']:.3f}ms "
              f"thru={p['throughput_windows_per_s']:.0f}/s")
    save_json(profiles, metrics_dir(cfg) / "operating.json")


if __name__ == "__main__":
    main()
