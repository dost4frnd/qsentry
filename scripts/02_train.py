"""Train all supervised detectors and the one-class autoencoder."""
from _common import (banner, ckpt_dir, data_dir, load_experiment,
                     resolve_domains, resolve_models, save_json)
from qsentry.datasets import Preprocessor, get_xy, load_domain, split_frame
from qsentry.models import build_model
from qsentry.physics import N_CHANNELS
from qsentry.train import (TrainConfig, save_checkpoint, train_autoencoder,
                           train_supervised)


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    models = resolve_models(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len

    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")
    x_tr, y_tr, _ = get_xy(df, "train", pre, known_only=True)
    x_va, y_va, _ = get_xy(df, "val", pre, known_only=True)
    n_classes = len(pre.classes_)

    histories = {}
    for name, spec in models.items():
        banner(f"Training {name}")
        tcfg = TrainConfig(device=device, **spec["train"])
        model = build_model(spec["model"], n_features=N_CHANNELS,
                            seq_len=seq_len, n_classes=n_classes,
                            params=spec["params"])
        if spec["model"] == "autoencoder":
            tr_n = split_frame(df, "train")
            tr_n = tr_n[tr_n["label"] == "normal"]
            va_n = split_frame(df, "val")
            va_n = va_n[va_n["label"] == "normal"]
            x_n = pre.transform(tr_n, as_sequence=True)
            x_vn = pre.transform(va_n, as_sequence=True)
            hist = train_autoencoder(model, x_n, x_vn, tcfg)
        else:
            hist = train_supervised(model, x_tr, y_tr, x_va, y_va, tcfg)
        save_checkpoint(model, ckpt_dir(cfg) / f"{name}.pt",
                        meta={"model": spec["model"], "params": spec["params"],
                              "n_params": int(model.num_parameters())})
        histories[name] = hist
        print(f"  {name}: {model.num_parameters():,} params")
    save_json(histories, ckpt_dir(cfg) / "histories.json")
    print("training complete.")


if __name__ == "__main__":
    main()
