"""Shared helpers for the numbered pipeline scripts.

Run every script from the repository root, e.g.::

    python scripts/00_generate_data.py configs/experiment/full.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# make ``src`` importable without installation
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qsentry.config import deep_merge, load_yaml          # noqa: E402
from qsentry.datasets import DomainSpec                    # noqa: E402

DEFAULT_EXPERIMENT = "configs/experiment/full.yaml"

_DOMAIN_FIELDS = {
    "name", "seq_len", "n_train", "n_val", "n_test", "drift_strength",
    "asym_strength", "attack_intensity", "mixed", "include_unknown_in_test",
    "n_test_normal", "seed",
}


def experiment_path() -> Path:
    arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPERIMENT
    return Path(arg)


def load_experiment(path: Path | None = None) -> tuple[dict, Path]:
    path = Path(path or experiment_path())
    cfg = load_yaml(path)
    config_root = path.parent.parent       # configs/
    return cfg, config_root


def resolve_domains(cfg: dict, config_root: Path) -> dict[str, DomainSpec]:
    seq_len = cfg.get("seq_len", 32)
    raw: dict[str, dict] = {}
    if cfg.get("domains"):
        raw = cfg["domains"]
    elif cfg.get("domain_configs"):
        for name, rel in cfg["domain_configs"].items():
            raw[name] = load_yaml(config_root / rel)
    else:
        raise ValueError("experiment config has neither 'domains' nor 'domain_configs'")

    specs = {}
    for name, d in raw.items():
        d = dict(d)
        d.setdefault("name", name)
        d.setdefault("seq_len", seq_len)
        kwargs = {k: v for k, v in d.items() if k in _DOMAIN_FIELDS}
        specs[name] = DomainSpec(**kwargs)
    return specs


def resolve_models(cfg: dict, config_root: Path) -> dict[str, dict]:
    global_train = cfg.get("train", {})
    raw: dict[str, dict] = {}
    if cfg.get("models"):
        raw = cfg["models"]
    elif cfg.get("model_configs"):
        for name, rel in cfg["model_configs"].items():
            raw[name] = load_yaml(config_root / rel)
    else:
        raise ValueError("experiment config has neither 'models' nor 'model_configs'")

    out = {}
    for name, m in raw.items():
        m = dict(m)
        train = deep_merge(global_train, m.get("train", {}))
        out[name] = {"model": m.get("model", name),
                     "params": m.get("params", {}),
                     "train": train}
    return out


def resolve_robustness(cfg: dict, config_root: Path) -> dict:
    rob = dict(cfg.get("robustness", {}))
    conf = rob.get("config")
    if isinstance(conf, str):
        rob["config"] = load_yaml(config_root / conf)
    return rob


# -- paths + io ------------------------------------------------------------- #

def paths(cfg: dict) -> dict:
    return cfg.get("paths", {})


def ensure_dirs(cfg: dict) -> None:
    for key, p in paths(cfg).items():
        Path(p).mkdir(parents=True, exist_ok=True)


def metrics_dir(cfg) -> Path:
    return Path(paths(cfg).get("metrics_dir", "results/metrics"))


def data_dir(cfg) -> Path:
    return Path(paths(cfg).get("data_dir", "data"))


def ckpt_dir(cfg) -> Path:
    return Path(paths(cfg).get("ckpt_dir", "results/checkpoints"))


def fig_dir(cfg) -> Path:
    return Path(paths(cfg).get("fig_dir", "paper/figures"))


def tab_dir(cfg) -> Path:
    return Path(paths(cfg).get("tab_dir", "paper/tables"))


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def banner(msg: str) -> None:
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def load_model_from_ckpt(cfg: dict, name: str, n_classes: int, seq_len: int):
    """Rebuild a model from its checkpoint metadata and load weights."""
    import torch

    from qsentry.models import build_model
    from qsentry.physics import N_CHANNELS
    from qsentry.train import load_checkpoint

    path = ckpt_dir(cfg) / f"{name}.pt"
    meta = torch.load(path, map_location="cpu")["meta"]
    model = build_model(meta["model"], n_features=N_CHANNELS, seq_len=seq_len,
                        n_classes=n_classes, params=meta.get("params", {}))
    load_checkpoint(model, path, device=cfg.get("device", "cpu"))
    return model, meta
