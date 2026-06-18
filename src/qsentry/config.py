"""Lightweight configuration handling.

Configs are plain YAML files.  A single experiment config may reference data
and model configs which are deep-merged at load time.  Command-line overrides
of the form ``key.subkey=value`` are also supported so that scripts can be
driven without editing files.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` (non-destructive)."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(value: str) -> Any:
    """Best-effort string -> python literal."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides in place and return the config."""
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Malformed override (expected key=value): {item}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def resolve_experiment(path: str | Path, config_root: str | Path | None = None,
                       overrides: list[str] | None = None) -> dict:
    """Load an experiment YAML, resolving referenced data/model configs.

    An experiment file may contain ``data_config`` and ``model_configs``
    (a mapping of name -> path) keys whose targets are merged into ``data``
    and ``models`` sections respectively.
    """
    path = Path(path)
    config_root = Path(config_root) if config_root else path.parent.parent
    cfg = load_yaml(path)

    if "data_config" in cfg:
        data_path = config_root / cfg.pop("data_config")
        cfg["data"] = deep_merge(load_yaml(data_path), cfg.get("data", {}))

    model_cfgs = cfg.pop("model_configs", {})
    models: dict[str, Any] = cfg.get("models", {})
    for name, rel in model_cfgs.items():
        loaded = load_yaml(config_root / rel)
        models[name] = deep_merge(loaded, models.get(name, {}))
    if models:
        cfg["models"] = models

    return apply_overrides(cfg, overrides)
