"""Training loops and checkpoint IO."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8           # early-stopping patience (0 disables)
    grad_clip: float = 1.0
    device: str = "cpu"
    verbose: bool = True


def _loader(x: np.ndarray, y: np.ndarray | None, batch_size: int,
            shuffle: bool) -> DataLoader:
    xt = torch.as_tensor(x, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(xt)
    else:
        ds = TensorDataset(xt, torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# Supervised
# --------------------------------------------------------------------------- #

def train_supervised(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray,
                     x_val: np.ndarray, y_val: np.ndarray,
                     cfg: TrainConfig) -> dict:
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    tr = _loader(x_train, y_train, cfg.batch_size, shuffle=True)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, bad = np.inf, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(tr.dataset)

        vl, va = _evaluate_supervised(model, x_val, y_val, crit, device,
                                      cfg.batch_size)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        if cfg.verbose:
            print(f"  epoch {epoch + 1:3d}/{cfg.epochs} "
                  f"train_loss={train_loss:.4f} val_loss={vl:.4f} "
                  f"val_acc={va:.4f}")

        if vl < best_val - 1e-5:
            best_val, best_state, bad = vl, _clone_state(model), 0
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                if cfg.verbose:
                    print(f"  early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def _evaluate_supervised(model, x, y, crit, device, batch_size):
    model.eval()
    loader = _loader(x, y, batch_size, shuffle=False)
    loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss += crit(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return loss / n, correct / n


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, device: str = "cpu",
                   batch_size: int = 256) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    out = []
    for (xb,) in _loader(x, None, batch_size, shuffle=False):
        out.append(model(xb.to(dev)).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def extract_embeddings(model: nn.Module, x: np.ndarray, device: str = "cpu",
                       batch_size: int = 256) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    out = []
    for (xb,) in _loader(x, None, batch_size, shuffle=False):
        out.append(model.embed(xb.to(dev)).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# One-class autoencoder
# --------------------------------------------------------------------------- #

def train_autoencoder(model: nn.Module, x_normal: np.ndarray,
                      x_val_normal: np.ndarray, cfg: TrainConfig) -> dict:
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    tr = _loader(x_normal, None, cfg.batch_size, shuffle=True)
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, bad = np.inf, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for (xb,) in tr:
            xb = xb.to(device)
            opt.zero_grad()
            rec = model(xb)
            loss = ((xb - rec) ** 2).mean()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(tr.dataset)
        vl = _ae_val_loss(model, x_val_normal, device, cfg.batch_size)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vl)
        if cfg.verbose:
            print(f"  epoch {epoch + 1:3d}/{cfg.epochs} "
                  f"train_mse={train_loss:.5f} val_mse={vl:.5f}")
        if vl < best_val - 1e-7:
            best_val, best_state, bad = vl, _clone_state(model), 0
        else:
            bad += 1
            if cfg.patience and bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def _ae_val_loss(model, x, device, batch_size):
    model.eval()
    loss, n = 0.0, 0
    for (xb,) in _loader(x, None, batch_size, shuffle=False):
        xb = xb.to(device)
        rec = model(xb)
        loss += ((xb - rec) ** 2).mean().item() * xb.size(0)
        n += xb.size(0)
    return loss / n


@torch.no_grad()
def reconstruction_scores(model: nn.Module, x: np.ndarray, device: str = "cpu",
                          batch_size: int = 256) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    out = []
    for (xb,) in _loader(x, None, batch_size, shuffle=False):
        out.append(model.reconstruction_error(xb.to(dev)).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #

def _clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(model: nn.Module, path: str | Path,
                    device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
