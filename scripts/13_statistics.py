"""Statistical summaries: per-class F1, bootstrap CIs, McNemar test."""
import numpy as np

from _common import (banner, data_dir, ckpt_dir, load_experiment,
                     load_model_from_ckpt, metrics_dir, resolve_domains,
                     resolve_models, save_json)
from qsentry.datasets import Preprocessor, get_xy, load_domain
from qsentry.metrics import softmax
from qsentry.train import predict_logits


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(np.mean(y_true[idx] == y_pred[idx]))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(np.mean(accs)), float(lo), float(hi)


def mcnemar(y_true, pred_a, pred_b):
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    from scipy.stats import chi2
    p = float(1 - chi2.cdf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p}


def main():
    cfg, root = load_experiment()
    specs = resolve_domains(cfg, root)
    models = resolve_models(cfg, root)
    device = cfg.get("device", "cpu")
    train_domain = cfg.get("train_domain", "clean")
    seq_len = specs[train_domain].seq_len
    pre = Preprocessor.load(ckpt_dir(cfg) / "preprocessor.npz")
    df = load_domain(data_dir(cfg) / f"tfqkd_{train_domain}.csv")
    x, y, _ = get_xy(df, "test", pre, known_only=True)

    banner("Statistical analysis (clean test split)")
    from sklearn.metrics import f1_score
    preds = {}
    out = {"bootstrap_accuracy": {}, "per_class_f1": {}}
    for name in models:
        if models[name]["model"] == "autoencoder":
            continue
        model, _ = load_model_from_ckpt(cfg, name, len(pre.classes_), seq_len)
        pred = softmax(predict_logits(model, x, device=device)).argmax(1)
        preds[name] = pred
        mean, lo, hi = bootstrap_ci(y, pred)
        out["bootstrap_accuracy"][name] = {"mean": mean, "ci95": [lo, hi]}
        f1s = f1_score(y, pred, average=None,
                       labels=list(range(len(pre.classes_))), zero_division=0)
        out["per_class_f1"][name] = dict(zip(pre.classes_, f1s.tolist()))
        print(f"  {name:12s} acc={mean:.4f} [{lo:.4f},{hi:.4f}]")

    names = list(preds)
    if len(names) >= 2:
        out["mcnemar"] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]}_vs_{names[j]}"
                out["mcnemar"][key] = mcnemar(y, preds[names[i]], preds[names[j]])
                print(f"  McNemar {key}: p={out['mcnemar'][key]['p_value']:.4f}")
    save_json(out, metrics_dir(cfg) / "statistics.json")


if __name__ == "__main__":
    main()
