# Reproducibility guide

This bundle regenerates every dataset, model, metric, figure, and table in the
paper from scratch. Two run modes are provided:

| Mode  | Config                          | Purpose                                   | Time |
|-------|---------------------------------|-------------------------------------------|------|
| smoke | `configs/experiment/smoke.yaml` | verify the pipeline end-to-end (tiny)     | minutes |
| full  | `configs/experiment/full.yaml`  | publication-scale datasets and training   | hours (CPU) |

## 1. Environment

Python 3.10–3.12. Install dependencies (CPU is sufficient):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate qsentry
```

Notes:
- `torch` is pinned to a version compatible with `numpy<2`. If you install a
  newer `torch`/`numpy` pairing, keep them ABI-compatible.
- `pennylane==0.36.0` requires `autoray==0.6.12` (pinned in `requirements.txt`);
  newer `autoray` breaks that PennyLane release.

## 2. Smoke test (recommended first)

```bash
bash scripts/run_smoke.sh
```

This runs all 14 stages on tiny data/models and writes example figures to
`paper/figures/` and tables to `paper/tables/` so the manuscript compiles.

## 3. Full run

```bash
bash scripts/run_all.sh configs/experiment/full.yaml
```

Stages (each can also be run individually with the same config argument):

```
00_generate_data     synthesize the four domains -> data/
01_preprocess        fit train-only scaler + label encoder
02_train             train Transformer, LSTM, QLSTM, Autoencoder
03_evaluate          same-domain closed-set metrics
04_cross_domain      clean-trained models across all domains
05_open_set          autoencoder novelty detection + ROC/PR/DET curves
06_calibration       ECE, reliability, temperature scaling
07_robustness        severity sweeps (drift / asym / attack intensity)
08_operating         latency + parameter budget
09_interpret         attention maps + t-SNE
10_channel_ablation  permutation channel importance
11_make_figures      render all figures -> paper/figures/
12_make_tables       render all LaTeX tables -> paper/tables/
13_statistics        bootstrap CIs + McNemar tests
```

> The QLSTM uses the parameter-shift rule by default (faithful to the paper).
> Parameter-shift is **slow** under state-vector simulation. For rapid
> iteration set `params.diff_method: backprop` (or `adjoint`) in
> `configs/models/qlstm.yaml`; the smoke config already does this.

## 4. Outputs

```
data/                     generated CSVs + integrity audits (full run)
results/metrics/*.json    all numeric results
results/checkpoints/*.pt  trained models + fitted preprocessor
paper/figures/*.pdf       publication figures
paper/tables/*.tex        LaTeX tables
```

## 5. Build the paper

See `paper/README_PAPER.md` (download `llncs.cls` + `splncs04.bst`, then
`pdflatex`/`bibtex`).

## 6. Determinism

Global seeding (`src/qsentry/seeding.py`) covers Python, NumPy, and PyTorch.
Exact bit-for-bit reproducibility can still vary across BLAS/PyTorch builds and
thread counts; qualitative results and reported metrics are stable across seeds
(see `13_statistics.py` for bootstrap confidence intervals).
