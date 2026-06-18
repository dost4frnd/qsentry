# Architecture overview

## Package layout (`src/qsentry/`)
```
physics.py      physically-grounded TF-QKD telemetry generator (Eq. 1, OU drift,
                asymmetric loss, attack signatures, held-out novelty class)
datasets.py     domain construction, 388-column flattened layout, leakage-free
                splits, train-only Preprocessor (scaler + closed-set encoder)
models/
  transformer.py  encoder classifier exposing attention maps + embeddings
  lstm.py         recurrent baseline
  qlstm.py        hybrid QLSTM (4 VQC gates via PennyLane TorchLayer)
  autoencoder.py  one-class reconstruction scorer (Stage 2)
  registry.py     build_model() factory
train.py        supervised + one-class training loops, checkpoint IO
metrics.py      closed-set, open-set, calibration (ECE/reliability), DET curves
evaluate.py     closed-set / cross-domain / open-set reports, channel importance
robustness.py   severity sweeps over drift / asym / attack intensity
operating.py    latency + parameter-budget profiling
interpret.py    attention extraction + t-SNE
viz.py          all matplotlib figures (vector PDF)
tables.py       all LaTeX (booktabs) tables
config.py, seeding.py   YAML config handling + global determinism
```

## Data flow
```
configs ─▶ 00 generate ─▶ data/*.csv ─▶ 01 preprocess ─▶ preprocessor.npz
                                              │
                                              ▼
                          02 train ─▶ checkpoints/*.pt
                                              │
        ┌─────────────────────────────────────┼───────────────────────────┐
        ▼                 ▼                     ▼                ▼           ▼
   03 closed-set    04 cross-domain      05 open-set      06 calibration  07 robustness
        └─────────────────────────────────────┼───────────────────────────┘
                                              ▼
                       08 operating · 09 interpret · 10 channel ablation
                                              ▼
                         results/metrics/*.json  ──▶  11 figures ──▶ paper/figures/
                                                 └─▶  12 tables  ──▶ paper/tables/
                                                 └─▶  13 statistics
```

## QSentry runtime design (the paper's system)
- **Stage 1 (closed-set detector):** Transformer (recommended) labels known
  attacks and localizes them in time.
- **Stage 2 (novelty detector):** one-class autoencoder scores out-of-vocabulary
  deviations by reconstruction error; threshold `tau = Q_0.95` of normal-train
  error, plus full operating curves.
- **Trust layer:** monitors ECE / reliability of Stage 1 and ranking quality of
  Stage 2; supports temperature scaling.
- **Network level:** replicate the per-link monitor per node; aggregate alarms at
  the controller (forward-looking extension).
