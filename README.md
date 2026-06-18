# QSentry

**A telemetry-driven intelligent security layer for resilient quantum communication networks.**

QSentry continuously audits the optical control plane of Twin-Field QKD (TF-QKD)
links and flags implementation attacks. It separates two tasks that are often
(incorrectly) merged: **closed-set detection** of known attacks and **open-set
detection** of novel ones — and wraps them in a **trust layer** that monitors
calibration and turns model scores into operator-facing alarms. This repository
is the complete, reproducible bundle behind the QUANCOM 2026 paper *"QSentry: A
Telemetry-Driven Intelligent Security Layer for Resilient Quantum Communication
Networks."*

> The benchmark is a **physically-grounded simulation** (its phase-basis QBER
> channel is computed from interference physics), not a hardware capture. Numbers
> are meaningful for relative comparison and methodology; see `docs/DATASHEET.md`.

---

## What's inside

```
src/qsentry/      physics generator, datasets, 4 models, training, metrics,
                  evaluation, robustness, operating, interpretability, viz, tables
scripts/          numbered pipeline (00–13) + run_all.sh / run_smoke.sh
configs/          data, model, and experiment configs (full + smoke)
paper/            LNCS manuscript (main.tex), references, example figures/tables
docs/             REPRODUCIBILITY, DATASHEET, ARCHITECTURE, PACKAGING
tests/            unit tests for physics, datasets, models, metrics
```

Four model families: **Transformer**, **LSTM**, **hybrid QLSTM** (PennyLane VQC
gates with the parameter-shift rule), and a one-class **Autoencoder**.

Four domains: `clean`, `drift` (Ornstein–Uhlenbeck phase drift), `asym`
(asymmetric loss), `unknown` (mixed shift + a held-out novelty class for
zero-shot evaluation). Windows are `T×F = 32×12`.

## Quickstart

```bash
# 1. install (CPU is fine)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. verify everything end-to-end on tiny data (minutes)
bash scripts/run_smoke.sh

# 3. full publication-scale run (hours on CPU)
bash scripts/run_all.sh configs/experiment/full.yaml
```

Outputs land in `results/metrics/*.json`, `results/checkpoints/*.pt`,
`paper/figures/*.pdf`, and `paper/tables/*.tex`.

## Reproduce → results → paper

1. **Generate datasets** — `scripts/00_generate_data.py`
2. **Train models** — `scripts/01_preprocess.py`, `scripts/02_train.py`
3. **Run evaluations** — `scripts/03`–`scripts/10` (closed-set, cross-domain,
   open-set, calibration, robustness, operating, interpretability, ablation)
4. **Figures & tables** — `scripts/11_make_figures.py`, `scripts/12_make_tables.py`
5. **Insert into manuscript** — figures/tables are written into `paper/figures/`
   and `paper/tables/`; the manuscript `\includegraphics`/`\input`s them.
6. **Compile the paper** — see `paper/README_PAPER.md` (download `llncs.cls` +
   `splncs04.bst`, then `pdflatex`/`bibtex`).

The whole chain is a single command: `bash scripts/run_all.sh <config>`.

## Headline findings (preserved from the study)

- The **Transformer** is the strongest and most robust closed-set detector across
  every domain (macro-F1 0.9990 → 0.9262 → 0.8686 → 0.8993 from clean to unknown).
- The **QLSTM** is competitive but **not** superior — no empirical quantum
  advantage in this benchmark — and is far costlier per window under simulation.
- The **Autoencoder** is best used as a **ranking** novelty detector (high
  ROC-AUC / average precision) rather than a thresholded classifier.
- **Calibration degrades under domain shift**, robustness degrades **gracefully**,
  the classical models meet **control-plane latency** budgets, and
  **physically-meaningful channels** (phase-lock error, QBER phase, reference
  power, coincidence rate) carry the decisive signal.

## Notes

- `pennylane==0.36.0` needs `autoray==0.6.12`; `torch 2.2.x` needs `numpy<2`.
  Both are pinned in `requirements.txt`.
- The QLSTM defaults to the **parameter-shift** rule (faithful to the paper but
  slow under simulation). Set `params.diff_method: backprop` in
  `configs/models/qlstm.yaml` for fast iteration; the smoke config already does.
- Run all scripts from the repository root.

## License

MIT — see `LICENSE`. If you use this work, see `CITATION.cff`.
