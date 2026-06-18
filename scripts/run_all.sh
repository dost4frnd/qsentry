#!/usr/bin/env bash
# End-to-end pipeline. Usage: bash scripts/run_all.sh [configs/experiment/full.yaml]
set -euo pipefail
CFG="${1:-configs/experiment/full.yaml}"
cd "$(dirname "$0")/.."          # repo root
echo "### QSentry pipeline | config=$CFG"
python scripts/00_generate_data.py    "$CFG"
python scripts/01_preprocess.py       "$CFG"
python scripts/02_train.py            "$CFG"
python scripts/03_evaluate.py         "$CFG"
python scripts/04_cross_domain.py     "$CFG"
python scripts/05_open_set.py         "$CFG"
python scripts/06_calibration.py      "$CFG"
python scripts/07_robustness.py       "$CFG"
python scripts/08_operating.py        "$CFG"
python scripts/09_interpret.py        "$CFG"
python scripts/10_channel_ablation.py "$CFG"
python scripts/11_make_figures.py     "$CFG"
python scripts/12_make_tables.py      "$CFG"
python scripts/13_statistics.py       "$CFG"
echo "### Done. Figures in paper/figures, tables in paper/tables, metrics in results/."
