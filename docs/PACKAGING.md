# Packaging for release

## What to commit
- `src/`, `scripts/`, `configs/`, `tests/`, `docs/`, `paper/` (sources)
- `paper/figures/` and `paper/tables/` example outputs (so the paper compiles)
- `README.md`, `LICENSE`, `requirements.txt`, `environment.yml`,
  `pyproject.toml`, `Makefile`, `CITATION.cff`, `.gitignore`

## What NOT to commit (regenerated)
- `data/`, `data_smoke/` (generated datasets)
- `results/`, `results_smoke/` (metrics, checkpoints, logs)
- `.venv/`, `__pycache__/`

These are listed in `.gitignore`.

## Make a release ZIP
From the repo root:

```bash
make clean        # remove generated data/results/caches
make zip          # -> dist/qsentry-bundle.zip
```

Or manually:

```bash
zip -r qsentry-bundle.zip . \
  -x '*/__pycache__/*' '*.pyc' \
     './data/*' './data_smoke/*' \
     './results/*' './results_smoke/*' \
     './.venv/*' './dist/*'
```

## Artifacts & Demo track
QUANCOM 2026 has a separate Artifacts & Demo track. This bundle is structured for
it: a single `bash scripts/run_smoke.sh` reproduces the full workflow quickly,
`docs/REPRODUCIBILITY.md` documents the full run, and `docs/DATASHEET.md`
documents the dataset.
