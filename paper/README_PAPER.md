# Compiling the QSentry manuscript

The manuscript (`main.tex`) is written for the **Springer LNCS** format required by
QUANCOM 2026. Two official Springer files are **not** bundled here (they are
distributed by Springer for preparing LNCS submissions):

- `llncs.cls`
- `splncs04.bst`

### One-time setup (get the LNCS files)

Pick **one**:

1. **Overleaf (easiest).** Open the official template and either compile there or
   download `llncs.cls` and `splncs04.bst` into this `paper/` folder:
   https://www.overleaf.com/latex/templates/springer-lecture-notes-in-computer-science/kzwwpvhwnvfj
2. **Springer LaTeX ZIP.** Download from the LNCS author pages and copy
   `llncs.cls` + `splncs04.bst` here:
   https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines

### Build

From this `paper/` directory, once `llncs.cls` and `splncs04.bst` are present and
the figures/tables have been generated (see the top-level `README.md`):

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

This produces `main.pdf`.

### How results flow into the paper

- **Preserved experiments** (same-domain closed-set, cross-domain, open-set,
  dataset dimensions) are written **inline** in `main.tex` with the reported
  numbers, so the paper always shows the headline results.
- **New experiments** are pulled in automatically:
  - `\input{tables/tab_calibration.tex}`
  - `\input{tables/tab_efficiency.tex}`
  - `\input{tables/tab_channel_importance.tex}`
  - figures via `\includegraphics` from `figures/`.

The committed `figures/` and `tables/` are produced by the **smoke** run so the
document compiles immediately. To populate them with **publication-scale**
numbers, run the full pipeline from the repo root:

```bash
bash scripts/run_all.sh configs/experiment/full.yaml
```

and rebuild. You may also choose to replace the inline preserved tables with the
freshly generated `tables/tab_*_generated.tex` if you want the paper to report
your own full-run numbers end to end.

### Before camera-ready

- Replace the placeholder author block and ORCID IDs with the final author list
  (the order cannot change after the deadline).
- Reconcile the `peng` reference (and any other best-effort entries) in
  `references.bib` against the authoritative versions.
- Confirm the page count is within the 16-page LNCS research-paper limit.
