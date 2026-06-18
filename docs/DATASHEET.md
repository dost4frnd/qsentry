# Datasheet: QSentry TF-QKD control-telemetry benchmark

Following the spirit of *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation
The benchmark supports research on telemetry-driven security monitoring for
Twin-Field QKD (TF-QKD) links in quantum communication networks. It is designed
to evaluate (a) closed-set detection of known implementation attacks, (b)
open-set detection of novel attacks, and (c) robustness/calibration under
realistic domain shift. It is **synthetic but physically grounded**: it is not a
hardware capture.

## Composition
- **Unit:** a monitoring *window* of shape `T x F = 32 x 12` (32 time steps, 12
  physical channels), flattened to 384 sequence features + 4 metadata columns
  (`sample_id, domain, split, label`) = 388 columns.
- **Channels (12):** phase-lock error (rad), phase-basis QBER, time-basis QBER,
  interference visibility, reference power, coincidence rate, mean photon number,
  detector count rate, dark-count rate, wavelength offset (pm), phase-drift rate,
  timing jitter (ps).
- **Closed-set classes (8):** normal, detector_blinding_attack,
  reference_light_tamper, wavelength_switching_attack, phase_drift_attack,
  asymmetric_loss_attack, synchronization_jitter_attack, combined_attack.
- **Novelty class (1, held out):** trojan_horse_probe — present only in the
  `unknown` test split, never in training.
- **Domains (4):** `clean`, `drift` (Ornstein–Uhlenbeck phase drift), `asym`
  (asymmetric loss), `unknown` (mixed shift + novelty).
- **Sizes (full config):** clean/drift/asym = 6,400 windows each
  (train 4,480 / val 960 / test 960); unknown = 7,200 (train 4,480 / val 960 /
  test 1,760, of which 1,640 attack and 120 normal).

## Collection / generation process
Windows are produced by `src/qsentry/physics.py`. The phase-basis QBER channel is
computed from visibility and phase-lock error via
`E = (1 - V cos(delta_phi))/2 + E_opt`, so the data manifold reflects
interference physics. Attacks apply class-specific, time-localized perturbations
to the relevant channels; magnitudes are loosely calibrated to published TF-QKD
operating points and to the wavelength-switching attack of Peng et al. (2025).
Domain shift is injected via OU drift and asymmetric attenuation with tunable
severity. Generation is fully seeded.

## Preprocessing / cleaning / labeling
Features are standardized with statistics fit on the **training split only**.
Labels are provided directly by the generator. An automated audit
(`audit_*.json`) verifies: no duplicate rows, no missing/infinite values, and no
`sample_id` leakage across splits.

## Recommended uses
Closed-set classification, open-set/zero-shot novelty detection, domain-shift
robustness studies, calibration/trust analysis, channel-importance and
interpretability studies, and as a template that can later ingest real telemetry.

## Limitations and caveats
- **Synthetic.** Absolute metrics reflect benchmark behaviour, not field
  performance. Use for relative comparison and methodology.
- Channel ranges are illustrative, not metrological.
- The novelty class is a single held-out attack type; real deployments face a
  broader, open-ended novelty distribution.

## Maintenance
Regenerate at any time via `scripts/00_generate_data.py`. Severity and sizes are
controlled in `configs/data/*.yaml`. Extending the channel set or attack classes
requires editing `src/qsentry/physics.py` (canonical `CHANNELS`,
`CLOSED_SET_CLASSES`, `UNKNOWN_CLASS`).
