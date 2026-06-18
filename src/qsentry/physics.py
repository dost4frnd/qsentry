"""Physically-grounded TF-QKD control-telemetry generator.

This module synthesises multivariate control telemetry for a Twin-Field QKD
station.  Each window is a sequence of ``T`` timesteps over ``F = 12`` physical
channels.  The generator is *physically structured* rather than abstractly
adversarial: the phase-basis QBER channel is computed from the instantaneous
visibility and phase-lock error using the standard interferometric relation

    E_t = (1 - V_t cos(delta_phi_t)) / 2 + E_opt,                        (Eq. 1)

so the telemetry manifold reflects the same physics that the downstream models
must learn.  Attack classes perturb specific channels in ways that mirror the
implementation-attack literature (detector blinding, reference-light tamper,
wavelength switching, asymmetric loss, synchronisation jitter, ...), and the
held-out ``trojan_horse_probe`` class provides a genuine open-set novelty for
zero-shot evaluation.

Domain shift is injected through:
  * an Ornstein-Uhlenbeck (OU) non-stationary phase drift (``drift`` domain),
  * asymmetric channel attenuation (``asym`` domain),
  * a mixture of both plus the novel class (``unknown`` domain).

Severity of each shift is controllable, which the robustness sweep exploits.

The numeric ranges are loosely calibrated to published TF-QKD operating points
and to the wavelength-switching attack of Peng et al. (npj QI, 2025); they are
illustrative, not metrological.  See ``docs/DATASHEET.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Channel and class definitions
# --------------------------------------------------------------------------- #

#: The 12 physical telemetry channels, in canonical order (F axis).
CHANNELS: tuple[str, ...] = (
    "phase_lock_error_rad",   # OPLL residual phase error (rad)
    "qber_phase",             # phase-basis QBER (derived from Eq. 1)
    "qber_time",              # time-basis QBER
    "visibility",             # interference visibility V_t in [0, 1]
    "reference_power",        # reference / local-oscillator power (a.u.)
    "coincidence_rate",       # two-fold coincidence rate (a.u.)
    "mean_photon_number",     # source intensity mu
    "detector_count_rate",    # single-detector count rate (a.u.)
    "dark_count_rate",        # dark-count rate (a.u.)
    "wavelength_offset_pm",   # reference wavelength deviation (pm)
    "phase_drift_rate",       # slow phase drift rate (rad/step)
    "timing_jitter_ps",       # synchronisation jitter (ps)
)
N_CHANNELS = len(CHANNELS)
CH = {name: i for i, name in enumerate(CHANNELS)}

#: Closed-set classes used for supervised training (8).
CLOSED_SET_CLASSES: tuple[str, ...] = (
    "normal",
    "detector_blinding_attack",
    "reference_light_tamper",
    "wavelength_switching_attack",
    "phase_drift_attack",
    "asymmetric_loss_attack",
    "synchronization_jitter_attack",
    "combined_attack",
)

#: Held-out novelty class, present only in the ``unknown`` domain test split.
UNKNOWN_CLASS = "trojan_horse_probe"

ATTACK_CLASSES = tuple(c for c in CLOSED_SET_CLASSES if c != "normal")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class TelemetryConfig:
    """Knobs controlling telemetry synthesis and domain shift."""

    seq_len: int = 32
    # baseline (normal) channel means
    base: dict = field(default_factory=lambda: {
        "phase_lock_error_rad": 0.0,
        "qber_time": 0.018,
        "visibility": 0.985,
        "reference_power": 1.0,
        "coincidence_rate": 1.0,
        "mean_photon_number": 0.30,
        "detector_count_rate": 1.0,
        "dark_count_rate": 0.02,
        "wavelength_offset_pm": 0.0,
        "phase_drift_rate": 0.0,
        "timing_jitter_ps": 5.0,
    })
    # baseline per-channel white-noise std (fraction of typical scale)
    noise: dict = field(default_factory=lambda: {
        "phase_lock_error_rad": 0.004,
        "qber_time": 0.0015,
        "visibility": 0.004,
        "reference_power": 0.02,
        "coincidence_rate": 0.03,
        "mean_photon_number": 0.01,
        "detector_count_rate": 0.03,
        "dark_count_rate": 0.004,
        "wavelength_offset_pm": 0.4,
        "phase_drift_rate": 0.0008,
        "timing_jitter_ps": 0.8,
    })
    e_opt: float = 0.012          # optical-misalignment / dark-count QBER floor
    onset_frac: float = 0.45      # attack onset (fraction of the window)
    # ---- domain-shift severities (1.0 == nominal) ----
    drift_strength: float = 0.0   # OU phase-drift amplitude multiplier
    drift_theta: float = 0.15     # OU mean-reversion rate
    asym_strength: float = 0.0    # asymmetric-loss magnitude (0 == none)
    attack_intensity: float = 1.0 # global attack-signature scale


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ou_process(rng: np.random.Generator, n: int, theta: float, sigma: float,
                x0: float = 0.0) -> np.ndarray:
    """Discrete Ornstein-Uhlenbeck process of length ``n``."""
    x = np.empty(n, dtype=np.float64)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (0.0 - x[t - 1]) + sigma * rng.standard_normal()
    return x


def _qber_phase_from_physics(visibility: np.ndarray, phase_err: np.ndarray,
                             e_opt: float) -> np.ndarray:
    """Eq. (1): phase-basis QBER from visibility and phase-lock error."""
    return (1.0 - visibility * np.cos(phase_err)) / 2.0 + e_opt


def _step_ramp(seq_len: int, onset_frac: float) -> np.ndarray:
    """Time-localised activation: ~0 before onset, ramping to 1 afterwards."""
    onset = int(round(onset_frac * seq_len))
    ramp = np.zeros(seq_len, dtype=np.float64)
    if onset < seq_len:
        tail = seq_len - onset
        ramp[onset:] = np.clip(np.linspace(0.0, 1.0, tail) * 1.4, 0.0, 1.0)
    return ramp


# --------------------------------------------------------------------------- #
# Core generator
# --------------------------------------------------------------------------- #

class TelemetryGenerator:
    """Generates per-window TF-QKD telemetry for a given class and domain."""

    def __init__(self, cfg: TelemetryConfig | None = None):
        self.cfg = cfg or TelemetryConfig()

    # -- baseline -------------------------------------------------------- #
    def _baseline(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        T = cfg.seq_len
        x = np.zeros((T, N_CHANNELS), dtype=np.float64)

        # phase-lock error: small zero-mean noise (+ OU drift if any)
        ple = rng.normal(cfg.base["phase_lock_error_rad"],
                         cfg.noise["phase_lock_error_rad"], size=T)
        if cfg.drift_strength > 0:
            ou = _ou_process(rng, T, cfg.drift_theta,
                             0.012 * cfg.drift_strength)
            ple = ple + ou
        x[:, CH["phase_lock_error_rad"]] = ple

        # visibility (slightly degraded by |phase error|)
        vis = rng.normal(cfg.base["visibility"], cfg.noise["visibility"], size=T)
        vis = np.clip(vis - 0.25 * np.abs(ple), 0.0, 1.0)
        x[:, CH["visibility"]] = vis

        # phase-basis QBER from physics (Eq. 1)
        x[:, CH["qber_phase"]] = _qber_phase_from_physics(vis, ple, cfg.e_opt)

        # remaining channels: independent baselines
        for name in ("qber_time", "reference_power", "coincidence_rate",
                     "mean_photon_number", "detector_count_rate",
                     "dark_count_rate", "wavelength_offset_pm",
                     "phase_drift_rate", "timing_jitter_ps"):
            x[:, CH[name]] = rng.normal(cfg.base[name], cfg.noise[name], size=T)

        # phase_drift_rate tracks the derivative of the OU drift if present
        if cfg.drift_strength > 0:
            x[1:, CH["phase_drift_rate"]] += np.diff(ple)
        return x

    # -- domain shift ---------------------------------------------------- #
    def _apply_asymmetric_loss(self, x: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
        """Rescale flux-like channels to emulate asymmetric channel loss."""
        s = self.cfg.asym_strength
        if s <= 0:
            return x
        # asymmetric attenuation factor in (0, 1]; larger s => stronger loss
        atten = np.exp(-0.12 * s * (1.0 + 0.3 * rng.standard_normal()))
        for name in ("coincidence_rate", "detector_count_rate",
                     "reference_power"):
            x[:, CH[name]] *= atten
        # visibility/count-rate imbalance perturbs the phase QBER slightly
        x[:, CH["qber_phase"]] += 0.02 * s * (1.0 - atten)
        return x

    # -- attack signatures ----------------------------------------------- #
    def _apply_attack(self, x: np.ndarray, cls: str,
                      rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        T = cfg.seq_len
        g = cfg.attack_intensity
        ramp = _step_ramp(T, cfg.onset_frac)

        if cls == "normal":
            return x

        if cls == "detector_blinding_attack":
            # APD forced out of Geiger mode: count rate saturates high,
            # dark counts suppressed, QBER statistics distorted.
            x[:, CH["detector_count_rate"]] += ramp * g * (1.8 + 0.3 * rng.standard_normal())
            x[:, CH["dark_count_rate"]] -= ramp * g * 0.015
            x[:, CH["qber_time"]] += ramp * g * 0.05
            x[:, CH["qber_phase"]] += ramp * g * 0.04

        elif cls == "reference_light_tamper":
            # Reference / LO path manipulated: reference power and visibility
            # perturbed; clearest separation in the phase-QBER trace.
            x[:, CH["reference_power"]] += ramp * g * (0.35 + 0.1 * rng.standard_normal())
            x[:, CH["visibility"]] -= ramp * g * 0.06
            x[:, CH["qber_phase"]] += ramp * g * 0.06
            x[:, CH["phase_lock_error_rad"]] += ramp * g * 0.01

        elif cls == "wavelength_switching_attack":
            # Peng et al. 2025: small reference-wavelength change raises mu and
            # reference power while interference stays roughly intact.
            x[:, CH["wavelength_offset_pm"]] += ramp * g * (12.0 + 2.0 * rng.standard_normal())
            x[:, CH["mean_photon_number"]] += ramp * g * 0.08
            x[:, CH["reference_power"]] += ramp * g * 0.12
            x[:, CH["visibility"]] -= ramp * g * 0.012

        elif cls == "phase_drift_attack":
            # Injected phase excursions: phase-lock error and phase QBER rise.
            exc = _ou_process(rng, T, 0.08, 0.02 * g) + ramp * g * 0.03
            x[:, CH["phase_lock_error_rad"]] += exc
            x[:, CH["phase_drift_rate"]][1:] += np.diff(exc)
            x[:, CH["qber_phase"]] = _qber_phase_from_physics(
                x[:, CH["visibility"]], x[:, CH["phase_lock_error_rad"]], cfg.e_opt)

        elif cls == "asymmetric_loss_attack":
            # Deliberate transmittance imbalance (distinct from the asym domain).
            atten = np.exp(-0.2 * g)
            x[:, CH["coincidence_rate"]] *= (1.0 - ramp * (1.0 - atten))
            x[:, CH["detector_count_rate"]] *= (1.0 - ramp * (1.0 - atten) * 0.8)
            x[:, CH["reference_power"]] *= (1.0 - ramp * (1.0 - atten) * 0.5)

        elif cls == "synchronization_jitter_attack":
            # Diffuse timing perturbation.
            jit = _ou_process(rng, T, 0.05, 1.2 * g) + ramp * g * 4.0
            x[:, CH["timing_jitter_ps"]] += jit
            x[:, CH["coincidence_rate"]] -= ramp * g * 0.05
            x[:, CH["qber_time"]] += ramp * g * 0.02

        elif cls == "combined_attack":
            # Superposition of reference tamper + wavelength switch + jitter.
            x[:, CH["reference_power"]] += ramp * g * 0.2
            x[:, CH["wavelength_offset_pm"]] += ramp * g * 7.0
            x[:, CH["mean_photon_number"]] += ramp * g * 0.05
            x[:, CH["timing_jitter_ps"]] += ramp * g * 2.5
            x[:, CH["qber_phase"]] += ramp * g * 0.04

        elif cls == UNKNOWN_CLASS:  # trojan_horse_probe (open-set novelty)
            # Back-reflected probe: tiny visibility dip, reference-power ripple,
            # and a dark-count rise -- a signature absent from training.
            ripple = 0.08 * g * np.sin(np.linspace(0, 6 * np.pi, T)) * ramp
            x[:, CH["reference_power"]] += ripple
            x[:, CH["dark_count_rate"]] += ramp * g * 0.03
            x[:, CH["visibility"]] -= ramp * g * 0.02
            x[:, CH["detector_count_rate"]] += ripple * 0.5
        else:
            raise ValueError(f"Unknown class: {cls}")

        # keep physically-bounded channels in range
        x[:, CH["visibility"]] = np.clip(x[:, CH["visibility"]], 0.0, 1.0)
        x[:, CH["qber_phase"]] = np.clip(x[:, CH["qber_phase"]], 0.0, 0.5)
        x[:, CH["qber_time"]] = np.clip(x[:, CH["qber_time"]], 0.0, 0.5)
        x[:, CH["dark_count_rate"]] = np.clip(x[:, CH["dark_count_rate"]], 0.0, None)
        return x

    # -- public API ------------------------------------------------------ #
    def sample_window(self, cls: str, rng: np.random.Generator) -> np.ndarray:
        x = self._baseline(rng)
        x = self._apply_asymmetric_loss(x, rng)
        x = self._apply_attack(x, cls, rng)
        return x.astype(np.float32)

    def sample_class(self, cls: str, n: int,
                     rng: np.random.Generator) -> np.ndarray:
        out = np.empty((n, self.cfg.seq_len, N_CHANNELS), dtype=np.float32)
        for i in range(n):
            out[i] = self.sample_window(cls, rng)
        return out


def make_generator(seq_len: int = 32, drift_strength: float = 0.0,
                   asym_strength: float = 0.0, attack_intensity: float = 1.0,
                   **overrides) -> TelemetryGenerator:
    """Convenience factory building a :class:`TelemetryGenerator`."""
    cfg = TelemetryConfig(seq_len=seq_len, drift_strength=drift_strength,
                          asym_strength=asym_strength,
                          attack_intensity=attack_intensity)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return TelemetryGenerator(cfg)
