import numpy as np
from qsentry.physics import (CHANNELS, CLOSED_SET_CLASSES, N_CHANNELS,
                             UNKNOWN_CLASS, make_generator)
from qsentry.seeding import new_rng


def test_channel_count():
    assert N_CHANNELS == 12
    assert len(CHANNELS) == 12
    assert len(set(CHANNELS)) == 12


def test_window_shape_and_finiteness():
    gen = make_generator(seq_len=32)
    rng = new_rng(0)
    x = gen.sample_window("normal", rng)
    assert x.shape == (32, N_CHANNELS)
    assert np.isfinite(x).all()


def test_qber_phase_from_physics():
    # phase QBER must equal (1 - V cos(phi))/2 + e_opt within noise bounds
    gen = make_generator(seq_len=32)
    rng = new_rng(1)
    x = gen.sample_window("normal", rng)
    ci = {c: i for i, c in enumerate(CHANNELS)}
    v = x[:, ci["visibility"]]
    phi = x[:, ci["phase_lock_error_rad"]]
    q = x[:, ci["qber_phase"]]
    recon = (1 - v * np.cos(phi)) / 2 + gen.cfg.e_opt
    assert np.allclose(q, recon, atol=1e-4)


def test_visibility_and_qber_bounds():
    gen = make_generator(seq_len=32, attack_intensity=2.0)
    rng = new_rng(2)
    for cls in CLOSED_SET_CLASSES + (UNKNOWN_CLASS,):
        x = gen.sample_class(cls, 4, rng)
        ci = {c: i for i, c in enumerate(CHANNELS)}
        assert (x[:, :, ci["visibility"]] >= 0).all()
        assert (x[:, :, ci["visibility"]] <= 1).all()
        assert (x[:, :, ci["qber_phase"]] >= 0).all()


def test_attack_changes_distribution():
    gen = make_generator(seq_len=32)
    rng = new_rng(3)
    normal = gen.sample_class("normal", 32, rng).mean(0)
    blind = gen.sample_class("detector_blinding_attack", 32, rng).mean(0)
    ci = {c: i for i, c in enumerate(CHANNELS)}
    # detector blinding should raise the detector count rate on average
    assert blind[:, ci["detector_count_rate"]].mean() > normal[:, ci["detector_count_rate"]].mean()
