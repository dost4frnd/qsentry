import numpy as np
import torch
from qsentry.models import build_model
from qsentry.physics import N_CHANNELS


def _batch(n=6, t=32, f=N_CHANNELS):
    return torch.randn(n, t, f)


def test_transformer_forward_and_attention():
    m = build_model("transformer", n_features=N_CHANNELS, seq_len=32,
                    n_classes=8, params={"d_model": 32, "nhead": 2, "num_layers": 1})
    x = _batch()
    logits = m(x)
    assert logits.shape == (6, 8)
    attn = m.attention_maps(x)
    assert attn.shape == (32, 32)
    assert m.embed(x).shape[0] == 6


def test_lstm_forward():
    m = build_model("lstm", n_features=N_CHANNELS, seq_len=32, n_classes=8,
                    params={"hidden_size": 16})
    assert m(_batch()).shape == (6, 8)


def test_qlstm_forward_and_backward():
    m = build_model("qlstm", n_features=N_CHANNELS, seq_len=8, n_classes=8,
                    params={"hidden_size": 4, "n_qubits": 4, "n_layers": 1,
                            "diff_method": "backprop"})
    x = torch.randn(3, 8, N_CHANNELS)
    logits = m(x)
    assert logits.shape == (3, 8)
    loss = logits.sum()
    loss.backward()  # gradients must flow through the VQC gates
    grads = [p.grad is not None for p in m.parameters()]
    assert any(grads)


def test_autoencoder_reconstruction_error():
    m = build_model("autoencoder", n_features=N_CHANNELS, seq_len=32,
                    n_classes=8, params={"latent_dim": 8, "hidden": 32})
    x = _batch()
    rec = m(x)
    assert rec.shape == x.shape
    err = m.reconstruction_error(x)
    assert err.shape == (6,)
    assert (err >= 0).all()
