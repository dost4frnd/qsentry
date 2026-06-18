"""Hybrid quantum LSTM (QLSTM) — Qiskit circuit definition, PyTorch statevector backend.

The VQC is *defined* using Qiskit's ``QuantumCircuit`` API (replacing
PennyLane) and *executed* via a differentiable statevector simulation in
PyTorch.  Gradients flow through PyTorch autograd (equivalent to Qiskit Aer
statevector + backprop mode), avoiding the PennyLane batched-tape limitation
and the overhead of external per-step simulator calls.

Use :func:`to_qiskit_circuit` to obtain a Qiskit ``QuantumCircuit`` for
inspection, hardware transpilation, or Qiskit Aer simulation.

Circuit structure (per layer, Eqs. 10-12 of the manuscript):
  RY(x[q])           for q in 0..n_qubits-1  — AngleEmbedding (data re-upload)
  RX(w) RY(w) RZ(w)  for q in 0..n_qubits-1  — trainable rotations
  CX(q, (q+1)%n)     for q in 0..n_qubits-1  — ring entanglement
Readout: <Z_q> for q in 0..n_qubits-1

The ``diff_method`` parameter is accepted for config/API compatibility;
differentiation always uses PyTorch autograd (statevector backprop).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

GATES = ("f", "i", "c", "o")


# --------------------------------------------------------------------------- #
# Quantum gate helpers (operate on complex64 tensors)
# --------------------------------------------------------------------------- #

def _ry(theta: torch.Tensor) -> torch.Tensor:
    """RY rotation matrix.  theta: () or (B,) → (2,2) or (B,2,2) complex64."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    if theta.dim() == 0:
        return torch.stack([torch.stack([c, -s]),
                            torch.stack([s,  c])]).to(torch.complex64)
    return torch.stack([torch.stack([ c, -s], dim=-1),
                        torch.stack([ s,  c], dim=-1)], dim=-2).to(torch.complex64)


def _rx(theta: torch.Tensor) -> torch.Tensor:
    """RX rotation matrix.  theta: () or (B,) → (2,2) or (B,2,2) complex64."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    z = torch.zeros_like(c)
    if theta.dim() == 0:
        return torch.stack([
            torch.stack([torch.complex(c, z),  torch.complex(z, -s)]),
            torch.stack([torch.complex(z, -s), torch.complex(c, z)])
        ])
    return torch.stack([
        torch.stack([torch.complex(c, z),  torch.complex(z, -s)], dim=-1),
        torch.stack([torch.complex(z, -s), torch.complex(c, z)],  dim=-1),
    ], dim=-2)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    """RZ rotation matrix.  theta: () or (B,) → (2,2) or (B,2,2) complex64."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    z = torch.zeros_like(c)
    if theta.dim() == 0:
        return torch.stack([
            torch.stack([torch.complex(c, -s), torch.complex(z, z)]),
            torch.stack([torch.complex(z, z),  torch.complex(c, s)])
        ])
    return torch.stack([
        torch.stack([torch.complex(c, -s), torch.complex(z, z)], dim=-1),
        torch.stack([torch.complex(z, z),  torch.complex(c, s)], dim=-1),
    ], dim=-2)


def _apply_gate(state: torch.Tensor, gate: torch.Tensor, qubit: int,
                n_qubits: int) -> torch.Tensor:
    """Apply a 2×2 gate to ``qubit`` of a batched statevector.

    state : (B, 2^n) complex64
    gate  : (2, 2) [shared across batch] or (B, 2, 2) [per-sample]
    Returns (B, 2^n) complex64.
    """
    B = state.shape[0]
    # Reshape to (B, 2, 2, ..., 2) — one axis per qubit
    sv = state.reshape(B, *([2] * n_qubits))

    # Bring target qubit's axis to position 1 (swap with dim 1)
    perm = list(range(n_qubits + 1))
    perm[1], perm[qubit + 1] = perm[qubit + 1], perm[1]
    sv = sv.permute(perm)
    shape = sv.shape                         # (B, 2, d1, d2, ...)

    sv_r = sv.reshape(B, 2, -1)             # (B, 2, R)
    if gate.dim() == 2:                      # shared gate
        new_sv = torch.einsum("ij,bjk->bik", gate, sv_r)
    else:                                    # per-sample gate (B, 2, 2)
        new_sv = torch.bmm(gate, sv_r)

    new_sv = new_sv.reshape(shape)
    # Undo the permutation
    inv = [0] * (n_qubits + 1)
    for dst, src in enumerate(perm):
        inv[src] = dst
    new_sv = new_sv.permute(inv)
    return new_sv.reshape(B, 2 ** n_qubits)


@lru_cache(maxsize=None)
def _cnot_perm(n_qubits: int, control: int, target: int) -> list[int]:
    """Precompute the CNOT basis-state permutation (big-endian qubit order)."""
    n = 2 ** n_qubits
    perm = list(range(n))
    ctrl_bit = n_qubits - 1 - control
    tgt_mask = 1 << (n_qubits - 1 - target)
    for i in range(n):
        if (i >> ctrl_bit) & 1:
            perm[i] = i ^ tgt_mask
    return perm


def _apply_cnot(state: torch.Tensor, control: int, target: int,
                n_qubits: int) -> torch.Tensor:
    """CX gate via a fixed index permutation — no learnable parameters."""
    perm = _cnot_perm(n_qubits, control, target)
    return state[:, perm]


def _expval_z(state: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Pauli-Z expectation value on ``qubit``.  Returns (B,) float32."""
    n_states = 2 ** n_qubits
    bit_pos = n_qubits - 1 - qubit
    signs = torch.tensor(
        [(-1.0) ** ((i >> bit_pos) & 1) for i in range(n_states)],
        dtype=torch.float32, device=state.device,
    )
    probs = state.real.pow(2) + state.imag.pow(2)  # (B, 2^n)
    return (probs * signs).sum(dim=1)              # (B,)


# --------------------------------------------------------------------------- #
# Qiskit circuit export
# --------------------------------------------------------------------------- #

def to_qiskit_circuit(n_qubits: int, n_layers: int,
                      weights: Sequence[float] | None = None) -> QuantumCircuit:
    """Return a Qiskit ``QuantumCircuit`` matching the VQC structure.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    n_layers:
        Number of circuit layers.
    weights:
        Optional flat array of ``n_layers * n_qubits * 3`` numeric values to
        bind as trainable rotation angles.  If ``None``, Qiskit
        ``ParameterVector`` entries are left symbolic (useful for transpilation
        and visualisation).
    """
    x = ParameterVector("x", n_qubits)              # input (embedding) angles
    w = ParameterVector("w", n_layers * n_qubits * 3)  # trainable weights

    qc = QuantumCircuit(n_qubits)
    w_idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(x[q], q)                           # AngleEmbedding
        for q in range(n_qubits):
            qc.rx(w[w_idx],     q)                   # trainable Rx
            qc.ry(w[w_idx + 1], q)                   # trainable Ry
            qc.rz(w[w_idx + 2], q)                   # trainable Rz
            w_idx += 3
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)             # ring entanglement

    if weights is not None:
        weights = list(weights)
        bindings = {w[i]: weights[i] for i in range(len(weights))}
        qc = qc.assign_parameters(bindings)

    return qc


# --------------------------------------------------------------------------- #
# Variational quantum circuit layer
# --------------------------------------------------------------------------- #

class VQCLayer(nn.Module):
    """Differentiable VQC implementing Eqs. 10-12 of the manuscript.

    Input : ``(B, n_qubits)`` — bounded angles from classical pre-projection.
    Output: ``(B, n_qubits)`` — Pauli-Z expectation values in ``[-1, 1]``.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Trainable rotation angles: (n_layers, n_qubits, 3) for Rx Ry Rz
        self.weights = nn.Parameter(
            torch.empty(n_layers, n_qubits, 3).uniform_(-math.pi, math.pi)
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        B, n = angles.shape[0], self.n_qubits
        # Initialise |0...0⟩
        state = torch.zeros(B, 2 ** n, dtype=torch.complex64, device=angles.device)
        state[:, 0] = 1.0 + 0j

        for layer_idx in range(self.n_layers):
            w = self.weights[layer_idx]          # (n_qubits, 3)

            # AngleEmbedding: RY(x[q]) per qubit — input angles, batched
            for q in range(n):
                state = _apply_gate(state, _ry(angles[:, q]), q, n)

            # Trainable rotations: RX RY RZ per qubit — shared across batch
            for q in range(n):
                state = _apply_gate(state, _rx(w[q, 0]), q, n)
                state = _apply_gate(state, _ry(w[q, 1]), q, n)
                state = _apply_gate(state, _rz(w[q, 2]), q, n)

            # Ring entanglement: CX(q, (q+1)%n)
            for q in range(n):
                state = _apply_cnot(state, q, (q + 1) % n, n)

        return torch.stack([_expval_z(state, q, n) for q in range(n)], dim=1)


# --------------------------------------------------------------------------- #
# QLSTM cell and classifier
# --------------------------------------------------------------------------- #

class QLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_qubits: int = 4,
                 n_layers: int = 1, diff_method: str = "parameter-shift"):
        super().__init__()
        self.hidden_size = hidden_size
        cat = input_size + hidden_size
        self.pre  = nn.ModuleDict({g: nn.Linear(cat, n_qubits) for g in GATES})
        self.vqc  = nn.ModuleDict({g: VQCLayer(n_qubits, n_layers) for g in GATES})
        self.post = nn.ModuleDict({g: nn.Linear(n_qubits, hidden_size) for g in GATES})

    def _gate(self, g: str, v: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(self.pre[g](v)) * math.pi  # → [-π, π]
        q_out  = self.vqc[g](angles)                   # (B, n_qubits)
        return self.post[g](q_out)

    def forward(self, x_t: torch.Tensor, state):
        h, c = state
        v = torch.cat([x_t, h], dim=-1)
        f = torch.sigmoid(self._gate("f", v))
        i = torch.sigmoid(self._gate("i", v))
        g = torch.tanh(   self._gate("c", v))
        o = torch.sigmoid(self._gate("o", v))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class QLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 8,
                 n_qubits: int = 4, n_layers: int = 1,
                 diff_method: str = "parameter-shift", **_):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = QLSTMCell(n_features, hidden_size, n_qubits, n_layers,
                              diff_method)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        b, t, _ = x.shape
        h = x.new_zeros(b, self.hidden_size)
        c = x.new_zeros(b, self.hidden_size)
        for step in range(t):
            h, c = self.cell(x[:, step, :], (h, c))
        logits = self.head(h)
        if return_embedding:
            return logits, h
        return logits

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        _, emb = self.forward(x, return_embedding=True)
        return emb

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
