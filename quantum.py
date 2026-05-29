"""
GrafoPropagation v26-APEX — Quantum Learning-Rate Modulation

Encodes the epoch index into an 8-qubit PennyLane circuit and produces
a smoothed scalar multiplier in [0.7, 1.3] for learning-rate scaling.

(c) 2025-2026 Claudio Fernandes. All rights reserved.
"""

import math
from collections import deque

import torch
import pennylane as qml


def _choose_backend(name: str = "lightning.gpu", wires: int = 8):
    """Try GPU-accelerated simulator; fall back to CPU qubit simulator."""
    try:
        return qml.device(name, wires=wires)
    except Exception:
        return qml.device("lightning.qubit", wires=wires)


_dev_lr = _choose_backend("lightning.gpu", 8)


@qml.qnode(_dev_lr, interface="torch", diff_method=None)
def _qlr_circuit(epoch_idx):
    """Encode epoch as bit-string on 8 qubits, apply parameterised rotation
    and entangling layers, return ⟨Z⟩ on each qubit."""
    for i in range(8):
        if (epoch_idx >> i) & 1:
            qml.PauliX(wires=i)
    for i in range(8):
        qml.RY(0.5 + 0.1 * i, wires=i)
    for i in range(7):
        qml.CZ(wires=[i, i + 1])
    for i in range(8):
        qml.RX(0.7 - 0.05 * i, wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]


_lr_queue: deque = deque(maxlen=3)


def quantum_lr_modulation(epoch: int) -> float:
    """
    Encode epoch index into an 8-qubit circuit; return a smoothed scalar
    multiplier in the range [0.7, 1.3] suitable for learning-rate scaling.
    """
    v = 0.7 + 0.6 * (
        (torch.stack(_qlr_circuit(epoch)).sum().item() + 8.0) / 16.0
    )
    _lr_queue.append(v)
    return sum(_lr_queue) / len(_lr_queue)
