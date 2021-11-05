# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random

import quantumflow as qf

from ..config_test import REPS


def test_CX() -> None:
    gate0 = qf.CX(1, 4)

    assert gate0.control_qubits == (1,)
    assert gate0.control_qubit_nb == 1
    assert isinstance(gate0.target, qf.X)
    assert gate0.target.qubits == (4,)
    assert gate0.target.qubit_nb == 1

    # FIXME: More tests


def test_XY() -> None:
    gate = qf.XY(-0.1, 0, 1)
    assert qf.gates_close(gate, qf.ISwap(0, 1) ** 0.2)


def test_CV() -> None:
    gate0 = qf.CX(0, 1) ** 0.5
    gate1 = qf.CV(0, 1)
    assert qf.gates_close(gate0, gate1)

    gate2 = qf.CV_H(0, 1)
    assert qf.gates_close(gate0.H, gate2)

    assert qf.gates_close(gate1.H.H, gate1)


def test_CH() -> None:
    gate1 = qf.CH(0, 1)

    # I picked up this circuit for a CH gate from qiskit
    # qiskit/extensions/standard/ch.py
    # But it clearly far too long. CH is locally equivalent to CNOT,
    # so requires only one CNOT gate.
    circ2 = qf.Circuit(
        qf.H(1),
        qf.S_H(1),
        qf.CX(0, 1),
        qf.H(1),
        qf.T(1),
        qf.CX(0, 1),
        qf.T(1),
        qf.H(1),
        qf.S(1),
        qf.X(1),
        qf.S(0),
    )
    assert qf.gates_close(gate1, circ2.asgate())

    # Here's a better decomposition
    circ1 = qf.Circuit(qf.YPow(+0.25, 1), qf.CX(0, 1), qf.YPow(-0.25, 1))
    assert qf.gates_close(gate1, circ1.asgate())
    assert qf.circuits_close(circ1, circ2)


def test_cnot_reverse() -> None:
    # Hadamards reverse control on CX
    gate0 = qf.H(0) @ qf.Identity([0, 1])
    gate0 = qf.H(1) @ gate0
    gate0 = qf.CX(1, 0) @ gate0
    gate0 = qf.H(0) @ gate0
    gate0 = qf.H(1) @ gate0

    assert qf.gates_close(qf.CX(0, 1), gate0)


def test_xy() -> None:
    assert qf.gates_close(qf.XY(0, 0, 1), qf.Identity([0, 1]))
    assert qf.gates_close(qf.XY(-0.5, 0, 1), qf.ISwap(0, 1))
    assert qf.gates_close(qf.XY(0.5, 0, 1), qf.ISwap(0, 1).H)


def test_cv() -> None:
    gate0 = qf.CV(0, 1) ** 2
    assert qf.gates_close(gate0, qf.CX(0, 1))

    gate1 = qf.CV_H(0, 1) ** 2
    assert qf.gates_close(gate1, qf.CX(0, 1).H)


def test_CXPow() -> None:
    q0, q1 = 2, 3
    for _ in range(REPS):
        t = random.uniform(-4, +4)
        gate0 = qf.CXPow(t, q0, q1)
        gate1 = qf.ControlGate(qf.XPow(t, q1), [q0])
        gate2 = qf.CX(q0, q1) ** t
        gate3 = qf.CXPow(1, q0, q1) ** t

        assert qf.gates_close(gate0, gate1)
        assert qf.gates_close(gate0, gate2)
        assert qf.gates_close(gate0, gate3)
