# Copyright 2021-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import inspect

import numpy as np

import quantumflow as qf


def test_base_abstract() -> None:
    """Make sure base classes are in fact abstract"""
    assert inspect.isabstract(qf.QuantumOperation)
    assert inspect.isabstract(qf.QuantumGate)
    assert inspect.isabstract(qf.QuantumStdGate)
    assert inspect.isabstract(qf.QuantumStdCtrlGate)
    assert inspect.isabstract(qf.QuantumComposite)


def test_Gate_run() -> None:
    ket = qf.zero_state([0, 1, 2])
    ket = qf.X(1).run(ket)
    assert ket.tensor[0, 1, 0] == 1
    ket = qf.CNot(1, 2).run(ket)
    assert ket.tensor[0, 1, 0] == 0
    assert ket.tensor[0, 1, 1] == 1


def test_Gate_permute() -> None:
    gate0 = qf.CNot(0, 1)

    backwards_cnot = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    gate10 = gate0.permute([1, 0])
    assert np.allclose(
        gate10.operator,
        backwards_cnot,
    )


def test_Gate_matmul() -> None:
    gate0 = qf.CNot(0, 1) @ qf.CNot(0, 1)
    assert qf.almost_identity(gate0)

    gate1 = qf.CNot(0, 1) @ qf.CNot(1, 0) @ qf.CNot(0, 1)
    assert qf.gates_close(gate1, qf.Swap(0, 1))

    gate2 = gate1 @ gate1
    assert qf.almost_identity(gate2)

    theta0 = 0.34
    theta1 = 0.11
    gate3 = qf.Rx(theta1, "a") @ qf.Rx(theta0, "a")
    assert qf.gates_close(gate3, qf.Rx(theta0 + theta1, "a"))

    gate4 = qf.I(0) @ qf.I(1)
    assert gate4.qubits == (1, 0)
    assert qf.almost_identity(gate4)


def test_Gate_relable() -> None:
    gate0 = qf.Unitary.from_gate(qf.CNot(1, 0))
    gate1 = gate0.relabel({0: "a", 1: "b"})
    assert gate1.qubits == ("b", "a")
    assert qf.gates_close(gate1, qf.Unitary.from_gate(qf.CNot("b", "a")))


def test_StdGate_relabel() -> None:
    gate0 = qf.CNot(1, 0)
    gate1 = gate0.relabel({0: "a", 1: "b"})
    assert gate1.qubits == ("b", "a")
    assert qf.gates_close(gate1, qf.CNot("b", "a"))


# fin
