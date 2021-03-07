# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np
import pytest
from sympy import Symbol

import quantumflow as qf

# == test Gate ==


def test_gate_mul() -> None:
    # three cnots same as one swap
    gate0 = qf.IdentityGate([0, 1])

    gate1 = qf.CNot(1, 0)
    gate2 = qf.CNot(0, 1)
    gate3 = qf.CNot(1, 0)

    gate = gate1 @ gate0
    gate = gate2 @ gate
    gate = gate3 @ gate
    assert qf.gates_close(gate, qf.Swap(0, 1))

    # Again, but with labels
    gate0 = qf.IdentityGate(["a", "b"])

    gate1 = qf.CNot("b", "a")
    gate2 = qf.CNot("a", "b")
    gate3 = qf.CNot("b", "a")

    gate = gate1 @ gate0
    gate = gate2 @ gate
    gate = gate3 @ gate
    assert qf.gates_close(gate, qf.Swap("a", "b"))

    gate4 = qf.X("a")
    _ = gate4 @ gate

    with pytest.raises(NotImplementedError):
        _ = gate4 @ 3  # type: ignore


def test_gate_permute() -> None:
    gate0 = qf.CNot(0, 1)
    gate1 = qf.CNot(1, 0)

    assert not qf.gates_close(gate0, gate1)

    gate2 = gate1.permute([0, 1])
    assert gate2.qubits == (0, 1)
    assert qf.gates_close(gate1, gate2)

    gate3 = qf.ISwap(0, 1)
    gate4 = gate3.permute([1, 0])
    assert qf.gates_close(gate3, gate4)


def test_gates_evolve() -> None:
    rho0 = qf.zero_state(3).asdensity()
    qf.H(0).evolve(rho0)


def test_gate_H() -> None:
    gate0 = qf.X(0)
    assert gate0.cv_hermitian
    assert gate0.H is gate0

    gate1 = qf.ISwap(0, 1)
    assert not gate1.cv_hermitian
    assert qf.gates_close(gate1.H.H, gate1)


def test_su() -> None:
    su = qf.Swap(0, 1).su()
    assert np.isclose(np.linalg.det(su.asoperator()), 1.0)


def test_interchangeable() -> None:
    assert qf.Swap(0, 1).cv_interchangeable
    assert not qf.CNot(0, 1).cv_interchangeable


def test_gate_symbolic_params() -> None:
    theta = Symbol("θ")

    gate0 = qf.Rz(theta, 1)
    assert str(gate0) == "Rz(θ) 1"

    gate1 = gate0 ** 4
    assert str(gate1) == "Rz(4*θ) 1"

    circ = qf.Circuit([gate0, gate1])
    print(circ)
    diag = qf.circuit_to_diagram(circ)
    assert diag == "1: ───Rz(θ)───Rz(4*θ)───\n"

    gate2 = gate0.resolve({"θ": 2})
    assert gate2.param("theta") == 2.0

    with pytest.raises(KeyError):
        _ = gate2.param("asldfh")


def test_gate_rewire() -> None:
    gate0 = qf.CNot(1, 0)
    gate1 = gate0.on("B", "A")
    assert gate1.qubits == ("B", "A")

    gate2 = gate1.rewire({"A": "a", "B": "b", "C": "c"})
    assert gate2.qubits == ("b", "a")

    with pytest.raises(ValueError):
        _ = gate0.on("B", "A", "C")


def test_join_gates() -> None:
    gate = qf.join_gates(qf.H(0), qf.X(1))
    ket = qf.zero_state(2)
    ket = gate.run(ket)
    ket = qf.H(0).run(ket)
    ket = qf.X(1).run(ket)

    assert qf.states_close(ket, qf.zero_state(2))


# == test Unitary ==


def test_unitary_exceptions() -> None:
    tensor = qf.CNot(1, 0).tensor

    with pytest.raises(ValueError):
        qf.Unitary(tensor, [0])

    with pytest.raises(ValueError):
        qf.Unitary(tensor, [0, 1, 2])


# fin
