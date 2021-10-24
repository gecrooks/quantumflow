# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import numpy as np
import pytest
import scipy.linalg

import quantumflow as qf


def test_pauli_gates() -> None:
    assert qf.X(0)._terms == (((0,), "X", 1),)
    assert qf.Y(1)._terms == (((1,), "Y", 1),)
    assert qf.Z(2)._terms == (((2,), "Z", 1),)


def test_pauli_gate_mul() -> None:
    assert qf.X(0) * qf.Y(0) == qf.Z(0) * 1j
    assert qf.X(0) * qf.Z(0) == qf.Y(0) * -1j

    assert qf.Y(0) * qf.X(0) == -1j * qf.Z(0)
    assert qf.Y(0) * qf.Z(0) == qf.X(0) * 1j

    assert qf.Z(0) * qf.X(0) == qf.Y(0) * 1j
    assert qf.Z(0) * qf.Y(0) == qf.X(0) * -1j

    with pytest.raises(TypeError):
        qf.Z(0) * "NOT A THING"


def test_PauliElement_neg() -> None:
    ps = qf.Y(1) - qf.X(1)
    ps -= qf.Z(0)
    ps = -ps
    assert str(ps) == "+ Z(0) + X(1) - Y(1)"
    assert ps == (+ps)

    assert -qf.X(1) - 1 == -1 - qf.X(1)


def test_PauliElement_div() -> None:
    elem = qf.Y(1) + qf.X(1)
    elem *= 5
    elem /= 2
    assert str(elem) == "+2.5 X(1) +2.5 Y(1)"


def test_Pauli_zero() -> None:
    elem = qf.Pauli.term([2, 4], "XY", 0)
    assert elem._terms == ()

    with pytest.raises(ValueError):
        qf.Pauli.term([2, 4], "XQ", 0)


def test_Pauli_str() -> None:
    elem = qf.Pauli.term([2, 4], "XY", -1)
    assert str(elem) == "- X(2) Y(4)"
    term = 3.0 * qf.X(0) * qf.Y(1) * qf.Z(2)
    assert str(term) == "+3.0 X(0) Y(1) Z(2)"

    term = 3.0 * qf.X(0) * qf.X(0)
    assert str(term) == "+3.0"


def test_Pauli_repr() -> None:
    elem = qf.Pauli.term([2, 4], "XY", -1)
    elem2 = eval(repr(elem), {"Pauli": qf.Pauli})
    assert elem == elem2


def test_Pauli_relabel() -> None:
    term = qf.X(0) * qf.Y(1) * qf.Z(2)
    assert term.qubits == (0, 1, 2)
    term = term.on([5, 4, 3])
    assert term.qubits == (3, 4, 5)
    assert term == qf.X(5) * qf.Y(4) * qf.Z(3)
    term = term.relabel({5: "x", 4: "y", 3: "z"})
    assert term.qubits == ("x", "y", "z")

    with pytest.raises(ValueError):
        term.on([1, 2, 3, 4])


def test_Pauli_H() -> None:
    term = qf.X(0) * qf.Y(1) * qf.Z(2)
    assert term == term.H

    term = 1j * term
    assert term == -term.H


def test_pauli_decompose() -> None:
    gate = qf.X(0)
    H = gate.operator
    pl = qf.pauli_decompose(H)
    assert np.allclose(pl.operator, H)

    gate = qf.X(0)
    op = gate.operator
    H = -scipy.linalg.logm(op) / 1.0j
    pl = qf.pauli_decompose(H)
    assert np.allclose(pl.operator, H)

    N = 4
    gate2 = qf.RandomGate(list(range(N)))
    op = gate2.operator
    H = -scipy.linalg.logm(op) / 1.0j
    pl = qf.pauli_decompose(H)
    assert np.allclose(pl.operator, H)

    pl = qf.pauli_decompose(gate2.operator)
    assert np.allclose(pl.operator, gate2.operator)

    pl = qf.pauli_decompose(gate2.operator, qubits="ABCD")
    assert np.allclose(pl.operator, gate2.operator)
    assert pl.qubits == ("A", "B", "C", "D")

    with pytest.raises(ValueError):
        qf.pauli_decompose(gate2.operator, qubits="ABCDE")

    op = np.ones(shape=[2, 2, 2])
    with pytest.raises(ValueError):
        qf.pauli_decompose(op)

    op = np.ones(shape=[3, 3])
    with pytest.raises(ValueError):
        qf.pauli_decompose(op)


    N = 4
    gate2 = qf.RandomGate(list(range(N)))
    ham = gate2.hamiltonian
    

# fin
