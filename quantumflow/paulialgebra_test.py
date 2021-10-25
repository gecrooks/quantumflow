# Copyright 2019-, Gavin E. Crooks
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Unit tests for quantumflow.paulialgebra
"""

from itertools import product

import numpy as np
import pytest
import scipy.linalg

import quantumflow as qf
from quantumflow.paulialgebra import PAULI_OPS


def test_term() -> None:
    x = qf.Pauli.term([0], "X", -1)
    assert x._terms == (((0,), "X", -1),)

    x = qf.Pauli.term([1, 0, 5], "XYZ", 2j)
    assert x._terms == (((0, 1, 5), "YXZ", (0 + 2j)),)

    with pytest.raises(ValueError):
        qf.Pauli.term([1, 0, 5], "BYZ", 2j)


def test_pauli_str() -> None:
    x = qf.Pauli.term([2], "X", -1)
    assert str(x) == "- X(2)"
    assert repr(x) == "Pauli(((2,), 'X', -1))"


def test_sum() -> None:
    x = qf.X(1)
    y = 2.0 * qf.Y(2)
    s = qf.pauli_sum(x, y)
    assert s == qf.Pauli(([1], "X", 1.0), ([2], "Y", 2.0))

    s2 = qf.pauli_sum(x, x)
    assert s2 == qf.Pauli(([1], "X", 2))

    s3 = qf.pauli_sum(x, x, y)
    s4 = qf.pauli_sum(y, x, x)

    assert s3 == s4
    assert s3 == qf.Pauli(([1], "X", 2), ([2], "Y", 2))

    qf.pauli_sum(x, x, x)


def test_add() -> None:
    x = qf.X(1)
    y = 2.0 * qf.Y(2)
    s = y + x
    assert s == qf.Pauli(([1], "X", 1), ([2], "Y", 2))


def test_sub() -> None:
    x = qf.X(1)
    y = qf.Y(2) * 2.0
    s = x - y
    assert s == qf.Pauli(([1], "X", 1), ([2], "Y", -2))

    s = 2 - y
    assert str(s) == "+2 -2.0 Y(2)"


def test_hash() -> None:
    x = qf.X(5)
    d = {x: 4}
    assert d[x] == 4


def test_product() -> None:
    p = qf.pauli_product(qf.X(0), qf.Y(0))
    assert p == qf.Z(0) * 1j

    p = qf.pauli_product(qf.Y(0), qf.X(0))
    assert p == qf.Z(0) * -1j

    p = qf.pauli_product(qf.X(0), qf.Y(1))
    assert p == qf.Pauli.term([0, 1], "XY", 1)

    p = qf.pauli_product(qf.X(0), qf.Y(1), qf.Y(0))
    assert p == qf.Pauli.term([0, 1], "ZY", 1j)

    p = qf.pauli_product(qf.Y(0), qf.X(0), qf.Y(1))
    assert p == qf.Pauli.term([0, 1], "ZY", -1j)


def test_scalar() -> None:
    a = qf.Pauli.scalar(1.0)
    assert a.is_scalar()
    assert a.is_identity()

    b = qf.Pauli.scalar(2.0)
    assert b + b == qf.Pauli.scalar(4.0)
    assert b * b == qf.Pauli.scalar(4.0)

    assert -b == qf.Pauli.scalar(-2.0)
    assert +b == qf.Pauli.scalar(2.0)

    assert b * 3 == qf.Pauli.scalar(6.0)
    assert 3 * b == qf.Pauli.scalar(6.0)

    assert qf.X(0) * 2 == qf.X(0) * 2

    x = qf.X(0) + qf.Y(1)
    assert not x.is_scalar()
    assert not x.is_identity()

    c = qf.X(0) * qf.Y(1)
    assert not c.is_scalar()
    assert not c.is_identity()


def test_zero() -> None:
    z = qf.Pauli.scalar(0.0)
    assert z.is_zero()
    assert z.is_scalar()

    z2 = qf.Pauli.zero()
    assert z == z2

    assert qf.X(0) - qf.X(0) == qf.Pauli.zero()


def test_merge_sum() -> None:
    p = qf.pauli_sum(qf.Pauli.term([1], "Y", 3), qf.Pauli.term([1], "Y", 2))

    assert len(p._terms) == 1
    assert p._terms[0][2] == 5


def test_power() -> None:
    p = qf.X(0) + qf.Y(1) + qf.Pauli.term([2, 3], "XY")

    assert p ** 0 == qf.Pauli.identity()
    assert p ** 1 == p
    assert p * p == p ** 2
    assert p * p * p == p ** 3
    assert p * p * p * p * p * p * p * p * p * p == p ** 10

    with pytest.raises(ValueError):
        p ** -1

    p ** 400


def test_simplify() -> None:
    t1 = qf.Z(0) * qf.Z(1)
    t2 = qf.Z(0) * qf.Z(1)
    assert (t1 + t2) == 2 * qf.Z(0) * qf.Z(1)


def test_dont_simplify() -> None:
    t1 = qf.Z(0) * qf.Z(1)
    t2 = qf.Z(2) * qf.Z(3)
    assert (t1 + t2) != 2 * qf.Z(0) * qf.Z(1)


def test_zero_term() -> None:
    qubit_id = 0
    coefficient = 10
    ps = qf.I(qubit_id) + qf.X(qubit_id)
    assert coefficient * qf.Pauli.zero() == qf.Pauli.zero()
    assert qf.Pauli.zero() * coefficient == qf.Pauli.zero()
    assert qf.Pauli.zero() * qf.Pauli.identity() == qf.Pauli.zero()
    assert qf.Pauli.zero() + qf.Pauli.identity() == qf.Pauli.identity()
    assert qf.Pauli.zero() + ps == ps
    assert ps + qf.Pauli.zero() == ps


def test_neg() -> None:
    ps = qf.Y(0) - qf.X(0)
    ps -= qf.Z(1)
    ps = ps - 3
    _ = 3 + ps


def test_check_commutation() -> None:
    term1 = qf.X(0) * qf.X(1)
    term2 = qf.Y(0) * qf.Y(1)
    term3 = qf.Y(0) * qf.Z(2)

    assert qf.paulis_commute(term2, term3)
    assert qf.paulis_commute(term2, term3)
    assert not qf.paulis_commute(term1, term3)


def test_commuting_sets() -> None:
    term1 = qf.X(0) * qf.X(1)
    term2 = qf.Y(0) * qf.Y(1)
    term3 = qf.Y(0) * qf.Z(2)
    ps = term1 + term2 + term3
    pcs = qf.pauli_commuting_sets(ps)
    assert len(pcs) == 2

    pcs = qf.pauli_commuting_sets(term1)
    assert len(pcs) == 1


def test_get_qubits() -> None:
    term = qf.Z(0) * qf.X(1)
    assert term.qubits == (0, 1)
    term += qf.Y(10) + qf.X(2) + qf.Y(2)
    assert term.qubits == (0, 1, 2, 10)

    sum_term = qf.Pauli.term([0], "X", 0.5) + 0.5j * qf.Pauli.term(
        [10], "Y"
    ) * qf.Pauli.term([0], "Y", 0.5j)
    assert sum_term.qubits == (0, 10)


def test_check_commutation_rigorous() -> None:
    # more rigorous test.  Get all operators in Pauli group
    N = 3
    qubits = list(range(N))
    ps = [qf.Pauli.term(qubits, "".join(ops)) for ops in product(PAULI_OPS, repeat=N)]

    commuting = []
    non_commuting = []
    for left, right in product(ps, ps):
        if left * right == right * left:
            commuting.append((left, right))
        else:
            non_commuting.append((left, right))

    # now that we have our sets let's check against our code.
    for left, right in non_commuting:
        assert not qf.paulis_commute(left, right)

    for left, right in commuting:
        assert qf.paulis_commute(left, right)


def test_isclose() -> None:
    x = qf.X(1)
    y = qf.Y(2) * 2.0
    y2 = qf.Y(2) * 2.00000001

    assert qf.paulis_close(x, x)
    assert not qf.paulis_close(x, y)
    assert qf.paulis_close(y, y2)


# FIXME: More tests here
def test_run() -> None:
    x = qf.X(1)
    y = qf.Y(2) * 1.2
    s = x + y

    ket0 = qf.zero_state([0, 1, 2])
    _ = s.run(ket0)



# Start GEC 2021


def test_pauli_gates() -> None:
    assert qf.I(0)._terms == (((), "", 1),)
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
    print(repr(elem))
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
