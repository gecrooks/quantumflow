# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
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
from quantumflow.paulialgebra import PAULI_OPS, sI, sX, sY, sZ


def test_term() -> None:
    x = qf.Pauli.term([0], "X", -1)
    assert x.terms == (((0,), "X", -1),)

    x = qf.Pauli.term([1, 0, 5], "XYZ", 2j)
    assert x.terms == (((0, 1, 5), "YXZ", (0 + 2j)),)

    with pytest.raises(ValueError):
        qf.Pauli.term([1, 0, 5], "BYZ", 2j)


def test_pauli_str() -> None:
    x = qf.Pauli.term([2], "X", -1)
    assert str(x) == "- sX(2)"
    assert repr(x) == "Pauli((((2,), 'X', -1),))"


def test_pauli_sigma() -> None:
    assert qf.Pauli.term([0], "X", 1) == qf.Pauli.sigma(0, "X")
    assert qf.Pauli.term([1], "Y", 2) == qf.Pauli.sigma(1, "Y", 2)
    assert qf.Pauli.term([2], "Z") == qf.Pauli.sigma(2, "Z")
    # assert qf.Pauli.term([3], 'I', 1) == qf.Pauli.sigma('I', 3) # FIXME


def test_sigmas() -> None:
    assert qf.sX(0).terms == (((0,), "X", 1),)
    assert qf.sY(1).terms == (((1,), "Y", 1),)
    assert qf.sZ(2).terms == (((2,), "Z", 1),)


def test_sum() -> None:
    x = qf.sX(1)
    y = qf.sY(2, 2.0)
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
    x = sX(1)
    y = sY(2, 2.0)
    s = y + x
    assert s == qf.Pauli(([1], "X", 1), ([2], "Y", 2))


def test_sub() -> None:
    x = sX(1)
    y = sY(2, 2.0)
    s = x - y
    assert s == qf.Pauli(([1], "X", 1), ([2], "Y", -2))

    s = 2 - y
    assert str(s) == "+(2+0j) +(-2+0j) sY(2)"


def test_cmp() -> None:
    x = sX(1)
    x2 = sX(1)
    y = sY(2, 2.0)
    y2 = sY(2, 2.0)
    assert x < y
    assert x2 <= y
    assert x <= x2
    assert x == x2
    assert y2 > x
    assert y >= y2

    with pytest.raises(TypeError):
        "foo" > y

    with pytest.raises(TypeError):
        "foo" >= y

    with pytest.raises(TypeError):
        "foo" < y

    with pytest.raises(TypeError):
        "foo" <= y

    assert not 2 == y


def test_hash() -> None:
    x = sX(5)
    d = {x: 4}
    assert d[x] == 4


def test_product() -> None:
    p = qf.pauli_product(sX(0), sY(0))
    assert p == sZ(0, 1j)

    p = qf.pauli_product(sY(0), sX(0))
    assert p == sZ(0, -1j)

    p = qf.pauli_product(sX(0), sY(1))
    assert p == qf.Pauli.term([0, 1], "XY", 1)

    p = qf.pauli_product(sX(0), sY(1), sY(0))
    assert p == qf.Pauli.term([0, 1], "ZY", 1j)

    p = qf.pauli_product(sY(0), sX(0), sY(1))
    assert p == qf.Pauli.term([0, 1], "ZY", -1j)


def test_mul() -> None:
    # TODO CHECK ALL PAULI MULTIPLICATIONS HERE
    assert sX(0) * sY(0) == sZ(0, 1j)


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

    assert sX(0) * 2 == sX(0, 2)

    x = sX(0) + sY(1)
    assert not x.is_scalar()
    assert not x.is_identity()

    c = sX(0) * sY(1)
    assert not c.is_scalar()
    assert not c.is_identity()


def test_zero() -> None:
    z = qf.Pauli.scalar(0.0)
    assert z.is_zero()
    assert z.is_scalar()

    z2 = qf.Pauli.zero()
    assert z == z2

    assert sX(0) - sX(0) == qf.Pauli.zero()


def test_merge_sum() -> None:
    p = qf.pauli_sum(qf.Pauli.term([1], "Y", 3), qf.Pauli.term([1], "Y", 2))

    assert len(p) == 1
    assert p.terms[0][2] == 5


def test_power() -> None:
    p = sX(0) + sY(1) + qf.Pauli.term([2, 3], "XY")

    assert p ** 0 == qf.Pauli.identity()
    assert p ** 1 == p
    assert p * p == p ** 2
    assert p * p * p == p ** 3
    assert p * p * p * p * p * p * p * p * p * p == p ** 10

    with pytest.raises(ValueError):
        p ** -1

    p ** 400


def test_simplify() -> None:
    t1 = sZ(0) * sZ(1)
    t2 = sZ(0) * sZ(1)
    assert (t1 + t2) == 2 * sZ(0) * sZ(1)


def test_dont_simplify() -> None:
    t1 = sZ(0) * sZ(1)
    t2 = sZ(2) * sZ(3)
    assert (t1 + t2) != 2 * sZ(0) * sZ(1)


def test_zero_term() -> None:
    qubit_id = 0
    coefficient = 10
    ps = sI(qubit_id) + sX(qubit_id)
    assert coefficient * qf.Pauli.zero() == qf.Pauli.zero()
    assert qf.Pauli.zero() * coefficient == qf.Pauli.zero()
    assert qf.Pauli.zero() * qf.Pauli.identity() == qf.Pauli.zero()
    assert qf.Pauli.zero() + qf.Pauli.identity() == qf.Pauli.identity()
    assert qf.Pauli.zero() + ps == ps
    assert ps + qf.Pauli.zero() == ps


def test_neg() -> None:
    ps = sY(0) - sX(0)
    ps -= sZ(1)
    ps = ps - 3
    _ = 3 + ps


def test_paulisum_iteration() -> None:
    term_list = [sX(2), sZ(4)]
    ps = sX(2) + sZ(4)
    for ii, term in enumerate(ps):
        assert term_list[ii].terms[0] == term


def test_check_commutation() -> None:
    term1 = sX(0) * sX(1)
    term2 = sY(0) * sY(1)
    term3 = sY(0) * sZ(2)

    assert qf.paulis_commute(term2, term3)
    assert qf.paulis_commute(term2, term3)
    assert not qf.paulis_commute(term1, term3)


def test_commuting_sets() -> None:
    term1 = sX(0) * sX(1)
    term2 = sY(0) * sY(1)
    term3 = sY(0) * sZ(2)
    ps = term1 + term2 + term3
    pcs = qf.pauli_commuting_sets(ps)
    assert len(pcs) == 2

    pcs = qf.pauli_commuting_sets(term1)
    assert len(pcs) == 1


def test_get_qubits() -> None:
    term = sZ(0) * sX(1)
    assert term.qubits == (0, 1)
    term += sY(10) + sX(2) + sY(2)
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
    x = sX(1)
    y = sY(2, 2.0)
    y2 = sY(2, 2.00000001)

    assert qf.paulis_close(x, x)
    assert not qf.paulis_close(x, y)
    assert qf.paulis_close(y, y2)


def test_run() -> None:
    x = sX(1)
    y = sY(2, 1.2)
    s = x + y

    ket0 = qf.zero_state(3)
    _ = s.run(ket0)


def test_pauli_decompose_hermitian() -> None:
    gate = qf.X(0)
    H = gate.asoperator()
    pl = qf.pauli_decompose_hermitian(H)
    assert np.allclose(pl.asoperator(), H)

    gate = qf.X(0)
    op = gate.asoperator()
    H = -scipy.linalg.logm(op) / 1.0j
    pl = qf.pauli_decompose_hermitian(H)
    assert np.allclose(pl.asoperator(), H)

    N = 4
    gate2 = qf.RandomGate(range(N))
    op = gate2.asoperator()
    H = -scipy.linalg.logm(op) / 1.0j
    pl = qf.pauli_decompose_hermitian(H)
    assert np.allclose(pl.asoperator(), H)

    op = np.ones(shape=[2, 2, 2])
    with pytest.raises(ValueError):
        qf.pauli_decompose_hermitian(op)

    op = np.ones(shape=[4, 4])
    op[0, 1] = 10000
    with pytest.raises(ValueError):
        qf.pauli_decompose_hermitian(op)

    op = np.ones(shape=[3, 3])
    with pytest.raises(ValueError):
        qf.pauli_decompose_hermitian(op)


def test_pauli_rewire() -> None:
    term = sZ(0) * sX(1)
    term = term.on(5, 4)
    assert term.qubits == (4, 5)
    assert term == sZ(5) * sX(4)
    term = term.rewire({5: "a", 4: "b"})
    assert term.qubits == ("a", "b")
    assert term == sZ("a") * sX("b")

    with pytest.raises(ValueError):
        term.on(1, 2, 3, 4)


# fin
