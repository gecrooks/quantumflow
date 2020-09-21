# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unittests for QuantumFlow Gate Decompositions
"""

import numpy as np
import pytest
import scipy.stats
from numpy import pi

import quantumflow as qf
from quantumflow.decompositions import _eig_complex_symmetric

from .config_test import REPS


def test_bloch_decomposition() -> None:
    theta = 1.23

    gate0 = qf.Rn(theta, 1, 0, 0, 9)
    gate1 = qf.bloch_decomposition(gate0)[0]
    assert isinstance(gate1, qf.Gate)
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.Rn(theta, 0, 1, 0, 9)
    gate1 = qf.bloch_decomposition(gate0)[0]
    assert isinstance(gate1, qf.Gate)
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.Rn(theta, 0, 0, 1, 9)
    gate1 = qf.bloch_decomposition(gate0)[0]
    assert isinstance(gate1, qf.Gate)
    assert qf.gates_close(gate0, gate1)

    gate0 = qf.Rn(pi, np.sqrt(2), 0, np.sqrt(2), 9)
    gate1 = qf.bloch_decomposition(gate0)[0]
    assert isinstance(gate1, qf.Gate)
    assert qf.gates_close(gate0, gate1)

    for _ in range(REPS):
        gate3 = qf.RandomGate(qubits=[0])
        gate4 = qf.bloch_decomposition(gate3)[0]
        assert isinstance(gate4, qf.Gate)
        assert qf.gates_close(gate3, gate4)

    gate5 = qf.I(0)
    gate6 = qf.bloch_decomposition(gate5)[0]
    assert isinstance(gate6, qf.Gate)
    assert qf.gates_close(gate5, gate6)


def test_bloch_decomp_errors() -> None:
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.bloch_decomposition(qf.CNot(0, 1))


def test_zyz_decomp_errors() -> None:
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.zyz_decomposition(qf.CNot(0, 1))


def test_zyz_decomposition() -> None:
    gate0 = qf.RandomGate([1])
    circ1 = qf.zyz_decomposition(gate0)
    gate1 = circ1.asgate()
    assert qf.gates_close(gate0, gate1)


def test_euler_decomposition() -> None:
    gate0 = qf.RandomGate([1])

    for order in ["XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"]:
        circ1 = qf.euler_decomposition(gate0, euler=order)
        gate1 = circ1.asgate()
        assert qf.gates_close(gate0, gate1)


def test_kronecker_decomposition() -> None:
    for _ in range(REPS):
        left = qf.RandomGate([1]).asoperator()
        right = qf.RandomGate([1]).asoperator()
        both = np.kron(left, right)
        gate0 = qf.Unitary(both, [0, 1])
        circ = qf.kronecker_decomposition(gate0)
        gate1 = circ.asgate()

        assert qf.gates_close(gate0, gate1)

    circ2 = qf.Circuit()
    circ2 += qf.Z(0)
    circ2 += qf.H(1)
    gate2 = circ2.asgate()
    circ3 = qf.kronecker_decomposition(gate2)
    gate3 = circ3.asgate()
    assert qf.gates_close(gate2, gate3)

    circ4 = qf.kronecker_decomposition(gate0, euler="XYX")
    gate4 = circ4.asgate()
    assert qf.gates_close(gate0, gate4)


def test_kronecker_decomp_errors() -> None:
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.kronecker_decomposition(qf.X(0))

    # Not kronecker product
    with pytest.raises(ValueError):
        qf.kronecker_decomposition(qf.CNot(0, 1))


def test_canonical_decomposition() -> None:
    for tt1 in range(0, 6):
        for tt2 in range(tt1):
            for tt3 in range(tt2):
                t1, t2, t3 = tt1 / 12, tt2 / 12, tt3 / 12
                if t3 == 0 and t1 > 0.5:
                    continue
                coords = np.asarray((t1, t2, t3))

                circ0 = qf.Circuit()
                circ0 += qf.RandomGate([0])
                circ0 += qf.RandomGate([1])
                circ0 += qf.Can(t1, t2, t3, 0, 1)
                circ0 += qf.RandomGate([0])
                circ0 += qf.RandomGate([1])
                gate0 = circ0.asgate()

                circ1 = qf.canonical_decomposition(gate0)
                assert qf.gates_close(gate0, circ1.asgate())

                canon = circ1[1]
                new_coords = np.asarray([canon.param(n) for n in ["tx", "ty", "tz"]])
                assert np.allclose(coords, np.asarray(new_coords))

                coords2 = qf.canonical_coords(gate0)
                assert np.allclose(coords, np.asarray(coords2))


def test_canonical_decomp_sandwich() -> None:
    for _ in range(REPS):
        # Random CZ sandwich
        circ0 = qf.Circuit()
        circ0 += qf.RandomGate([0])
        circ0 += qf.RandomGate([1])
        circ0 += qf.CZ(0, 1)
        circ0 += qf.YPow(0.4, 0)
        circ0 += qf.YPow(0.25, 1)
        circ0 += qf.CZ(0, 1)
        circ0 += qf.RandomGate([0])
        circ0 += qf.RandomGate([1])

        gate0 = circ0.asgate()

        circ1 = qf.canonical_decomposition(gate0)
        gate1 = circ1.asgate()

        assert qf.gates_close(gate0, gate1)
        assert qf.almost_unitary(gate0)


def test_canonical_decomp_random() -> None:
    for _ in range(REPS * 2):
        gate0 = qf.RandomGate([0, 1])
        gate1 = qf.canonical_decomposition(gate0).asgate()
        assert qf.gates_close(gate0, gate1)


def test_canonical_decomp_errors() -> None:
    # Wrong number of qubits
    with pytest.raises(ValueError):
        qf.canonical_decomposition(qf.X(0))


def test_decomp_stdgates() -> None:
    gate0 = qf.IdentityGate([0, 1])
    gate1 = qf.canonical_decomposition(gate0).asgate()
    assert qf.gates_close(gate0, gate1)

    gate2 = qf.CNot(0, 1)
    gate3 = qf.canonical_decomposition(gate2).asgate()
    assert qf.gates_close(gate2, gate3)

    gate4 = qf.Swap(0, 1)
    gate5 = qf.canonical_decomposition(gate4).asgate()
    assert qf.gates_close(gate4, gate5)

    gate6 = qf.ISwap(0, 1)
    gate7 = qf.canonical_decomposition(gate6).asgate()
    assert qf.gates_close(gate6, gate7)

    gate8 = qf.CNot(0, 1) ** 0.5
    gate9 = qf.canonical_decomposition(gate8).asgate()
    assert qf.gates_close(gate8, gate9)

    gate10 = qf.Swap(0, 1) ** 0.5
    gate11 = qf.canonical_decomposition(gate10).asgate()
    assert qf.gates_close(gate10, gate11)

    gate12 = qf.ISwap(0, 1) ** 0.5
    gate13 = qf.canonical_decomposition(gate12).asgate()
    assert qf.gates_close(gate12, gate13)


def test_decomp_sqrtswap_sandwich() -> None:
    circ0 = qf.Circuit()
    circ0 += qf.Can(1 / 4, 1 / 4, 1 / 4, 0, 1)
    circ0 += qf.RandomGate([0])
    circ0 += qf.RandomGate([1])
    circ0 += qf.Can(1 / 4, 1 / 4, 1 / 4, 0, 1)

    gate0 = circ0.asgate()
    circ1 = qf.canonical_decomposition(gate0)
    gate1 = circ1.asgate()
    assert qf.gates_close(gate0, gate1)


def test_eig_complex_symmetric() -> None:
    samples = 100
    for _ in range(samples):

        # Build a random symmetric complex matrix
        orthoganal = scipy.stats.special_ortho_group.rvs(4)
        eigvals = (
            np.random.normal(size=(4,)) + 1j * np.random.normal(size=(4,))
        ) / np.sqrt(2.0)
        M = orthoganal @ np.diag(eigvals) @ orthoganal.T

        eigvals, eigvecs = _eig_complex_symmetric(M)
        assert np.allclose(M, eigvecs @ np.diag(eigvals) @ eigvecs.T)


def test_eigcs_errors() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        _eig_complex_symmetric(np.random.normal(size=(4, 4)))


def test_b_decomposition() -> None:
    for _ in range(REPS):
        gate0 = qf.RandomGate([4, 8])
        circ = qf.b_decomposition(gate0)
        gate1 = circ.asgate()
        assert qf.gates_close(gate0, gate1)


def test_cnot_decomposition() -> None:
    for _ in range(REPS):
        gate0 = qf.RandomGate([4, 8])
        circ = qf.cnot_decomposition(gate0)
        gate1 = circ.asgate()
        assert qf.gates_close(gate0, gate1)


def test_convert_can_to_weyl() -> None:
    for r in range(100):
        tx = np.random.uniform(-10, +10)
        ty = np.random.uniform(-10, +10)
        tz = np.random.uniform(-10, +10)

        gate0 = qf.Can(tx, ty, tz, 1, 2)
        circ = qf.convert_can_to_weyl(gate0, euler="XYX")

        assert qf.gates_close(gate0, circ.asgate())


# Fin
