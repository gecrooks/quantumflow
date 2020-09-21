# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random

import numpy as np

import quantumflow as qf

from ..config_test import REPS


def test_U3() -> None:
    theta = 0.2
    phi = 2.3
    lam = 1.1

    gate3 = qf.Circuit(
        [qf.U3(theta, phi, lam, 0), qf.U3(theta, phi, lam, 0).H]
    ).asgate()
    assert qf.almost_identity(gate3)

    gate2 = qf.Circuit([qf.U2(phi, lam, 0), qf.U2(phi, lam, 0).H]).asgate()
    assert qf.almost_identity(gate2)


def test_cu1() -> None:
    # Test that QASM's cu1 gate is the same as CPHASE up to global phase

    for _ in range(REPS):
        theta = random.uniform(0, 4)
        circ0 = qf.Circuit(
            [
                qf.Rz(theta / 2, 0),
                qf.CNot(0, 1),
                qf.Rz(-theta / 2, 1),
                qf.CNot(0, 1),
                qf.Rz(theta / 2, 1),
            ]
        )
        gate0 = circ0.asgate()
        gate1 = qf.CPhase(theta, 0, 1)
        assert qf.gates_close(gate0, gate1)


def test_CU3() -> None:
    theta = 0.2
    phi = 2.3
    lam = 1.1

    gate = qf.Circuit(
        [qf.CU3(theta, phi, lam, 0, 1), qf.CU3(theta, phi, lam, 0, 1).H]
    ).asgate()
    assert qf.almost_identity(gate)

    cgate = qf.ControlGate([0], qf.U3(theta, phi, lam, 1))
    assert qf.gates_close(qf.CU3(theta, phi, lam, 0, 1), cgate)


def test_CRZ() -> None:
    theta = 0.23
    gate0 = qf.CRZ(theta, 0, 1)
    coords = qf.canonical_coords(gate0)
    assert np.isclose(coords[0], 0.5 * theta / np.pi)
    assert np.isclose(coords[1], 0.0)
    assert np.isclose(coords[2], 0.0)

    coords = qf.canonical_coords(gate0 ** 3.3)
    assert np.isclose(coords[0], 3.3 * 0.5 * theta / np.pi)

    gate1 = qf.Circuit([qf.CRZ(theta, 0, 1), qf.CRZ(theta, 0, 1).H]).asgate()
    assert qf.almost_identity(gate1)


def test_RZZ() -> None:
    theta = 0.23
    gate0 = qf.RZZ(theta, 0, 1)
    gate1 = qf.ZZ(theta / np.pi, 0, 1)
    assert qf.gates_close(gate0, gate1)
    assert qf.gates_close(gate0.H, gate1.H)
    assert qf.gates_close(gate0 ** 0.12, gate1 ** 0.12)


# fin
