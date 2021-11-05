# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random

import numpy as np

import quantumflow as qf

from ..config_test import REPS


def test_XPow() -> None:
    assert qf.gates_close(qf.V(0), qf.XPow(0.5, 0))
    assert qf.gates_close(qf.V_H(0), qf.XPow(-0.5, 0))
    assert qf.gates_close(qf.V(1).H, qf.V_H(1))
    assert qf.gates_close(qf.V(1).H.H, qf.V(1))
    assert qf.gates_close(qf.V(0) ** 0.5, qf.XPow(0.25, 0))
    assert qf.gates_close(qf.V_H(0) ** 0.5, qf.XPow(-0.25, 0))
    assert qf.gates_close(qf.V(0) @ qf.V(0), qf.X(0))


def test_rn() -> None:
    theta = 1.23

    gate = qf.Rn(theta, 1, 0, 0, "q0")
    assert qf.gates_close(gate, qf.Rx(theta, "q0"))

    gate = qf.Rn(theta, 0, 1, 0, "q0")
    assert qf.gates_close(gate, qf.Ry(theta, "q0"))

    gate = qf.Rn(theta, 0, 0, 1, "q0")
    assert qf.gates_close(gate, qf.Rz(theta, "q0"))

    gate = qf.Rn(np.pi, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), "q0")
    assert qf.gates_close(gate, qf.H("q0"))


def test_parametric_Z() -> None:
    assert qf.gates_close(qf.ZPow(0.25, 1), qf.T(1))
    assert qf.gates_close(qf.ZPow(0.5, 2), qf.S(2))
    assert qf.gates_close(qf.ZPow(1.0, 3), qf.Z(3))


def test_hadamard() -> None:
    gate = qf.Rz(np.pi / 2, 0) @ qf.I(0)
    gate = qf.Rx(np.pi / 2, 0) @ gate
    gate = qf.Rz(np.pi / 2, 0) @ gate

    assert qf.gates_close(gate, qf.H(0))


def test_rotation_gates() -> None:
    assert qf.gates_close(qf.I(0), qf.I(0))
    assert qf.gates_close(qf.Rx(np.pi, 0), qf.X(0))
    assert qf.gates_close(qf.Ry(np.pi, 0), qf.Y(0))
    assert qf.gates_close(qf.Rz(np.pi, 0), qf.Z(0))


def test_phaseshift() -> None:
    gate = qf.T(0) @ qf.T(0)

    assert qf.gates_close(gate, gate)

    assert qf.gates_close(gate, qf.S(0))
    assert qf.gates_close(qf.S(0), qf.PhaseShift(np.pi / 2, 0))
    assert qf.gates_close(qf.T(0), qf.PhaseShift(np.pi / 4, 0))

    # PhaseShift and RZ are the same up to a global phase.
    for _ in range(REPS):
        theta = random.uniform(-4 * np.pi, +4 * np.pi)
        assert qf.gates_close(qf.Rz(theta, 0), qf.PhaseShift(theta, 0))

    # Causes a rounding error that can result in NANs if not corrected for.
    theta = -2.5700302313621375
    assert qf.gates_close(qf.Rz(theta, 0), qf.PhaseShift(theta, 0))


# fin
