# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random

import numpy as np

import quantumflow as qf

from ..config_test import REPS


def test_CPhase_gates() -> None:
    for _ in range(REPS):
        theta = random.uniform(-4 * np.pi, +4 * np.pi)

        gate11 = qf.ControlGate([0], qf.PhaseShift(theta, 1))
        assert qf.gates_close(gate11, qf.CPhase(theta, 0, 1))

        gate00 = qf.X(0) @ qf.IdentityGate([0, 1])
        gate00 = qf.X(1) @ gate00
        gate00 = gate11 @ gate00
        gate00 = qf.X(0) @ gate00
        gate00 = qf.X(1) @ gate00
        assert qf.gates_close(gate00, qf.CPhase00(theta))

        gate10 = qf.X(0) @ qf.IdentityGate([0, 1])
        gate10 = qf.X(1) @ gate10
        gate10 = qf.CPhase01(theta, 0, 1) @ gate10
        gate10 = qf.X(0) @ gate10
        gate10 = qf.X(1) @ gate10
        assert qf.gates_close(gate10, qf.CPhase10(theta))

        gate0 = qf.CPhase(theta) ** 2
        gate1 = qf.CPhase(theta * 2)
        assert qf.gates_close(gate0, gate1)


def test_CPhase_pow() -> None:
    gate0 = qf.CZ(0, 1) ** 0.4
    gate1 = qf.CPhase(0.4 * np.pi, 0, 1)
    assert qf.gates_close(gate0, gate1)


def test_pswap() -> None:
    assert qf.gates_close(qf.Swap(2, 4), qf.PSwap(0, 2, 4))
    assert qf.gates_close(qf.ISwap(3, 2), qf.PSwap(np.pi / 2, 3, 2))


# fin
