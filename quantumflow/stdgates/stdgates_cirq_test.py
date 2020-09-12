# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import random

import numpy as np

import quantumflow as qf

from ..config_test import REPS


def test_FSim() -> None:
    for _ in range(REPS):
        theta = random.uniform(-np.pi, +np.pi)
        phi = random.uniform(-np.pi, +np.pi)
        gate0 = qf.FSim(theta, phi, 0, 1)

        # Test with decomposition from Cirq.
        circ = qf.Circuit()
        circ += qf.XX(theta / np.pi, 0, 1)
        circ += qf.YY(theta / np.pi, 0, 1)
        circ += qf.CZ(0, 1) ** (-phi / np.pi)
        gate1 = circ.asgate()
        assert qf.gates_close(gate0, gate1)

        assert qf.gates_close(gate1.H, gate0.H)


def test_PhasedX() -> None:
    q0 = "4"
    for _ in range(REPS):
        t = random.uniform(0, 1)
        p = random.uniform(-2, +2)

        gate0 = qf.PhasedX(p, q0)
        gate1 = qf.PhasedXPow(p, t, q0)
        assert qf.gates_close(gate0 ** t, gate1)
        assert (gate0 ** t).qubits == (q0,)

        gate0.H
        gate1.H
        gate2 = gate1 ** t
        p2, t2 = gate2.params
        assert p2 == p
        assert np.isclose(t2 - t ** 2, 0.0)

        assert qf.gates_close(gate0, gate0.specialize())

    assert qf.gates_close(qf.PhasedX(-2.0, q0).specialize(), qf.X(q0))
