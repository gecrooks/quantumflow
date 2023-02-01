# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.xbraket
"""

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.xbraket import (
    BraketSimulator,
    braket_to_circuit,
    circuit_to_braket,
)

pytest.importorskip("braket")


def test_braket_to_circuit() -> None:
    from braket.circuits import Circuit as bkCircuit

    bkcirc = bkCircuit().h(0).cnot(0, 1)

    circ = braket_to_circuit(bkcirc)
    print(circ)

    bkcirc = bkcirc.rx(1, 0.2)
    bkcirc = bkcirc.xx(0, 1, np.pi * 0.5)
    bkcirc = bkcirc.xy(0, 2, np.pi * 0.5)
    circ = braket_to_circuit(bkcirc)
    print(circ)


def test_circuit_to_qiskit() -> None:
    circ = qf.Circuit([qf.CNot(0, 1), qf.Rz(0.2, 1)])
    bkcirc = circuit_to_braket(circ)
    print(bkcirc)


def test_braketsimulator() -> None:
    circ = qf.Circuit()
    circ += qf.Rx(0.4, 0)
    circ += qf.X(0)
    circ += qf.H(1)
    circ += qf.Y(2)
    circ += qf.Rx(0.3, 0)
    circ += qf.XX(0.2, 0, 1)
    circ += qf.XY(0.3, 0, 1)
    circ += qf.ZZ(0.4, 0, 1)

    circ += qf.Can(0.1, 0.2, 0.2, 0, 1)
    circ += qf.V(0)
    circ += qf.CV(2, 3)
    circ += qf.CPhase01(2, 3)

    sim = BraketSimulator(circ)
    assert qf.states_close(circ.run(), sim.run())


# fin
