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


@pytest.mark.parametrize("angle", [0.0, 0.37, -1.1, np.pi, 4.0])
@pytest.mark.parametrize("targets", [(0, 1), (1, 0)])
@pytest.mark.parametrize("prefix", ["none", "rotation", "cnot", "xy"])
def test_braket_xy_preserves_operator_and_parameters(angle, targets, prefix) -> None:
    from braket.circuits import Circuit as bkCircuit

    bkcirc = bkCircuit()
    if prefix == "rotation":
        bkcirc.rx(0, 0.83)
    elif prefix == "cnot":
        bkcirc.cnot(1, 0)
    elif prefix == "xy":
        bkcirc.xy(1, 0, -0.61)
    bkcirc.xy(*targets, angle)

    circ = braket_to_circuit(bkcirc)
    xy = circ[-1]
    assert isinstance(xy, qf.XY)
    assert xy.qubits == targets
    assert xy.params == (-angle / (2 * np.pi),)
    # Both source and converted matrices order active wires ascending, with
    # the first wire as the most significant tensor axis. Compare full complex
    # operators, including phase, not just one input state's probabilities.
    assert circ.qubits == (0, 1)
    np.testing.assert_allclose(
        circ.asgate().asoperator(), bkcirc.to_unitary(), atol=1e-12, rtol=1e-12
    )
    exported = circuit_to_braket(circ)
    assert tuple(exported.instructions[-1].target) == targets
    assert float(exported.instructions[-1].operator.angle) == pytest.approx(angle)
    np.testing.assert_allclose(
        exported.to_unitary(), bkcirc.to_unitary(), atol=1e-12, rtol=1e-12
    )


def test_braket_xy_asymmetric_surrounding_operations() -> None:
    from braket.circuits import Circuit as bkCircuit

    bkcirc = (bkCircuit().h(0).ry(1, 0.37).t(0).cnot(1, 0)
              .xy(1, 0, -1.1).rz(0, 0.83).xy(0, 1, 0.29))
    circ = braket_to_circuit(bkcirc)
    np.testing.assert_allclose(
        circ.asgate().asoperator(), bkcirc.to_unitary(), atol=1e-12, rtol=1e-12
    )


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
