# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import numpy as np
import pytest

import quantumflow as qf


def test_CompositeGate() -> None:
    circ0 = qf.Circuit(qf.H(0), qf.CNot(0, 1), qf.CNot(1, 2))
    gate0 = qf.CompositeGate(*circ0)

    assert qf.gates_close(circ0.asgate(), gate0)

    assert circ0[0] in gate0

    # FIXME
    # assert qf.channels_close(circ0.aschannel(), gate0.aschannel())
    # assert qf.states_close(circ0.run(), gate0.run())
    # assert qf.densities_close(circ0.evolve(), gate0.evolve())

    gate1 = qf.CompositeGate(*circ0, qubits=[2, 3, 5, 4, 0, 1])
    assert gate1.qubits == (2, 3, 5, 4, 0, 1)
    assert gate1.H.qubits == (2, 3, 5, 4, 0, 1)

    assert qf.almost_identity(gate1 @ gate1.H)

    with pytest.raises(ValueError):
        qf.CompositeGate(qf.Circuit(qf.X(0)))  # type: ignore

    assert qf.gates_close(gate0 ** 0.4, qf.Circuit(*gate0).asgate() ** 0.4)

    # diag = qf.circuit_to_diagram(qf.Circuit(gate0))  # FIXME
    # print(diag)

    # gate2 = qf.ControlGate(gate0, [4, 5, 6])
    # diag = qf.circuit_to_diagram(qf.Circuit(gate2))
    # print(diag)

    # print()
    # s = str(gate0)
    # assert len(s.split("\n")) == len(gate0.circuit) + 1
    # print(s)

    gate3 = gate0.on([4, 3, 2])
    assert gate3.qubits == (4, 3, 2)
    assert qf.gates_close(gate3, qf.CompositeGate(*circ0.on([4, 3, 2])))

    gate4 = gate0.relabel({0: 3, 1: 5, 2: 4})
    assert gate4.qubits == (3, 5, 4)

    # circ5 = qf.Circuit(qf.Rx(0.1, 0), qf.Ry(0.2, 0), qf.Rz(0.2, 2))
    # gate5 = qf.CompositeGate(*circ5)
    # print(gate5.params)

    # with pytest.raises(ValueError):
    #     gate0.param("theta")


def test_Identity() -> None:
    gate0 = qf.Identity([0, 1, 4])
    assert qf.almost_identity(gate0)
    assert gate0.cv_operator_structure == qf.OperatorStructure.identity
    assert gate0.H is gate0
    assert gate0 ** 4 is gate0

    arr = np.asarray(gate0.sym_operator).astype(np.complex128)
    assert np.allclose(gate0.operator, arr)


def test_Unitary() -> None:
    gate0 = qf.X(0)
    gate1 = qf.Unitary.from_gate(gate0)

    assert gate0.qubit_nb == gate1.qubit_nb
    assert gate0.qubits == gate1.qubits
    assert np.allclose(gate0.operator, gate1.operator)

    arr = np.asarray(gate1.sym_operator).astype(np.complex128)
    assert np.allclose(gate1.operator, arr)

    # FIXME: more tests
    _ = gate1.H
    _ = gate1 ** -1
    _ = gate1 ** -0.3
