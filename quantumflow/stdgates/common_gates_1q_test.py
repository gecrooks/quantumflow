# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import numpy as np
import pytest

import quantumflow as qf


def test_Rx() -> None:
    assert qf.Rx.name in qf.OPERATIONS
    assert qf.Rx.name in qf.GATES
    assert qf.Rx.name in qf.STDGATES
    assert qf.Rx.name not in qf.STDCTRLGATES

    gate0 = qf.Rx(0.2, 1)

    assert gate0.name == "Rx"
    assert gate0.qubits == (1,)
    assert gate0.qubit_nb == 1
    assert gate0.addrs == ()
    assert gate0.addrs_nb == 0
    assert gate0.theta == 0.2

    with pytest.raises(AttributeError):
        gate0.t

    assert gate0.qubit_nb == gate0.cv_qubit_nb
    assert gate0.args == (0.2,)
    assert gate0.cv_operator_structure == qf.OperatorStructure.unstructured
    assert gate0.asgate() is gate0

    assert qf.gates_close(gate0, qf.Unitary.from_gate(gate0))
    assert qf.gates_close(gate0 ** -0.2, qf.Unitary.from_gate(gate0) ** -0.2)


def test_SqrtY() -> None:
    assert qf.gates_close(qf.SqrtY(0), qf.Y(0) ** 0.5)
    assert qf.gates_close(qf.SqrtY_H(0), qf.Y(0) ** -0.5)

    assert qf.gates_close(qf.SqrtY(0), qf.YPow(0.5, 0))
    assert qf.gates_close(qf.SqrtY_H(0), qf.YPow(-0.5, 0))

    assert np.allclose(qf.SqrtY(0).operator, qf.YPow(0.5, 0).operator)
    assert np.allclose(qf.SqrtY_H(0).operator, qf.YPow(-0.5, 0).operator)
