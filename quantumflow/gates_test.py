# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import numpy as np

import quantumflow as qf


def test_Identity() -> None:
    gate0 = qf.Identity([0, 1, 4])
    assert qf.almost_identity(gate0)
    assert gate0.cv_operator_structure == qf.OperatorStructure.identity
    assert gate0.H is gate0
    assert gate0 ** 4 is gate0

    arr = np.asarray(gate0.sym_operator).astype(np.complex128)
    assert np.allclose(gate0.operator, arr)


def test_unitary() -> None:
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
