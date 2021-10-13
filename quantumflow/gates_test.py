# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import numpy as np

import quantumflow as qf


def test_unitary() -> None:
    gate0 = qf.X(0)
    gate1 = qf.Unitary.from_gate(gate0)

    assert gate0.qubit_nb == gate1.qubit_nb
    assert gate0.qubits == gate1.qubits
    assert np.allclose(gate0.operator, gate1.operator)

    # FIXME: more tests
    _ = gate1.H
    _ = gate1 ** -1
    _ = gate1 ** -0.3
