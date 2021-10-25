# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import quantumflow as qf


def test_CX() -> None:
    gate0 = qf.CX(1, 4)

    assert gate0.control_qubits == (1,)
    assert gate0.control_qubit_nb == 1
    assert isinstance(gate0.target, qf.X)
    assert gate0.target.qubits == (4,)
    assert gate0.target.qubit_nb == 1

    # FIXME: More tests
