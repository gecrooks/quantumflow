# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import quantumflow as qf


def test_compile() -> None:
    circ0 = qf.addition_circuit([0], [1], [2, 3])
    circ1 = qf.compile_circuit(circ0)
    assert qf.circuits_close(circ0, circ1)
    assert circ1.size() == 76

    dagc = qf.DAGCircuit(circ1)
    assert dagc.depth(local=False) == 16
    counts = qf.count_operations(dagc)
    assert counts[qf.ZPow] == 27
    assert counts[qf.XPow] == 32
    assert counts[qf.CZ] == 17


def test_merge() -> None:
    circ0 = qf.Circuit(
        [
            qf.XPow(0.4, 0),
            qf.XPow(0.2, 0),
            qf.YPow(0.1, 1),
            qf.YPow(0.1, 1),
            qf.ZPow(0.1, 1),
            qf.ZPow(0.1, 1),
        ]
    )
    dagc = qf.DAGCircuit(circ0)

    qf.merge_tx(dagc)
    qf.merge_tz(dagc)
    qf.merge_ty(dagc)
    circ1 = qf.Circuit(dagc)
    assert len(circ1) == 3


# fin
