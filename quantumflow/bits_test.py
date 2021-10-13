# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import quantumflow as qf


def test_sorted_bits() -> None:

    qubits: qf.Qubits = ["z", "a", 3, 4, 5, (4, 2), (2, 3)]
    qbs = qf.sorted_bits(qubits)
    assert qbs == (3, 4, 5, "a", "z", (2, 3), (4, 2))

    cbits: qf.Cbits = ["z", "a", 3, 4, 5, (4, 2), (2, 3)]
    bs = qf.sorted_bits(cbits)
    assert bs == (3, 4, 5, "a", "z", (2, 3), (4, 2))
