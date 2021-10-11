# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np
import pytest

import quantumflow as qf


def test_asqutensor() -> None:

    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    vec1 = qf.asqutensor(arr)
    vec2 = qf.asqutensor(arr, ndim=1)
    assert np.allclose(vec1, vec2)

    op = qf.asqutensor(arr, ndim=2)
    assert op.ndim == 2

    superop = qf.asqutensor(arr, ndim=4)
    assert superop.ndim == 4

    with pytest.raises(ValueError):
        qf.asqutensor([0, 1, 2, 3, 4])


def test_sorted_bits() -> None:

    bits: qf.Qubits = ["z", "a", 3, 4, 5, (4, 2), (2, 3)]
    s = qf.sorted_qubits(bits)
    assert s == (3, 4, 5, "a", "z", (2, 3), (4, 2))

    bits = ["z", "a", 3, 4, 5, (4, 2), (2, 3)]
    s = qf.sorted_cbits(bits)
    assert s == (3, 4, 5, "a", "z", (2, 3), (4, 2))
