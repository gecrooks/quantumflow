# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import numpy as np

import quantumflow as qf


def test_ccnot() -> None:
    ket = qf.zero_state(3)
    ket = qf.CCNot(0, 1, 2).run(ket)
    assert ket.tensor[0, 0, 0] == 1.0

    ket = qf.X(1).run(ket)
    ket = qf.CCNot(0, 1, 2).run(ket)
    assert ket.tensor[0, 1, 0] == 1.0

    ket = qf.X(0).run(ket)
    ket = qf.CCNot(0, 1, 2).run(ket)
    assert ket.tensor[1, 1, 1] == 1.0


def test_cswap() -> None:
    ket = qf.zero_state(3)
    ket = qf.X(1).run(ket)
    ket = qf.CSwap(0, 1, 2).run(ket)
    assert ket.tensor[0, 1, 0] == 1.0

    ket = qf.X(0).run(ket)
    ket = qf.CSwap(0, 1, 2).run(ket)
    assert ket.tensor[1, 0, 1] == 1.0


def test_ccz() -> None:
    ket = qf.zero_state(3)
    ket = qf.X(0).run(ket)
    ket = qf.X(1).run(ket)
    ket = qf.H(2).run(ket)
    ket = qf.CCZ(0, 1, 2).run(ket)
    ket = qf.H(2).run(ket)
    qf.print_state(ket)
    assert np.isclose(ket.tensor[1, 1, 1], 1.0)

    gate0 = qf.CCZ()
    assert gate0.H is gate0


def test_deutsch() -> None:
    gate0 = qf.Deutsch(5 * np.pi / 2, 0, 1, 2)
    gate1 = qf.CCNot(0, 1, 2)
    assert qf.gates_close(gate0, gate1)


# fin
