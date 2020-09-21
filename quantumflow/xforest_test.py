# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for quantumflow.xforest"""


import numpy as np
import pytest

import quantumflow as qf  # noqa: 402
from quantumflow import xforest  # noqa: 402

pytest.importorskip("pyquil")  # noqa: 402


def test_circuit_to_pyquil() -> None:
    circ = qf.Circuit()
    circ += qf.X(0)

    prog = xforest.circuit_to_pyquil(circ)
    assert str(prog) == "X 0\n"

    circ = qf.Circuit()
    circ1 = qf.Circuit()
    circ2 = qf.Circuit()
    circ1 += qf.Ry(np.pi / 2, 0)
    circ1 += qf.Rz(np.pi, 0)
    circ1 += qf.Ry(np.pi / 2, 1)
    circ1 += qf.Rx(np.pi, 1)
    circ1 += qf.CNot(0, 1)
    circ2 += qf.Rx(-np.pi / 2, 1)
    circ2 += qf.Ry(4.71572463191, 1)
    circ2 += qf.Rx(np.pi / 2, 1)
    circ2 += qf.CNot(0, 1)
    circ2 += qf.Rx(-2 * 2.74973750579, 0)
    circ2 += qf.Rx(-2 * 2.74973750579, 1)
    circ += circ1
    circ += circ2

    prog = xforest.circuit_to_pyquil(circ)
    new_circ = xforest.pyquil_to_circuit(prog)

    assert qf.circuits_close(circ, new_circ)


# fin
