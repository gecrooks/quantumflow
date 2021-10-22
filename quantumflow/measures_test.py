# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import quantumflow as qf
from quantumflow.config import ATOL


def test_almost_identity() -> None:
    assert qf.almost_identity(qf.I(1))
    assert qf.almost_identity(qf.Identity([0, 1, 2]))
    assert qf.almost_identity(qf.Ph(0.3, "q0"))  # Phase irrelevant

    assert not qf.almost_identity(qf.Rx(0.1, 0))
    assert qf.almost_identity(qf.Rx(ATOL, 0))
