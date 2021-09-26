# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import TextIO

import numpy as np

from . import tensors, utils
from .ops import Gate, Unitary
from .qubits import Qubit
from .tensors import QubitTensor

__all__ = (
    "join_gates",
    "P0",
    "P1",
)


# DOCME: Can also use circuit.asgate()
# Deprecate. No longer used anywhere?
def join_gates(gate0: Gate, gate1: Gate) -> Unitary:
    """Direct product of gates. Qubit count is the sum of each gate's
    bit count."""
    tensor = tensors.outer(gate0.tensor, gate1.tensor, rank=2)
    return Unitary(tensor, tuple(gate0.qubits) + tuple(gate1.qubits))


# FIXME: P0 and P1 should be channels or Projectors ???


class P0(Gate):
    r"""Project qubit to zero.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 0.
    """
    text_labels = ["|0><0|"]

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor([[1, 0], [0, 0]])


class P1(Gate):
    r"""Project qubit to one.

    A non-unitary gate that represents the effect of a measurement. The norm
    of the resultant state is multiplied by the probability of observing 1.
    """
    text_labels = ["|1><1|"]

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor([[0, 0], [0, 1]])


# fin
