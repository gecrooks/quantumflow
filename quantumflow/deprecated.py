# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


""" Quantumflow Deprecated

A module for obsolete functions and classes, where they can live out
their pitiful existence of quite desperation.
"""

from .gates import P0, P1, join_gates
from .ops import Gate, UnitaryGate
from .qubits import Qubit
from .utils import deprecated

__all__ = ("conditional_gate",)


# Replaced by ConditionalGate
@deprecated
def conditional_gate(control: Qubit, gate0: Gate, gate1: Gate) -> UnitaryGate:
    """Return a conditional unitary gate. Do gate0 on bit 1 if bit 0 is zero,
    else do gate1 on 1

    Deprecated: Use the ConditionalGate class instead.
    """
    assert gate0.qubits == gate1.qubits  # FIXME

    tensor = join_gates(P0(control), gate0).tensor
    tensor += join_gates(P1(control), gate1).tensor
    gate = UnitaryGate(tensor, (control,) + tuple(gate0.qubits))
    return gate


# fin
