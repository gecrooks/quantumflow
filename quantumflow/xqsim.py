# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# DOCME

from typing import cast

import cirq
import qsimcirq

from .circuits import Circuit
from .ops import Operation
from .qubits import Qubits
from .states import State, zero_state
from .stdgates import (
    CZ,
    CNot,
    FSim,
    H,
    I,
    ISwap,
    S,
    T,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
)
from .translate import circuit_translate, select_translators
from .xcirq import circuit_to_cirq

# Note missing Swap, I, IDEN
QSIM_GATES = (I, X, Y, Z, S, T, H, XPow, YPow, ZPow, CZ, ISwap, CNot, FSim)


class QSimSimulator(Operation):
    """Interface to the qsim quantum simulator. Adapts a QF Circuit (or
    other sequence of Operations).

    Ref:
        https://github.com/quantumlib/qsim
    """

    def __init__(self, *elements: Operation, translate: bool = False) -> None:
        circ = Circuit(*elements)
        if translate:
            circ = translate_circuit_to_qsim(circ)
        self._circuit = circ
        self._cirq = circuit_to_cirq(self._circuit)
        self._qsim_circuit = qsimcirq.QSimCircuit(self._cirq)

    @property
    def qubits(self) -> Qubits:
        return self._circuit.qubits

    def run(self, ket: State = None) -> State:
        if ket is None:
            qubits = self.qubits
            ket = zero_state(qubits=qubits)
        else:
            raise NotImplementedError("Not yet implemented in qsim")

        sim = qsimcirq.QSimSimulator()
        res = sim.simulate(self._qsim_circuit)
        res = cast(cirq.WaveFunctionTrialResult, res)
        tensor = res.state_vector()
        return State(tensor, ket.qubits, ket.memory)


def translate_circuit_to_qsim(circ: Circuit) -> Circuit:
    """Convert QF gates to gates supported by qsim"""
    trans = select_translators(QSIM_GATES)
    circ = circuit_translate(circ, trans)
    return circ
