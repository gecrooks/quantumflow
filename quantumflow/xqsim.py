# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumflow.xquirk

Interface to qsim

https://github.com/quantumlib/qsim

"""

from typing import Optional

from .circuits import Circuit
from .gatesets import QSIM_GATES
from .states import State
from .stdops import Simulator
from .translate import circuit_translate
from .xcirq import circuit_to_cirq

__all__ = ["QSimSimulator", "translate_circuit_to_qsim", "QSIM_GATES"]


_IMPORT_ERROR_MSG = """External dependency 'qsimcirq' not installed. Install
with 'pip install qsimcirq'"""


class QSimSimulator(Simulator):
    """Interface to the qsim quantum simulator. Adapts a QF Circuit (or
    other sequence of Operations).

    Ref:
        https://github.com/quantumlib/qsim
    """

    def __init__(self, circ: Circuit, translate: bool = True) -> None:
        try:
            import qsimcirq
        except ModuleNotFoundError as err:  # pragma: no cover
            raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

        circ = translate_circuit_to_qsim(circ)
        super().__init__(circ)
        self._cirq = circuit_to_cirq(self.circuit)
        self._qsim_circuit = qsimcirq.QSimCircuit(self._cirq)

    def run(self, ket: Optional[State] = None) -> State:
        try:
            import qsimcirq
        except ModuleNotFoundError as err:  # pragma: no cover
            raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

        if ket is not None:
            raise NotImplementedError("Not yet supported")

        sim = qsimcirq.QSimSimulator()
        res = sim.simulate(self._qsim_circuit)
        tensor = res.state_vector()
        return State(tensor, self.qubits)


def translate_circuit_to_qsim(circ: Circuit) -> Circuit:
    """Convert QF gates to gates supported by qsim"""
    circ = circuit_translate(circ, targets=QSIM_GATES)
    return circ
