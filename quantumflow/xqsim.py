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


try:
    import qsimcirq
except ModuleNotFoundError as err:  # pragma: no cover
    raise ModuleNotFoundError(
        "External dependency 'qsimcirq' not installed. Install"
        "with 'pip install qsimcirq'"
    ) from err

from .circuits import Circuit
from .gatesets import QSIM_GATES
from .ops import Operation
from .qubits import Qubits
from .states import State
from .translate import circuit_translate
from .xcirq import circuit_to_cirq

__all__ = ["QSimSimulator", "translate_circuit_to_qsim", "QSIM_GATES"]


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
