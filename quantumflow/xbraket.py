# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
.. contents:: :local:
.. currentmodule:: quantumflow.xqiskit

Interface between IBM's Qiskit and QuantumFlow

.. autoclass:: QiskitSimulator
.. autofunction:: qiskit_to_circuit
.. autofunction:: circuit_to_qiskit
"""

import numpy as np

try:
    from braket.circuits import Circuit as bkCircuit
    from braket.circuits.angled_gate import AngledGate as bkAngledGate
    from braket.devices import LocalSimulator as bkLocalSimulator
except ModuleNotFoundError as err:  # pragma: no cover
    raise ModuleNotFoundError(
        "External dependency 'braket' not installed. Install"
        "with 'pip install amazon-braket-sdk'"
    ) from err


from .circuits import Circuit
from .gatesets import BRAKET_GATES
from .ops import Operation, StdGate
from .qubits import Qubits
from .states import State
from .translate import circuit_translate
from .utils import invert_map

__all__ = [
    "BRAKET_GATES",
    "braket_to_circuit",
    "circuit_to_braket",
    "translate_to_braket",
]


BRAKET_TO_QF = {
    "CCNot": "CCNot",
    "CNot": "CNot",
    "CPhaseShift": "CPhase",
    "CPhaseShift00": "CPhase00",
    "CPhaseShift01": "CPhase01",
    "CPhaseShift10": "CPhase10",
    "CSwap": "CSwap",
    "CY": "CY",
    "CZ": "CZ",
    "H": "H",
    "I": "I",
    "ISwap": "ISwap",
    "PSwap": "PSwap",
    "PhaseShift": "PhaseShift",
    "Rx": "Rx",
    "Ry": "Ry",
    "Rz": "Rz",
    "S": "S",
    "Si": "S_H",
    "Swap": "Swap",
    "T": "T",
    "Ti": "T_H",
    "V": "V",
    "Vi": "V_H",
    "X": "X",
    "XX": "XX",  # Different parameterization
    "XY": "XY",  # Very different parameterization
    "Y": "Y",
    "YY": "YY",  # Different parameterization
    "Z": "Z",
    "ZZ": "ZZ",  # Different parameterization
}
"""Map from braket operation names to QuantumFlow names"""

# TODO: use as test
# BRAKET_GATES: Tuple[Type[Gate], ...] = tuple(
#     set([StdGate.cv_stdgates[n] for n in BRAKET_TO_QF.values()])
# )
# """Tuple of QuantumFlow gates that we can convert directly to braket"""

# TODO: Unitary


def braket_to_circuit(bkcircuit: bkCircuit) -> Circuit:
    """Convert a braket.Circuit to QuantumFlow's Circuit"""

    circ = Circuit()

    for inst in bkcircuit.instructions:
        op = inst.operator
        name = op.name
        qubits = [int(q) for q in inst.target]

        if name not in BRAKET_TO_QF:
            raise NotImplementedError("Unknown braket operation")

        qf_name = BRAKET_TO_QF[name]

        if isinstance(op, bkAngledGate):
            angle = op.angle
            if op.name in [
                "XX",
                "YY",
                "ZZ",
            ]:
                args = [angle / np.pi] + qubits  # Different parameterization
            elif name == "XY":
                args = [-0.5 * args[0] / np.pi] + qubits
            else:
                args = [angle] + qubits
        else:
            args = qubits

        gate = StdGate.cv_stdgates[qf_name](*args)

        circ += gate

    return circ


def circuit_to_braket(circ: Circuit, translate: bool = False) -> bkCircuit:
    """Convert a QuantumFlow's Circuit to a braket Circuit.

    Qubits are converted to contiguous integers if necessary.
    """

    # Braket regrettable follows the broken design of qiskit  where each gate
    # is defined as a class, and then for each class a method is monkey patched onto
    # Circuit which will create that gate and append it to the circuit.

    bkcircuit = bkCircuit()
    qbs = circ.qubits

    new_qbs = list(range(len(qbs)))
    circ = circ.on(*new_qbs)

    if translate:
        circ = translate_to_braket(circ)

    QF_TO_BRAKET = invert_map(BRAKET_TO_QF)

    for op in circ:
        name = QF_TO_BRAKET[op.name].lower()
        params = [float(p) for p in op.params]
        if name in ["xx", "yy", "zz"]:
            params = [params[0] * np.pi]
        elif name == "xy":
            params = [-params[0] * np.pi * 2]
        qbs = op.qubits
        getattr(bkcircuit, name)(*qbs, *params)

    return bkcircuit


def translate_to_braket(circ: Circuit) -> Circuit:
    """Convert QF gates to gates understood by qiskit"""
    return circuit_translate(circ, targets=BRAKET_GATES)


# FIXME: *elements? in __init__? Interface should be like circuit
class BraketSimulator(Operation):
    def __init__(self, *elements: Operation) -> None:
        self._circuit = Circuit(*elements)

    @property
    def qubits(self) -> Qubits:
        return self._circuit.qubits

    def run(self, ket: State = None) -> State:
        if ket is not None:
            raise NotImplementedError("Initial ket not supported")

        circ = self._circuit

        bkcircuit = circuit_to_braket(circ, translate=True)
        bkcircuit.state_vector()

        device = bkLocalSimulator()
        result = device.run(bkcircuit).result()
        tensor = result.values[0]

        res = State(tensor, self.qubits)

        return res


# fin
