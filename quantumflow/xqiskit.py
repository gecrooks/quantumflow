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

# Note: Installation of qiskit is optional, so we defer import

# Note that QASM specific gates are defined in quantumflow/stdgates/stdgates_qasm.py
# since you might want to use those gates in QuantumFlow without loading
# qiskit

from typing import TYPE_CHECKING, Optional

from .circuits import Circuit
from .gatesets import QISKIT_GATES
from .states import State
from .stdgates import STDGATES
from .stdops import If, Initialize, Simulator
from .translate import circuit_translate
from .utils import invert_map

if TYPE_CHECKING:
    import qiskit  # pragma: no cover


__all__ = [
    "QISKIT_GATES",
    "QiskitSimulator",
    "qiskit_to_circuit",
    "circuit_to_qiskit",
    "translate_to_qiskit",
    "circuit_to_qasm",
    "qasm_to_circuit",
    "translate_gates_to_qiskit",  # Deprecated
]

_IMPORT_ERROR_MSG = """External dependency 'qiskit' not installed. Install
with 'pip install qiskit'"""


QASM_TO_QF = {
    "ccx": "CCNot",
    "ch": "CH",
    "crz": "CRz",
    "cswap": "CSwap",
    "cu1": "CPhase",  # Legacy. Changed to 'cp'
    "cp": "CPhase",
    "cu3": "CU3",
    "csx": "CV",
    "cx": "CNot",
    "cy": "CY",
    "cz": "CZ",
    "h": "H",
    "id": "I",
    "rx": "Rx",
    "ry": "Ry",
    "rz": "Rz",
    "rzz": "Rzz",
    "s": "S",
    "sdg": "S_H",
    "swap": "Swap",
    "sx": "V",
    # "sxdg": "V_H",   # Not fully supported by Aer simulator
    "t": "T",
    "tdg": "T_H",
    "u1": "PhaseShift",  # Legacy. Changed to 'p'
    "p": "PhaseShift",
    "u2": "U2",
    "u3": "U3",  # Legacy. Changed to 'u'
    "u": "U3",
    "x": "X",
    "y": "Y",
    "z": "Z",
    # 'barrier': 'Barrier',   # TODO TESTME
    # 'measure': 'Measure'   # TODO
    #  'initialize': 'Initialize',
}
"""Map from qiskit operation names to QuantumFlow names"""


# TODO: 'multiplexer', 'snapshot', 'unitary'


class QiskitSimulator(Simulator):
    def run(self, ket: Optional[State] = None) -> State:
        circ = self.circuit
        if ket is not None:
            circ = Circuit(Initialize(ket)) + circ

        qkcircuit = circuit_to_qiskit(circ, translate=True)

        # The call to get_backend() automagically adds the save_statevector() method to
        # QuantumCircuit. WTF. Mutating classes!? This is a terrible design.

        from qiskit_aer import Aer

        simulator = Aer.get_backend("aer_simulator")

        qkcircuit.save_statevector()
        result = simulator.run(qkcircuit).result()
        tensor = result.get_statevector()

        res = State(tensor, list(reversed(self.qubits)))

        if ket is not None:
            res = res.permute(ket.qubits)
        return res


def qiskit_to_circuit(qkcircuit: "qiskit.QuantumCircuit") -> Circuit:
    """Convert a qsikit QuantumCircuit to QuantumFlow's Circuit"""
    # We assume that there is only one quantum register of qubits.

    circ = Circuit()

    qkqbs = qkcircuit.qregs[0][:]

    for instruction, qargs, cargs in qkcircuit:
        name = instruction.name
        if name not in QASM_TO_QF:
            raise NotImplementedError("Unknown qiskit operation")

        qf_name = QASM_TO_QF[name]
        qubits = [qkqbs.index(q) for q in qargs]

        args = [float(param) for param in instruction.params] + qubits
        gate = STDGATES[qf_name](*args)  # type: ignore

        if instruction.condition is None:
            circ += gate
        else:
            classical, value = instruction.condition
            circ += If(gate, classical, value)

    return circ


def circuit_to_qiskit(
    circ: Circuit, translate: bool = False
) -> "qiskit.QuantumCircuit":
    """Convert a QuantumFlow's Circuit to a qsikit QuantumCircuit."""

    # In qiskit each gate is defined as a class, and then a method is
    # monkey patched onto QuantumCircuit which will create that gate and
    # append it to the circuit. The method names correspond to the qasm
    # names in QASM_TO_QF

    try:
        import qiskit
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    if translate:
        circ = translate_to_qiskit(circ)

    # Note that with multiple mappings, the second one gets priority.
    QF_TO_QASM = invert_map(QASM_TO_QF)

    # We assume only one QuantumRegister. Represent qubits by index in register
    qreg = qiskit.QuantumRegister(circ.qubit_nb)
    qubit_map = {q: qreg[i] for i, q in enumerate(circ.qubits)}

    qkcircuit = qiskit.QuantumCircuit(qreg)

    for op in circ:
        if op.name == "Initialize":
            name = "initialize"

            # TODO: CHECK
            params = list(op.tensor.transpose().flatten())

            qbs = [qubit_map[qb] for qb in op.qubits]
            getattr(qkcircuit, name)(params, qbs)
        else:
            name = QF_TO_QASM[op.name]
            params = [float(p) for p in op.params]
            qbs = [qubit_map[qb] for qb in op.qubits]
            getattr(qkcircuit, name)(*params, *qbs)

        # TODO: Handle If, Reset, ...

    return qkcircuit


def translate_to_qiskit(circ: Circuit) -> Circuit:
    """Convert QF gates to gates understood by qiskit"""
    return circuit_translate(circ, targets=QISKIT_GATES)


# Deprecated
translate_gates_to_qiskit = translate_to_qiskit


def circuit_to_qasm(circ: Circuit, translate: bool = False) -> str:
    """Convert a QF circuit to a QASM formatted string"""
    circ = circuit_to_qiskit(circ, translate)
    from qiskit import qasm2

    return qasm2.dumps(circ)


def qasm_to_circuit(qasm_str: str) -> Circuit:
    """Convert a QASM circuit to a QF circuit"""
    try:
        import qiskit
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    qc = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
    return qiskit_to_circuit(qc)


# fin
