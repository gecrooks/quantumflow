# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. module:: quantumflow.xforest

Interface to pyQuil and the Rigetti Forest.

.. autofunction:: circuit_to_pyquil
.. autofunction:: pyquil_to_circuit


.. autodata:: QUIL_TO_QF

"""

from itertools import chain

from pyquil.quil import Program as pqProgram
from pyquil.quilbase import Declare as pqDeclare
from pyquil.quilbase import Gate as pqGate
from pyquil.quilbase import Halt as pqHalt
from pyquil.quilbase import Measurement as pqMeasurement
from pyquil.quilbase import Pragma as pqPragma

from . import utils
from .circuits import Circuit
from .ops import Gate, StdGate
from .stdops import Measure

__all__ = [
    "QUIL_TO_QF",
    "circuit_to_pyquil",
    "pyquil_to_circuit",
]


QUIL_TO_QF = {
    "I": "I",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "H": "H",
    "S": "S",
    "T": "T",
    "PhaseShift": "PhaseShift",
    "RX": "Rx",
    "RY": "Ry",
    "RZ": "Rz",
    "CZ": "CZ",
    "CNOT": "CNot",
    "SWAP": "Swap",
    "ISWAP": "ISwap",
    "CPHASE00": "CPhase00",
    "CPHASE01": "CPhase01",
    "CHPHASE10": "CPhase10",
    "CPHASE": "CPhase",
    "PSWAP": "PSwap",
    "CCNOT": "CCNot",
    "CSWAP": "CSwap",
}
"""Map from QUIL operation names to QuantumFlow names"""


def circuit_to_pyquil(circuit: Circuit) -> pqProgram:
    """Convert a QuantumFlow circuit to a pyQuil program"""
    prog = pqProgram()

    QF_TO_QUIL = utils.invert_map(QUIL_TO_QF)
    for elem in circuit:
        if isinstance(elem, Gate) and elem.name in QF_TO_QUIL:
            params = list(elem.params)
            name = QF_TO_QUIL[elem.name]
            prog.gate(name, params, elem.qubits)  # type: ignore
        else:
            raise ValueError("Cannot convert operation to pyquil")  # pragma: no cover

    return prog


def pyquil_to_circuit(program: pqProgram) -> Circuit:
    """Convert a protoquil pyQuil program to a QuantumFlow Circuit"""

    circ = Circuit()
    for inst in program.instructions:
        # print(type(inst))
        if isinstance(inst, pqDeclare):  # Ignore
            continue  # pragma: no cover
        if isinstance(inst, pqHalt):  # Ignore
            continue  # pragma: no cover
        if isinstance(inst, pqPragma):  # Ignore
            continue  # pragma: no cover
        elif isinstance(inst, pqMeasurement):  # pragma: no cover
            circ += Measure(inst.qubit.index)
        elif isinstance(inst, pqGate):
            name = QUIL_TO_QF[inst.name]
            defgate = StdGate.cv_stdgates[name]
            qubits = [q.index for q in inst.qubits]
            gate = defgate(*chain(inst.params, qubits))  # type: ignore
            circ += gate
        else:
            raise ValueError("PyQuil program is not protoquil")  # pragma: no cover

    return circ


# fin
