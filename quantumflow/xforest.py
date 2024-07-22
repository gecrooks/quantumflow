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

import numpy as np

from itertools import chain
from typing import TYPE_CHECKING

from . import utils, var
from .circuits import Circuit
from .gatesets import QUIL_GATES
from .ops import Gate
from .stdgates import STDGATES
from .stdops import Measure
from .translate import circuit_translate


if TYPE_CHECKING:
    from pyquil.quil import Program as pqProgram  # pragma: no cover

__all__ = [
    "QUIL_GATES",
    "QUIL_TO_QF",
    "circuit_to_pyquil",
    "pyquil_to_circuit",
]

_IMPORT_ERROR_MSG = """External dependency 'pyquil' not installed. Install
with 'pip install pyquil'"""


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


def circuit_to_pyquil(circ: Circuit, translate: bool = False) -> "pqProgram":
    """Convert a QuantumFlow circuit to a pyQuil program"""

    try:
        from pyquil.quil import Program as pqProgram
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    if translate:
        circ = translate_to_pyquil(circ)

    prog = pqProgram()

    QF_TO_QUIL = utils.invert_map(QUIL_TO_QF)
    for elem in circ:
        if isinstance(elem, Gate) and elem.name in QF_TO_QUIL:
            params = list(var.asfloat(p) for p in elem.params)
            name = QF_TO_QUIL[elem.name]
            prog.gate(name, params, elem.qubits)  # type: ignore
        else:
            raise ValueError("Cannot convert operation to pyquil")  # pragma: no cover

    return prog


def pyquil_to_circuit(program: "pqProgram") -> Circuit:
    """Convert a protoquil pyQuil program to a QuantumFlow Circuit"""
    try:
        from pyquil.quilbase import Declare as pqDeclare
        from pyquil.quilbase import Gate as pqGate
        from pyquil.quilbase import Halt as pqHalt
        from pyquil.quilbase import Measurement as pqMeasurement
        from pyquil.quilbase import Pragma as pqPragma
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

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
            circ += Measure(inst.qubit.index) # type: ignore
        elif isinstance(inst, pqGate):
            name = QUIL_TO_QF[inst.name]
            defgate = STDGATES[name]
            qubits = [q.index for q in inst.qubits] # type: ignore
            gate = defgate(*chain((np.real(p) for p in inst.params), qubits))  # type: ignore
            circ += gate
        else:
            raise ValueError("PyQuil program is not protoquil")  # pragma: no cover

    return circ


def translate_to_pyquil(circ: Circuit) -> Circuit:
    """Convert a circuit to gates understood by pyquil"""
    circ = circuit_translate(circ, targets=QUIL_GATES)
    return circ


# fin
