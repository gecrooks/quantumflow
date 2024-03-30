# Copyright 2021-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
.. contents:: :local:
.. currentmodule:: quantumflow.xqutip

Interface between QuTiP and QuantumFlow


==========================  =========== ============
Description                 QuantumFlow  QuTiP
==========================  =========== ============

* One qubit gates
Global phase                Ph          GLOBALPHASE
Pauli-X                     X           X
Pauli-Y                     Y           Y
Pauli-Z                     Z           Z
Hadamard                    H           SNOT
X-rotations                 Rx          RX
Y-rotations                 Ry          RY
Z-rotations                 Rz          RZ
Sqrt of Z                   S           S
Sqrt of S                   T           T
Phase shift                 PhaseShift  PHASEGATE
Sqrt of X                   V           SQRTNOT

* Two qubit gates
Controlled-Not              CNOT        CNOT
Controlled-Y                CY          CY
Controlled-Z                CZ          CSIGN
Controlled-S                .           CS
Controlled-T                .           CT
Swap                        Swap        SWAP
iSwap                       ISwap       ISWAP
Sqrt-iSWAP                  SqrtISwap   SQRTISWAP
Sqrt-SWAP                   SqrtSwap    SQRTSWAP
Controlled-RX               .           CRX
Controlled-RY               .           CRY
Controlled-RZ               CRz         CRZ

* Three qubit gates
Toffoli                     CCNot       TOFFOLI
Fredkin                     CSwap       FREDKIN

==========================  =========== ============



.. autofunction:: qutip_to_circuit
.. autofunction:: circuit_to_qutip
.. autofunction:: translate_to_qutip


"""
# Needs update to qutip: CZ, B, swapalpha, CPHASE
# QuTiPs CZ is defined weird. Could use CSIGN instead.

# QuTiP gates not supported
# CZ                Requires bugfix in QuTiP (Use CSIGN instead)
# B                 Requires bugfix in QuTiP
# swapalpha         Requires bugfix in QuTiP
# molmer_sorensen   Gate defined but not supported by QuTiP's QubitCircuit
# qrot
# QASMU

from typing import TYPE_CHECKING, Dict, Optional, Type, cast

from . import var
from .circuits import Circuit
from .gatesets import QUTIP_GATES
from .ops import Gate
from .stdgates import (  # B,; Exch,
    CS,
    CT,
    CY,
    CZ,
    CCNot,
    CNot,
    CPhase,
    CSwap,
    H,
    I,
    ISwap,
    Ph,
    PhaseShift,
    Rx,
    Ry,
    Rz,
    S,
    SqrtISwap,
    SqrtSwap,
    Swap,
    T,
    V,
    X,
    Y,
    Z,
)
from .translate import circuit_translate
from .utils import invert_map

if TYPE_CHECKING:
    import QubitCircuit  # pragma: no cover

__all__ = ("qutip_to_circuit", "circuit_to_qutip", "translate_to_qutip", "QUTIP_GATES")


_IMPORT_ERROR_MSG = """External dependency 'qutip' not installed. Install
with 'pip install qutip'"""


_QUTIP_GATE_NAMES: Dict[Type[Gate], Optional[str]] = {
    I: None,
    Ph: "GLOBALPHASE",
    Rx: "RX",
    Ry: "RY",
    Rz: "RZ",
    V: "SQRTNOT",
    H: "SNOT",
    PhaseShift: "PHASEGATE",
    CY: "CY",
    CZ: "CSIGN",
    S: "S",
    T: "T",
    X: "X",
    Y: "Y",
    Z: "Z",
    CS: "CS",
    CT: "CT",
    CNot: "CNOT",
    Swap: "SWAP",
    ISwap: "ISWAP",
    SqrtSwap: "SQRTSWAP",
    SqrtISwap: "SQRTISWAP",
    CCNot: "TOFFOLI",
    CSwap: "FREDKIN",
    CPhase: "CPHASE",
    # B: "BERKELEY",
    # Exch: "SWAPalpha",  # Same gate, different parameterization
}

# QUTIP_GATES: Tuple[Type[Gate], ...] = tuple(_QUTIP_GATE_NAMES.keys())
# """List of QuantumFlow gates that we know how to convert directly to and from QuTiP"""

_QUTIP_NAME_GATES = invert_map(_QUTIP_GATE_NAMES)
# _QUTIP_NAME_GATES["CSIGN"] = CZ


def qutip_to_circuit(qubitcircuit: "QubitCircuit") -> Circuit:
    """Convert a QuTiP circuit to a QuantumFlow circuit"""
    QUTIP_NAME_GATES = invert_map(_QUTIP_GATE_NAMES)
    circ = Circuit()

    for op in qubitcircuit.gates:
        if op.name not in QUTIP_NAME_GATES:
            raise ValueError(
                f"Cannot convert operation from qutip: {op.name}"
            )  # pragma: no cover

        gate_type = QUTIP_NAME_GATES[op.name]

        if op.controls is not None:
            qubits = tuple(op.controls) + tuple(op.targets)
        else:
            qubits = tuple(op.targets)

        if op.arg_value is not None:
            # if op.name == "SWAPalpha":
            #     gate = gate_type(op.arg_value / 2, *qubits)
            # else:
            gate = gate_type(op.arg_value, *qubits)

        else:
            gate = gate_type(*qubits)

        circ += gate

    return circ


def circuit_to_qutip(circ: Circuit, translate: bool = False) -> "QubitCircuit":
    """Convert a QuantumFlow circuit to a QuTiP circuit."""
    try:
        from qutip.qip.circuit import QubitCircuit
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    _ctrl_gates = [
        "CNOT",
        "CSIGN",
        "CRX",
        "CRY",
        "CRZ",
        "CY",
        "CZ",
        "CS",
        "CT",
        "CPHASE",
    ]
    _para_gates = [
        "RX",
        "RY",
        "RZ",
        "CPHASE",
        "SWAPalpha",
        "PHASEGATE",
        "GLOBALPHASE",
        "CRX",
        "CRY",
        "CRZ",
        "QASMU",
    ]

    if translate:
        circ = translate_to_qutip(circ)

    for q in circ.qubits:
        if not isinstance(q, int):
            raise ValueError("QuTiP qubits must be integers")

    N = cast(int, max(circ.qubits))
    qbc = QubitCircuit(N + 1)

    for op in circ:
        if type(op) not in _QUTIP_GATE_NAMES:
            raise ValueError(f"Cannot convert operation to qutip: {op}")

        gate_name = _QUTIP_GATE_NAMES[cast(Type[Gate], type(op))]

        if gate_name is None:
            continue

        if gate_name in _para_gates:
            arg_value = var.asfloat(op.params[0])
            # if gate_name == "SWAPalpha":
            #     arg_value *= 2
        else:
            arg_value = None

        if gate_name in _ctrl_gates or gate_name == "FREDKIN" or gate_name == "CPHASE":
            controls = [op.qubits[0]]
            targets = list(op.qubits[1:])
        elif gate_name == "TOFFOLI":
            controls = list(op.qubits[0:2])
            targets = [op.qubits[2]]
        else:
            controls = None
            targets = list(op.qubits)

        qbc.add_gate(gate_name, controls=controls, targets=targets, arg_value=arg_value)

    return qbc


def translate_to_qutip(circ: Circuit) -> Circuit:
    """Convert QF gates to gates understood by qutip"""
    circ = circuit_translate(circ, targets=QUTIP_GATES)
    return circ


# fin
