# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
.. contents:: :local:
.. currentmodule:: quantumflow.xcirq

Interface between Google's Cirq and QuantumFlow


.. autoclass:: CirqSimulator

.. autofunction:: cirq_to_circuit
.. autofunction:: circuit_to_cirq

.. autofunction:: from_cirq_qubit
.. autofunction:: to_cirq_qubit

"""

# Conventions
# cqc: Abbreviation for Cirq circuit

from typing import TYPE_CHECKING, Optional

import numpy as np

from . import var
from .circuits import Circuit
from .gatesets import CIRQ_GATES
from .ops import Unitary
from .qubits import Qubit
from .states import State, zero_state
from .stdgates import (
    CCZ,
    CZ,
    XX,
    YY,
    ZZ,
    CCNot,
    CNot,
    CSwap,
    FSim,
    H,
    I,
    ISwap,
    S,
    Swap,
    T,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
)
from .stdops import Simulator
from .translate.translations import circuit_translate

if TYPE_CHECKING:
    import cirq  # pragma: no cover

__all__ = (
    "from_cirq_qubit",
    "to_cirq_qubit",
    "cirq_to_circuit",
    "circuit_to_cirq",
    "CirqSimulator",
    "CIRQ_GATES",
)


_IMPORT_ERROR_MSG = """External dependency 'cirq' not installed. Install
with 'pip install cirq'"""


class CirqSimulator(Simulator):
    """Interface to the Cirq quantum simulator. Adapts a QF Circuit (or
    other sequence of Operations). Can itself be included in Circuits,
    like any other Operation.

    Note that Cirq uses 64 bit complex floats (QF uses 128 bits), so
    results will not be as accurate.
    """

    def __init__(self, circ: Circuit) -> None:
        super().__init__(circ)
        self._cqc = circuit_to_cirq(self.circuit, translate=True)

    def run(self, ket: Optional[State] = None) -> State:
        try:
            import cirq
        except ModuleNotFoundError as err:  # pragma: no cover
            raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

        if ket is None:
            qubits = self.qubits
            ket = zero_state(qubits=qubits)

        tensor = ket.tensor.flatten()
        tensor = np.asarray(tensor, dtype=np.complex64)
        sim = cirq.Simulator()
        res = sim.simulate(self._cqc, initial_state=tensor)
        tensor = res.state_vector()  # type:ignore  # Needed for cirq <0.10.0
        return State(tensor, ket.qubits, ket.memory)


def from_cirq_qubit(qb: "cirq.Qid") -> Qubit:
    """
    Convert a cirq qubit (a subtype of Qid) into regular python type.
    A ``LineQubit`` becomes an int, a ``GridQubit`` becomes a tuple of two
    integers, and ``NamedQubit`` (and anything else) becomes a string
    """
    try:
        import cirq
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    if isinstance(qb, cirq.LineQubit):
        return qb.x
    elif isinstance(qb, cirq.GridQubit):
        return (qb.row, qb.col)
    elif isinstance(qb, cirq.NamedQubit):
        return qb.name
    return str(qb)  # pragma: no cover


def to_cirq_qubit(qubit: Qubit) -> "cirq.Qid":
    """Convert qubit names (any python object) into
    cirq qubits (subtypes of Qid). Returns either
    a LineQubit (for integers), GridQubit (for tuples of row and column),
    or a NamedQubit for all other objects.
    """
    try:
        import cirq
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    if isinstance(qubit, int):
        return cirq.LineQubit(qubit)
    elif (
        isinstance(qubit, tuple)
        and len(qubit) == 2
        and isinstance(qubit[0], int)
        and isinstance(qubit[1], int)
    ):
        return cirq.GridQubit(row=qubit[0], col=qubit[1])
    return cirq.NamedQubit(str(qubit))


def cirq_to_circuit(cqc: "cirq.Circuit") -> Circuit:
    """Convert a Cirq circuit to a QuantumFlow circuit"""
    try:
        import cirq
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    simple_gates = {
        cirq.CSwapGate: CSwap,
        # cirq.IdentityGate: I,
    }

    exponent_gates = {
        cirq.ops.pauli_gates._PauliX: X,
        cirq.ops.pauli_gates._PauliY: Y,
        cirq.ops.pauli_gates._PauliZ: Z,
        cirq.XPowGate: X,
        cirq.YPowGate: Y,
        cirq.ZPowGate: Z,
        cirq.HPowGate: H,
        cirq.CZPowGate: CZ,
        cirq.CNotPowGate: CNot,
        cirq.SwapPowGate: Swap,
        cirq.ISwapPowGate: ISwap,
        cirq.CCXPowGate: CCNot,
        cirq.CCZPowGate: CCZ,
    }

    parity_gates = {cirq.XXPowGate: XX, cirq.YYPowGate: YY, cirq.ZZPowGate: ZZ}

    decomposable_gates = {"PhasedISwapPowGate", "PhasedXZGate"}

    circ = Circuit()
    qubit_map = {q: from_cirq_qubit(q) for q in cqc.all_qubits()}
    for op in cqc.all_operations():
        gatetype = type(op.gate)

        qbs = [qubit_map[qb] for qb in op.qubits]
        t = getattr(op.gate, "exponent", 1)

        if gatetype is cirq.IdentityGate:
            for q in qbs:
                circ += I(q)
        elif gatetype in simple_gates:
            circ += simple_gates[gatetype](*qbs)  # type: ignore
        elif gatetype in exponent_gates:
            gate = exponent_gates[gatetype](*qbs)  # type: ignore
            if t != 1:
                gate **= t
            circ += gate
        elif gatetype in parity_gates:
            circ += parity_gates[gatetype](t, *qbs)  # type: ignore
        elif gatetype.__name__ in decomposable_gates:
            subcqc = cirq.Circuit(op._decompose_())  # type: ignore
            circ += cirq_to_circuit(subcqc)
        elif gatetype.__name__ == "FSimGate":
            circ += FSim(op.gate.theta, op.gate.phi, *qbs)  # type: ignore
        elif gatetype.__name__ == "MatrixGate":
            matrix = op.gate._matrix  # type: ignore
            circ += Unitary(matrix, qbs)
        elif gatetype.__name__ == "Rx":  # Added in cirq 0.11.0
            circ += X(*qbs) ** t
        elif gatetype.__name__ == "Ry":  # Added in cirq 0.11.0
            circ += Y(*qbs) ** t
        elif gatetype.__name__ == "Rz":  # Added in cirq 0.11.0
            circ += Z(*qbs) ** t
        else:
            raise NotImplementedError(str(op.gate))  # pragma: no cover

    circ = circ.specialize()

    return circ


def circuit_to_cirq(circ: Circuit, translate: bool = False) -> "cirq.Circuit":
    """Convert a QuantumFlow circuit to a Cirq circuit."""
    try:
        import cirq
    except ModuleNotFoundError as err:  # pragma: no cover
        raise ModuleNotFoundError(_IMPORT_ERROR_MSG) from err

    if translate:
        circ = translate_to_cirq(circ)

    qubit_map = {q: to_cirq_qubit(q) for q in circ.qubits}

    cqc = cirq.Circuit()

    operations = {
        I: cirq.I,
        X: cirq.X,
        Y: cirq.Y,
        Z: cirq.Z,
        S: cirq.S,
        T: cirq.T,
        H: cirq.H,
        CNot: cirq.CNOT,
        CZ: cirq.CZ,
        Swap: cirq.SWAP,
        ISwap: cirq.ISWAP,
        CCZ: cirq.CCZ,
        CCNot: cirq.CCX,
        CSwap: cirq.CSWAP,
    }

    # TODO: HPow -> cirq.ops.HPowGate,
    turn_gates = {
        XPow: cirq.X,
        YPow: cirq.Y,
        ZPow: cirq.Z,
        XX: cirq.XX,
        YY: cirq.YY,
        ZZ: cirq.ZZ,
    }

    for op in circ:
        qbs = [qubit_map[qb] for qb in op.qubits]

        if type(op) in operations:
            cqc.append(operations[type(op)].on(*qbs))
        elif type(op) in turn_gates:
            t = op.param("t")
            t = var.asfloat(t)
            cqc.append(turn_gates[type(op)](*qbs) ** t)
        elif isinstance(op, FSim):
            theta, phi = op.params
            theta = var.asfloat(theta)
            phi = var.asfloat(phi)
            gate = cirq.FSimGate(theta=theta, phi=phi).on(*qbs)
            cqc.append(gate)
        elif isinstance(op, Unitary):
            matrix = op.asoperator()
            gate = cirq.MatrixGate(matrix).on(*qbs)
            cqc.append(gate)
        else:
            raise NotImplementedError(repr(op))

    return cqc


def translate_to_cirq(circ: Circuit) -> Circuit:
    """Convert QF gates to gates understood by cirq"""
    circ = circuit_translate(circ, targets=CIRQ_GATES)
    return circ


# fin
