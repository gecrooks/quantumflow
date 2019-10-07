
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
.. contents:: :local:
.. currentmodule:: quantumflow

Interface between Google's Cirq and QuantumFlow


.. autoclass:: CirqSimulator

.. autofunction:: cirq_to_circuit
.. autofunction:: circuit_to_cirq

.. autofunction:: from_cirq_qubit
.. autofunction:: to_cirq_qubit

"""

# Conventions
# We import cirq as cq
# ('cirq' is too close to our common abbrevation of 'circ' for 'circuit'.)
# cqc: Abbreviation for Cirq circuit

from typing import Iterable

import numpy as np

from .qubits import Qubit, Qubits, asarray
from .ops import Operation
from .states import State, zero_state
from .circuits import Circuit
from .gates import (I, X, Y, Z, S, T, H, TX, TY, TZ, S_H, T_H,
                    CZ, SWAP, ISWAP, CNOT, XX, YY, ZZ,
                    CCNOT, CSWAP, CCZ, FSIM)

import cirq


__all__ = ('from_cirq_qubit', 'to_cirq_qubit',
           'cirq_to_circuit', 'circuit_to_cirq',
           'CirqSimulator')

# DOCME TESTME
# TODO: FSIM
CIRQ_GATESET = frozenset([I, X, Y, Z, S, T, H, TX, TY, TZ, S_H, T_H,
                          CZ, SWAP, ISWAP, CNOT, XX, YY, ZZ,
                          CCNOT, CSWAP, CCZ, FSIM])
"""Set of QuantumFlow gates that we know how to convert to Cirq"""


# TODO
class CirqSimulator(Operation):
    """Interface to Cirq's quantum simulator. Adapts a QF Circuit (or
    other sequence of Operations). Can itself be included in Circuits,
    like any other Operation.

    Note that Cirq uses 64 bit complex floats (QF uses 128 bist), so
    results will not be as accurate.

    """

    def __init__(self, elements: Iterable[Operation] = None) -> None:
        self._circuit = Circuit(elements)
        self._cirq = circuit_to_cirq(self._circuit)

        # TODO: Translate gates

    @property
    def qubits(self) -> Qubits:
        return self._circuit.qubits

    def run(self, ket: State = None) -> State:
        if ket is None:
            qubits = self.qubits
            ket = zero_state(qubits=qubits)

        tensor = asarray(ket.tensor).flatten()
        tensor = np.asarray(tensor, dtype=np.complex64)
        sim = cirq.Simulator()
        res = sim.simulate(self._cirq,
                           initial_state=tensor)
        tensor = res.state_vector()
        return State(tensor, self._circuit.qubits, ket.memory)

    # TODO: evolve, ...


def from_cirq_qubit(qb: cirq.Qid) -> Qubit:
    """
    Convert a cirq qubit (a subtype of Qid) into regular python type.
    A ``LineQubit`` becomes an int, a ``GridQubit`` becomes a tuple of two
    ints, and ``NamedQubit`` (and anything else) becomes a string
    """
    if isinstance(qb, cirq.LineQubit):
        return qb.x
    elif isinstance(qb, cirq.GridQubit):
        return (qb.row, qb.col)
    elif isinstance(qb, cirq.NamedQubit):
        return qb.name
    return str(qb)  # pragma: no cover


def to_cirq_qubit(qubit: Qubit) -> cirq.Qid:
    """Convert qubit names (any python object) into
    cirq qubits (subtypes of Qid). Returns either
    a LineQubit (for ints), GridQubit (for tuples of row and column),
    or a NamedQubit for all other objects.
    """
    if isinstance(qubit, int):
        return cirq.LineQubit(qubit)
    elif isinstance(qubit, tuple) and len(qubit) == 2 \
            and isinstance(qubit[0], int) and isinstance(qubit[1], int):
        return cirq.GridQubit(row=qubit[0], col=qubit[1])
    return cirq.NamedQubit(str(qubit))


# TODO: ops.FSimGate
def cirq_to_circuit(cqc: cirq.Circuit) -> Circuit:
    """Convert a Cirq circuit to a QuantumFlow circuit"""

    simple_gates = {
        cirq.ops.CSwapGate: CSWAP,
        cirq.ops.common_gates.IdentityGate: I,
    }

    exponent_gates = {
        cirq.ops.pauli_gates._PauliX: X,
        cirq.ops.pauli_gates._PauliY: Y,
        cirq.ops.pauli_gates._PauliZ: Z,
        cirq.ops.XPowGate: X,
        cirq.ops.YPowGate: Y,
        cirq.ops.ZPowGate: Z,
        cirq.ops.HPowGate: H,
        cirq.ops.CZPowGate: CZ,
        cirq.ops.CNotPowGate: CNOT,
        cirq.ops.SwapPowGate: SWAP,
        cirq.ops.ISwapPowGate: ISWAP,
        cirq.ops.CCXPowGate: CCNOT,
        cirq.ops.CCZPowGate: CCZ,
    }

    parity_gates = {
        cirq.ops.XXPowGate: XX,
        cirq.ops.YYPowGate: YY,
        cirq.ops.ZZPowGate: ZZ
    }

    circ = Circuit()
    qubit_map = {q: from_cirq_qubit(q) for q in cqc.all_qubits()}
    for op in cqc.all_operations():
        gatetype = type(op.gate)

        qbs = [qubit_map[qb] for qb in op.qubits]

        if gatetype in simple_gates:
            circ += simple_gates[gatetype](*qbs)
        elif gatetype in exponent_gates:
            gate = exponent_gates[gatetype](*qbs)
            if op.gate.exponent != 1:
                gate **= op.gate.exponent
            circ += gate
        elif gatetype in parity_gates:
            t = op.gate.exponent
            circ += parity_gates[gatetype](t, *qbs)
        else:
            raise NotImplementedError(str(op.gate))  # pragma: nocover

    circ = circ.specialize()

    return circ


def circuit_to_cirq(circ: Circuit) -> cirq.Circuit:
    """Convert a QuantumFlow circuit to a Cirq circuit."""
    qubit_map = {q: to_cirq_qubit(q) for q in circ.qubits}

    cqc = cirq.Circuit()

    operations = {
        I:      cirq.I,
        X:      cirq.X,
        Y:      cirq.Y,
        Z:      cirq.Z,
        S:      cirq.S,
        T:      cirq.T,
        H:      cirq.H,
        CNOT:   cirq.CNOT,
        CZ:     cirq.CZ,
        SWAP:   cirq.SWAP,
        ISWAP:  cirq.ISWAP,
        CCZ:    cirq.CCZ,
        CCNOT:  cirq.CCX,
        CSWAP:  cirq.CSWAP,
    }

    # TODO: TH -> cirq.ops.HPowGate,
    turn_gates = {
        TX:     cirq.ops.XPowGate,
        TY:     cirq.ops.YPowGate,
        TZ:     cirq.ops.ZPowGate,
        XX:     cirq.ops.XXPowGate,
        YY:     cirq.ops.YYPowGate,
        ZZ:     cirq.ops.ZZPowGate,
    }

    for op in circ:
        qbs = [qubit_map[qb] for qb in op.qubits]

        if type(op) in operations:
            cqc.append(operations[type(op)].on(*qbs))
        elif type(op) in turn_gates:
            t = op.params['t']
            cqc.append(turn_gates[type(op)]().on(*qbs) ** t)
        else:
            raise NotImplementedError(str(op))

    return cqc
