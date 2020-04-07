
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
# cqc: Abbreviation for Cirq circuit

from typing import Iterable, cast, List, Type

import numpy as np

from .qubits import Qubit, Qubits, asarray
from .ops import Operation, Unitary, Gate
from .states import State, zero_state
from .circuits import Circuit
from .gates import (I, X, Y, Z, S, T, H, TX, TY, TZ,
                    CZ, SWAP, ISWAP, CNOT, XX, YY, ZZ,
                    CCNOT, CSWAP, CCZ, FSim, IDEN)
from .translate import select_translators, circuit_translate

from .backends import backend as bk

import cirq

pi = bk.pi
PI = bk.PI


__all__ = ('from_cirq_qubit', 'to_cirq_qubit',
           'cirq_to_circuit', 'circuit_to_cirq',
           'CirqSimulator')

# DOCME TESTME
CIRQ_GATES: List[Type[Gate]] = [
    I, X, Y, Z, S, T, H, TX, TY, TZ,
    CZ, SWAP, ISWAP, CNOT, XX, YY, ZZ,
    CCNOT, CSWAP, CCZ, FSim, IDEN]
"""List of QuantumFlow gates that we know how to convert to Cirq"""


class CirqSimulator(Operation):
    """Interface to the Cirq quantum simulator. Adapts a QF Circuit (or
    other sequence of Operations). Can itself be included in Circuits,
    like any other Operation.

    Note that Cirq uses 64 bit complex floats (QF uses 128 bits), so
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
        res = cast(cirq.WaveFunctionTrialResult, res)
        tensor = res.state_vector()
        return State(tensor, ket.qubits, ket.memory)

    # TODO: evolve, ...


def from_cirq_qubit(qb: cirq.Qid) -> Qubit:
    """
    Convert a cirq qubit (a subtype of Qid) into regular python type.
    A ``LineQubit`` becomes an int, a ``GridQubit`` becomes a tuple of two
    integers, and ``NamedQubit`` (and anything else) becomes a string
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
    a LineQubit (for integers), GridQubit (for tuples of row and column),
    or a NamedQubit for all other objects.
    """
    if isinstance(qubit, int):
        return cirq.LineQubit(qubit)
    elif isinstance(qubit, tuple) and len(qubit) == 2 \
            and isinstance(qubit[0], int) and isinstance(qubit[1], int):
        return cirq.GridQubit(row=qubit[0], col=qubit[1])
    return cirq.NamedQubit(str(qubit))


def cirq_to_circuit(cqc: cirq.Circuit) -> Circuit:
    """Convert a Cirq circuit to a QuantumFlow circuit"""

    simple_gates = {
        cirq.CSwapGate: CSWAP,
        cirq.IdentityGate: IDEN,
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
        cirq.CNotPowGate: CNOT,
        cirq.SwapPowGate: SWAP,
        cirq.ISwapPowGate: ISWAP,
        cirq.CCXPowGate: CCNOT,
        cirq.CCZPowGate: CCZ,
    }

    parity_gates = {
        cirq.XXPowGate: XX,
        cirq.YYPowGate: YY,
        cirq.ZZPowGate: ZZ
    }

    decomposable_gates = {'PhasedISwapPowGate', 'PhasedXZGate'}

    circ = Circuit()
    qubit_map = {q: from_cirq_qubit(q) for q in cqc.all_qubits()}
    for op in cqc.all_operations():
        gatetype = type(op.gate)

        # Extra check needed for cirq 0.6, but not 0.7?
        if isinstance(op, cirq.ops.identity.IdentityOperation):
            gatetype = cirq.IdentityGate             # pragma: no cover

        qbs = [qubit_map[qb] for qb in op.qubits]
        t = getattr(op.gate, 'exponent', 1)

        if gatetype in simple_gates:
            circ += simple_gates[gatetype](*qbs)     # type: ignore
        elif gatetype in exponent_gates:
            gate = exponent_gates[gatetype](*qbs)    # type: ignore
            if t != 1:
                gate **= t
            circ += gate
        elif gatetype in parity_gates:
            circ += parity_gates[gatetype](t, *qbs)   # type: ignore
        elif gatetype.__name__ in decomposable_gates:
            subcqc = cirq.Circuit(op._decompose_())   # type: ignore
            circ.extend(cirq_to_circuit(subcqc))
        elif gatetype.__name__ == 'FSimGate':
            circ += FSim(op.gate.theta, op.gate.phi, *qbs)  # type: ignore
        elif gatetype.__name__ == 'MatrixGate':
            matrix = op.gate._matrix        # type: ignore
            circ += Unitary(matrix, *qbs)
        else:
            raise NotImplementedError(str(op.gate))  # pragma: no cover

    circ = circ.specialize()

    return circ


def circuit_to_cirq(circ: Circuit,
                    translate: bool = False) -> cirq.Circuit:
    """Convert a QuantumFlow circuit to a Cirq circuit."""

    if translate:
        circ = translate_to_cirq(circ)

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
        TX:     cirq.X,
        TY:     cirq.Y,
        TZ:     cirq.Z,
        XX:     cirq.XX,
        YY:     cirq.YY,
        ZZ:     cirq.ZZ,
    }

    for op in circ:
        qbs = [qubit_map[qb] for qb in op.qubits]

        if type(op) in operations:
            cqc.append(operations[type(op)].on(*qbs))
        elif type(op) in turn_gates:
            t = op.params['t']
            cqc.append(turn_gates[type(op)](*qbs) ** t)
        elif isinstance(op, IDEN):
            gate = cirq.IdentityGate(op.qubit_nb).on(*qbs)
            cqc.append(gate)
        elif isinstance(op, FSim):
            theta, phi = op.params.values()
            gate = cirq.FSimGate(theta=theta, phi=phi).on(*qbs)
            cqc.append(gate)
        elif isinstance(op, Unitary):
            matrix = bk.evaluate(op.asoperator())
            gate = cirq.MatrixGate(matrix).on(*qbs)
            cqc.append(gate)
        else:
            raise NotImplementedError(str(op))

    return cqc


def translate_to_cirq(circ: Circuit) -> Circuit:
    """Convert QF gates to gates understood by cirq"""
    target_gates = CIRQ_GATES
    trans = select_translators(target_gates)
    circ = circuit_translate(circ, trans)
    return circ
