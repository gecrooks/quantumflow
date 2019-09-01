"""
========================
Miscellaneous Operations
========================


.. currentmodule:: quantumflow

Various standard operations on quantum states, which arn't gates
or channels.

.. autofunction :: dagger
.. autoclass :: Measure
.. autoclass :: Reset
.. autoclass :: Barrier
.. autoclass :: Projection
.. autoclass :: QubitPermutation
.. autoclass :: Store
.. autoclass :: If
.. autoclass :: Display
.. autoclass :: StoreState
"""


from typing import Sequence, Hashable, Callable, Any

import numpy as np

from sympy.combinatorics import Permutation

from .qubits import Qubit, Qubits, asarray, QubitVector
from .states import State, Density
from .ops import Operation, Gate, Channel
from .gates import P0, P1
from .gates import SWAP, I
from .circuits import Circuit
from . import backend as bk


__all__ = ['dagger', 'Measure', 'Reset', 'Barrier', 'Store',
           'If', 'Projection', 'QubitPermutation', 'Display', 'StoreState']


def dagger(elem: Operation) -> Operation:
    """Return the complex conjugate of the Operation"""
    return elem.H


class Measure(Operation):
    """Measure a quantum bit and copy result to a classical bit"""
    def __init__(self, qubit: Qubit, cbit: Hashable = None) -> None:
        self.qubit = qubit
        self.cbit = cbit

    def __str__(self) -> str:
        if self.cbit is not None:
            return '{} {} {}'.format(self.name.upper(), self.qubit, self.cbit)
        return '{} {}'.format(self.name.upper(), self.qubit)

    @property
    def qubits(self) -> Qubits:
        return [self.qubit]

    def run(self, ket: State) -> State:
        prob_zero = asarray(P0(self.qubit).run(ket).norm())

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            ket = P0(self.qubit).run(ket).normalize()
            if self.cbit is not None:
                ket = ket.store({self.cbit: 0})
        else:  # measure one
            ket = P1(self.qubit).run(ket).normalize()
            if self.cbit is not None:
                ket = ket.store({self.cbit: 1})
        return ket

    def evolve(self, rho: Density) -> Density:
        p0 = P0(self.qubit).aschannel()
        p1 = P1(self.qubit).aschannel()

        prob_zero = asarray(p0.evolve(rho).norm())

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            rho = p0.evolve(rho).normalize()
            if self.cbit is not None:
                rho = rho.store({self.cbit: 0})
        else:  # measure one
            rho = p1.evolve(rho).normalize()
            if self.cbit is not None:
                rho = rho.store({self.cbit: 1})
        return rho


class Reset(Operation):
    r"""An operation that resets qubits to zero irrespective of the
    initial state.
    """
    def __init__(self, *qubits: Qubit) -> None:
        if not qubits:
            qubits = ()
        self._qubits = tuple(qubits)

        self.vec = QubitVector([[1, 1], [0, 0]], [0])

    @property
    def H(self) -> 'Reset':
        return self  # Hermitian

    def run(self, ket: State) -> State:
        if self.qubits:
            qubits = self.qubits
        else:
            qubits = ket.qubits

        indices = [ket.qubits.index(q) for q in qubits]
        ket_tensor = ket.tensor
        for idx in indices:
            ket_tensor = bk.tensormul(self.vec.tensor, ket_tensor, [idx])
        ket = State(ket_tensor, ket.qubits, ket.memory).normalize()
        return ket

    def evolve(self, rho: Density) -> Density:
        # TODO
        raise TypeError('Not yet implemented')

    def asgate(self) -> Gate:
        raise TypeError('Reset not convertible to Gate')

    def aschannel(self) -> Channel:
        raise TypeError('Reset not convertible to Channel')

    def __str__(self) -> str:
        if self.qubits:
            return 'RESET ' + ' '.join([str(q) for q in self.qubits])
        return 'RESET'


class Barrier(Operation):
    """An operation that prevents reordering of operations across the barrier.
    Has no effect on the quantum state."""
    def __init__(self, *qubits: Qubit) -> None:
        self._qubits = qubits

    @property
    def H(self) -> 'Barrier':
        return self  # Hermitian

    def run(self, ket: State) -> State:
        return ket  # NOP

    def evolve(self, rho: Density) -> Density:
        return rho  # NOP

    def __str__(self) -> str:
        return self.name.upper() + ' ' + ' '.join(str(q) for q in self.qubits)


class Projection(Operation):
    """A projection operator, representated as a sequence of state vectors
    """

    # TODO: evolve(), asgate(), aschannel()

    def __init__(self,
                 states: Sequence[State]):
        self.states = states

    @property
    def qubits(self) -> Qubits:
        """Return the qubits that this operation acts upon"""
        qbs = [q for state in self.states for q in state.qubits]    # gather
        qbs = list(set(qbs))                                        # unique
        qbs = sorted(qbs)                                           # sort
        return tuple(qbs)

    def run(self, ket: State) -> State:
        """Apply the action of this operation upon a pure state"""

        tensor = sum(state.tensor * bk.inner(state.tensor, ket.tensor)
                     for state in self.states)
        return State(tensor, qubits=ket.qubits)

    @property
    def H(self) -> 'Projection':
        return self


class QubitPermutation(Operation):
    """A permutation of qubits. A generalized multi-qubit SWAP."""
    def __init__(self, qubits_in: Qubits, qubits_out: Qubits) -> None:
        if set(qubits_in) != set(qubits_out):
            raise ValueError("Incompatible sets of qubits")

        self.qubits_out = tuple(qubits_out)
        self.qubits_in = tuple(qubits_in)

    @classmethod
    def from_circuit(cls, circ: Circuit) -> 'QubitPermutation':
        """Create a qubit pertumtation from a circuit of swap gates"""
        qubits_in = circ.qubits
        N = circ.qubit_nb
        perm = list(range(N))
        for elem in circ:
            if isinstance(elem, I):
                continue
            assert isinstance(elem, SWAP)
            q0, q1 = elem.qubits
            i0 = qubits_in.index(q0)
            i1 = qubits_in.index(q1)
            perm[i1], perm[i0] = perm[i0], perm[i1]
            # TODO: Should also accept QubitPermutations

        qubits_out = [qubits_in[p] for p in perm]
        return cls(qubits_in, qubits_out)

    @property
    def qubits(self) -> Qubits:
        return self.qubits_in

    @property
    def H(self) -> Operation:
        return QubitPermutation(self.qubits_out, self.qubits_in)

    def run(self, ket: State) -> State:
        qubits = ket.qubits
        N = ket.qubit_nb

        perm = list(range(N))
        for q0, q1 in zip(self.qubits_in, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        tensor = bk.transpose(ket.tensor, perm)

        return State(tensor, qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        qubits = rho.qubits
        N = rho.qubit_nb

        perm = list(range(N))
        for q0, q1 in zip(self.qubits_in, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)
        perm.extend([idx+N for idx in perm])

        tensor = bk.transpose(rho.tensor, perm)

        return Density(tensor, qubits, rho.memory)

    def asgate(self) -> Gate:
        N = self.qubit_nb
        qubits = self.qubits

        perm = list(range(2*N))
        for q0, q1 in zip(qubits, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        U = np.eye(2**N)
        U = np.reshape(U, [2]*2*N)
        U = np.transpose(U, perm)

        return Gate(U, qubits=qubits)

    def aschannel(self) -> 'Channel':
        return self.asgate().aschannel()

    def ascircuit(self) -> Circuit:
        """
        Returns a SWAP network for this permutation, assuming all-to-all
        connectivity.
        """
        circ = Circuit()
        qubits = self.qubits

        perm = [self.qubits.index(q) for q in self.qubits_out]
        for idx0, idx1 in (Permutation(perm).transpositions()):
            circ += SWAP(qubits[idx0], qubits[idx1])

        return circ


# TESTME
class Store(Operation):
    """Store a value in the classical memory of the state"""
    def __init__(self,
                 key: Hashable,
                 value: bool = True) -> None:
        super().__init__()
        self.key = key
        self.value = value

    def run(self, ket: State) -> State:
        return ket.store({self.key: self.value})

    def evolve(self, rho: Density) -> Density:
        return rho.store({self.key: self.value})


class If(Operation):
    """
    Look up key in classical memory, and apply the given
    quantum operation only if the truth value is the same as value.
    """
    def __init__(self, elem: Operation,
                 key: Hashable,
                 value: bool = True) -> None:
        super().__init__()
        self.element = elem
        self.key = key
        self.value = value

    @property
    def qubits(self) -> Qubits:
        return self.element.qubits

    def run(self, ket: State) -> State:
        if ket.memory[self.key] == self.value:
            ket = self.element.run(ket)
        return ket

    def evolve(self, rho: Density) -> Density:
        if rho.memory[self.key] == self.value:
            rho = self.element.evolve(rho)
        return rho


# TESTME
class Display(Operation):
    """A Display is an operation that extracts information from the
    quantum state and stores it in classical memory, without performing
    any effect on the qubits.
    """
    # Terminology comes from cirq: cirq/ops/display.py
    def __init__(self,
                 key: Hashable,
                 action: Callable[[State], Any]) -> None:
        super().__init__()
        self.key = key
        self.action = action

    def run(self, ket: State) -> State:
        return ket.store({self.key: self.action(ket)})

    def evolve(self, rho: Density) -> Density:
        return rho.store({self.key: self.action(rho)})


# TESTME
class StoreState(Display):
    """
    Store a copy of the state in the classical memory. (This operation
    can be memoty intensive, since it stores the entire quantum state.)
    """

    def __init__(self,
                 key: Hashable) -> None:
        super().__init__(key, lambda x: x)

# fin
