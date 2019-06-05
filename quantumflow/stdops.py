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
.. autoclass :: If
.. autoclass :: Projection
"""


from typing import Sequence

import numpy as np

from .cbits import Addr
from .qubits import Qubit, Qubits, asarray, QubitVector
from .states import State, Density
from .ops import Operation, Gate, Channel
from .gates import P0, P1
from . import backend as bk


__all__ = ['dagger', 'Measure', 'Reset', 'Barrier', 'If', 'Projection']


def dagger(elem: Operation) -> Operation:
    """Return the complex conjugate of the Operation"""
    return elem.H


class Measure(Operation):
    """Measure a quantum bit and copy result to a classical bit"""
    def __init__(self, qubit: Qubit, cbit: Addr = None) -> None:
        self.qubit = qubit
        self.cbit = cbit

    def quil(self) -> str:
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
                ket = ket.update({self.cbit: 0})
        else:  # measure one
            ket = P1(self.qubit).run(ket).normalize()
            if self.cbit is not None:
                ket = ket.update({self.cbit: 1})
        return ket

    def evolve(self, rho: Density) -> Density:
        p0 = P0(self.qubit).aschannel()
        p1 = P1(self.qubit).aschannel()

        prob_zero = asarray(p0.evolve(rho).norm())

        # generate random number to 'roll' for measurement
        if np.random.random() < prob_zero:
            rho = p0.evolve(rho).normalize()
            if self.cbit is not None:
                rho = rho.update({self.cbit: 0})
        else:  # measure one
            rho = p1.evolve(rho).normalize()
            if self.cbit is not None:
                rho = rho.update({self.cbit: 1})
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

    def quil(self) -> str:
        if self.qubits:
            return 'RESET ' + ' '.join([str(q) for q in self.qubits])
        return 'RESET'


# DOCME
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

    def quil(self) -> str:
        return self.name.upper() + ' ' + ' '.join(str(q) for q in self.qubits)


# DOCME
class If(Operation):
    def __init__(self, elem: Operation, condition: Addr, value: bool = True) \
            -> None:
        self.element = elem
        self.value = value
        self.condition = condition

    def run(self, ket: State) -> State:
        print(ket.memory[self.condition], self.value)

        if ket.memory[self.condition] == self.value:
            ket = self.element.run(ket)
        return ket

    def evolve(self, rho: Density) -> Density:
        if rho.memory[self.condition] == self.value:
            rho = self.element.evolve(rho)
        return rho


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
