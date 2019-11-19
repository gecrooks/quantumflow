
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
========================
Miscellaneous Operations
========================


.. currentmodule:: quantumflow

Various standard operations on quantum states, which arn't gates,
channels, circuits, or DAGCircuit's.

.. autofunction :: dagger
.. autoclass :: Moment

.. autoclass :: Measure
.. autoclass :: Reset
.. autoclass :: Initialize
.. autoclass :: Barrier
.. autoclass :: Projection
.. autoclass :: PermuteQubits
.. autoclass :: ReverseQubits
.. autoclass :: RotateQubits
.. autoclass :: Store
.. autoclass :: If
.. autoclass :: Display
.. autoclass :: StateDisplay
.. autoclass :: ProbabilityDisplay
.. autoclass :: DensityDisplay

"""

from typing import Sequence, Hashable, Any, Callable, Iterable, Union
import textwrap

import numpy as np

from sympy.combinatorics import Permutation

from .qubits import Qubit, Qubits, asarray
from .states import State, Density
from .ops import Operation, Gate, Channel
from .gates import P0, P1
from .gates import SWAP, I, IDEN
from .circuits import Circuit
from . import backend as bk


__all__ = ['dagger', 'Moment', 'Measure', 'Reset', 'Initialize', 'Barrier',
           'Store',
           'If', 'Projection',
           'PermuteQubits',  'ReverseQubits', 'RotateQubits',
           'Display', 'StateDisplay', 'ProbabilityDisplay', 'DensityDisplay']


def dagger(elem: Operation) -> Operation:
    """Return the complex conjugate of the Operation"""
    return elem.H


class Moment(Sequence, Operation):
    """
    Represents a collection of Operations that operate on disjoint qubits,
    so that they may be applied at the same moment of time.
    """
    def __init__(self, elements: Iterable[Operation]) -> None:
        circ = Circuit(Circuit(elements).flat())
        qbs = list(q for elem in circ for q in elem.qubits)
        if len(qbs) != len(set(qbs)):
            raise ValueError('Qubits of operations within Moments '
                             'must be disjoint.')

        self._qubits = tuple(qbs)
        self._circ = circ

    def __getitem__(self, key: Union[int, slice]) -> Any:
        return self._circ[key]

    def __len__(self) -> int:
        return self._circ.__len__()

    def run(self, ket: State = None) -> State:
        return self._circ.run(ket)

    def evolve(self, rho: Density = None) -> Density:
        return self._circ.evolve(rho)

    def asgate(self) -> 'Gate':
        return self._circ.asgate()

    def aschannel(self) -> 'Channel':
        return self._circ.aschannel()

    @property
    def H(self) -> 'Moment':
        return Moment(self._circ.H)

    # TESTME
    def __str__(self) -> str:
        circ_str = '\n'.join([str(elem) for elem in self])
        circ_str = textwrap.indent(circ_str, '    ')
        return '\n'.join([self.name, circ_str])


class Measure(Operation):
    """Measure a quantum bit and copy result to a classical bit"""
    def __init__(self, qubit: Qubit, cbit: Hashable = None) -> None:
        self.qubit = qubit
        self.cbit = cbit

    def __str__(self) -> str:
        if self.cbit is not None:
            return f'{self.name.upper()} {self.qubit} {self.cbit}'
        return f'{self.name.upper()} {self.qubit}'

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


# TODO: Should perhaps be 1-qubit gate like P0 and P1?
# Having no qubits specified screws up visualization
# and dagc
class Reset(Operation):
    r"""An operation that resets qubits to zero irrespective of the
    initial state.
    """
    def __init__(self, *qubits: Qubit) -> None:
        if not qubits:
            qubits = ()
        self._qubits = tuple(qubits)

        self._gate = Gate(tensor=[[1, 1], [0, 0]])

    def run(self, ket: State) -> State:
        if self.qubits:
            qubits = self.qubits
        else:
            qubits = ket.qubits

        for q in qubits:
            gate = self._gate.relabel([q])
            ket = gate.run(ket)
        ket = ket.normalize()
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


class Initialize(Operation):
    """ An operation that initilizes the quantum state"""
    def __init__(self, ket: State):
        self._ket = ket
        self._qubits = ket.qubits

    @property
    def tensor(self) -> bk.BKTensor:
        return self._ket.tensor

    def run(self, ket: State) -> State:
        return self._ket.permute(ket.qubits)

    def evolve(self, rho: Density) -> Density:
        return self._ket.permute(rho.qubits).asdensity()

    # TODO: aschannel? __str___?


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


class PermuteQubits(Operation):
    """A permutation of qubits. A generalized multi-qubit SWAP."""
    def __init__(self, qubits_in: Qubits, qubits_out: Qubits) -> None:
        if set(qubits_in) != set(qubits_out):
            raise ValueError("Incompatible sets of qubits")

        self.qubits_out = tuple(qubits_out)
        self.qubits_in = tuple(qubits_in)

    @classmethod
    def from_circuit(cls, circ: Circuit) -> 'PermuteQubits':
        """Create a qubit pertumtation from a circuit of swap gates"""
        qubits_in = circ.qubits
        N = circ.qubit_nb
        perm = list(range(N))
        for elem in circ:
            if isinstance(elem, I) or isinstance(elem, IDEN):
                continue
            assert isinstance(elem, SWAP)  # FIXME
            q0, q1 = elem.qubits
            i0 = qubits_in.index(q0)
            i1 = qubits_in.index(q1)
            perm[i1], perm[i0] = perm[i0], perm[i1]
            # TODO: Should also accept PermuteQubits

        qubits_out = [qubits_in[p] for p in perm]
        return cls(qubits_in, qubits_out)

    @property
    def qubits(self) -> Qubits:
        return self.qubits_in

    @property
    def H(self) -> Operation:
        return PermuteQubits(self.qubits_out, self.qubits_in)

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


class ReverseQubits(PermuteQubits):
    """A qubit permutation that reverses the order of qubits"""
    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits, tuple(reversed(qubits)))

    def ascircuit(self) -> Circuit:
        circ = Circuit()
        qubits = self.qubits
        for idx in range(self.qubit_nb//2):
            circ += SWAP(qubits[idx], qubits[-idx-1])
        return circ


class RotateQubits(PermuteQubits):
    def __init__(self, qubits: Qubits, shift: int = 1) -> None:
        qubits_in = tuple(qubits)
        nshift = shift % len(qubits)
        qubits_out = qubits_in[nshift:] + qubits_in[:nshift]

        super().__init__(qubits_in, qubits_out)
        self.shift = shift


# FIXME: Conflicts with Store in xforest?
class Store(Operation):
    """Store a value in the classical memory of the state.
    """
    def __init__(self,
                 key: Hashable,
                 value: Any,
                 qubits: Qubits = ()) -> None:
        super().__init__(qubits=qubits)
        self.key = key
        self.value = value

    def run(self, ket: State) -> State:
        return ket.store({self.key: self.value})

    def evolve(self, rho: Density) -> Density:
        return rho.store({self.key: self.value})


class If(Operation):
    """
    Look up key in classical memory, and apply the given
    quantum operation only if the truth value matches.
    """
    def __init__(self, elem: Operation,
                 key: Hashable,
                 value: bool = True) -> None:
        super().__init__(qubits=elem.qubits)
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


class Display(Operation):
    """A Display is an operation that extracts information from the
    quantum state and stores it in classical memory, without performing
    any effect on the qubits.
    """
    # Terminology 'Display' used by Quirk and cirq (cirq/ops/display.py).
    def __init__(self,
                 key: Hashable,
                 action: Callable,
                 qubits: Qubits = ()) -> None:
        super().__init__(qubits=qubits)
        self.key = key
        self.action = action

    def run(self, ket: State) -> State:
        print('runnin')
        return ket.store({self.key: self.action(ket)})

    def evolve(self, rho: Density) -> Density:
        return rho.store({self.key: self.action(rho)})


# TESTME
class StateDisplay(Display):
    """
    Store a copy of the state in the classical memory. (This operation
    can be memory intensive, since it stores the entire quantum state.)
    """

    def __init__(self,
                 key: Hashable,
                 qubits: Qubits = ()) -> None:
        super().__init__(key, lambda x: x, qubits=qubits)


# TESTME DOCME
# FIXME: Act on qubit subspace
class ProbabilityDisplay(Display):
    """
    Store the state probabilities in classical memory.
    """

    def __init__(self,
                 key: Hashable,
                 qubits: Qubits = ()) -> None:
        super().__init__(key, lambda state: state.probabilities(),
                         qubits=qubits)


# TESTME DOCME
class DensityDisplay(Display):
    def __init__(self,
                 key: Hashable,
                 qubits: Qubits) -> None:
        super().__init__(key,
                         lambda state: state.asdensity(self.qubits),
                         qubits=qubits)

# fin
