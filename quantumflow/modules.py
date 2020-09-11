
# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumflow

Gate Modules
############

MultiGates larger unitary computational modules that
can be broken up into standard gates.

.. autoclass:: MultiGate
    :members:

.. autoclass:: IdentityGate
    :members:

.. autoclass:: PauliGate
    :members:

.. autoclass:: MultiSwapGate
    :members:

.. autoclass:: ReversalGate
    :members:

.. autoclass:: CircularShiftGate
    :members:

"""

# TODO: Should change ascircuit to a typed decomposition?

# TODO: State prep.
# TODO: interleave


from abc import abstractmethod
from typing import Iterator

import numpy as np
from math import pi  # FIXME
from cmath import isclose  # type: ignore       # FIXME


from sympy.combinatorics import Permutation
import networkx as nx
from networkx.algorithms.approximation.steinertree import steiner_tree

from .qubits import Qubits
from .states import State, Density
from .ops import Gate
from .circuits import Circuit
from .utils import cached_property
from .variables import Variable
from .paulialgebra import Pauli
from .paulialgebra import pauli_commuting_sets
from .gates import H, SWAP, CNOT, CZ, Y, Z, X, I
# from .gates import TX, TY, TZ, CCNOT, ZZ

from .backends import get_backend, BKTensor
bk = get_backend()

# TODO: __all__
# TODO: Interleave gate

from .variables import variable_is_symbolic

class MultiGate(Gate):
    """
    Abstract base class for multi-qubit unitary operations.
    """

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits=qubits)

    def run(self, ket: State) -> State:
        return self.ascircuit().run(ket)

    def evolve(self, rho: Density) -> Density:
        return self.ascircuit().evolve(rho)

    @abstractmethod
    def ascircuit(self) -> Circuit:
        """Decompose this multi-qubit operation into a circuit of standard gates,
        each of which acts on only a few qubits."""
        raise NotImplementedError()

    @property
    def tensor(self) -> BKTensor:
        """Danger: CompoundGates may have many qubits, so explicitly creating
        the gate tensor may consume huge amounts of memory. Beware."""
        return self.ascircuit().asgate().tensor


class IdentityGate(MultiGate):
    r"""
    The multi-qubit identity gate.
    """
    interchangeable = True
    hermitian = True

    tensor_structure = 'identity'
    identity = True

    @property
    def hamiltonian(self) -> Pauli:
        return Pauli.zero()

    @cached_property
    def tensor(self) -> BKTensor:
        return bk.astensorproduct(np.eye(2 ** self.qubit_nb))

    def __pow__(self, t: Variable) -> 'IdentityGate':
        return self

    def run(self, ket: State) -> State:
        return ket

    def evolve(self, rho: Density) -> Density:
        return rho

    def ascircuit(self) -> Circuit:
        return Circuit(I(q) for q in self.qubits)

# end class IdentityGate


class MultiSwapGate(Gate):
    """A permutation of qubits. A generalized multi-qubit SWAP."""
    def __init__(self, qubits_in: Qubits, qubits_out: Qubits) -> None:
        if set(qubits_in) != set(qubits_out):
            raise ValueError("Incompatible sets of qubits")

        self.qubits_out = tuple(qubits_out)
        self.qubits_in = tuple(qubits_in)
        super().__init__(qubits=qubits_in)

    # FIXME: Instead of circuit, take iterable of gates from_gates
    @classmethod
    def from_gates(cls, gates: Iterator[Gate]) -> 'MultiSwapGate':
        """Create a qubit permutation from a circuit of swap gates"""
        circ = Circuit(gates)
        qubits_in = circ.qubits
        N = circ.qubit_nb
        perm = list(range(N))
        for elem in circ:
            if isinstance(elem, I) or isinstance(elem, IdentityGate):
                continue
            assert isinstance(elem, SWAP)  # FIXME
            q0, q1 = elem.qubits
            i0 = qubits_in.index(q0)
            i1 = qubits_in.index(q1)
            perm[i1], perm[i0] = perm[i0], perm[i1]
            # TODO: Should also accept PermutationGate!

        qubits_out = [qubits_in[p] for p in perm]
        return cls(qubits_in, qubits_out)

    @property
    def qubits(self) -> Qubits:
        return self.qubits_in

    @property
    def H(self) -> 'MultiSwapGate':
        return MultiSwapGate(self.qubits_out, self.qubits_in)

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

    @cached_property
    def tensor(self) -> BKTensor:
        N = self.qubit_nb
        qubits = self.qubits

        perm = list(range(2*N))
        for q0, q1 in zip(qubits, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        U = np.eye(2**N)
        U = np.reshape(U, [2]*2*N)
        U = np.transpose(U, perm)
        return bk.astensorproduct(U)

    def ascircuit(self) -> Circuit:
        """
        Returns a swap network for this permutation, assuming all-to-all
        connectivity.
        """
        circ = Circuit()
        qubits = self.qubits

        perm = [self.qubits.index(q) for q in self.qubits_out]
        for idx0, idx1 in (Permutation(perm).transpositions()):
            circ += SWAP(qubits[idx0], qubits[idx1])

        return circ

# end class MultiSwapGate


class ReversalGate(MultiSwapGate):
    """A qubit permutation that reverses the order of qubits"""
    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits, tuple(reversed(qubits)))

    def ascircuit(self) -> Circuit:
        circ = Circuit()
        qubits = self.qubits
        for idx in range(self.qubit_nb//2):
            circ += SWAP(qubits[idx], qubits[-idx-1])
        return circ

# end class ReversalGate


# DOCME
# Makes a circular buffer
class CircularShiftGate(MultiSwapGate):
    def __init__(self, qubits: Qubits, shift: int = 1) -> None:
        qubits_in = tuple(qubits)
        nshift = shift % len(qubits)
        qubits_out = qubits_in[nshift:] + qubits_in[:nshift]

        super().__init__(qubits_in, qubits_out)
        self.shift = shift

# end class CircularShiftGate


class QFTGate(MultiGate):
    """The Quantum Fourier Transform circuit.

    For 3-qubits
    ::
        0: ───H───Z^1/2───Z^1/4───────────────────x───
                  │       │                       │
        1: ───────●───────┼───────H───Z^1/2───────┼───
                          │           │           │
        2: ───────────────●───────────●───────H───x───
    """
    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits=qubits)

    @property
    def H(self) -> 'InvQFTGate':
        return InvQFTGate(self.qubits)

    def ascircuit(self) -> Circuit:
        qubits = self.qubits
        N = len(qubits)
        circ = Circuit()
        for n0 in range(N):
            q0 = qubits[n0]
            circ += H(q0)
            for n1 in range(n0+1, N):
                q1 = qubits[n1]
                circ += CZ(q1, q0) ** (1/2 ** (n1-n0))
        circ.extend(ReversalGate(qubits).ascircuit())
        return circ
# end class QFTGate


class InvQFTGate(MultiGate):
    """The inverse Quantum Fourier Transform"""
    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits=qubits)

    @property
    def H(self) -> 'QFTGate':
        return QFTGate(self.qubits)

    def ascircuit(self) -> Circuit:
        return QFTGate(self.qubits).ascircuit().H
# end class InvQFTGate


class PauliGate(MultiGate):
    """
    A Gate corresponding to the exponential of the Pauli algebra element,
    i.e. exp[-1.0j * alpha * element]
    """
    # Kudos: GEC (2019).

    def __init__(self,
                 element: Pauli,
                 alpha: float) -> None:

        super().__init__(qubits=element.qubits)
        self.element = element
        self.alpha = alpha

    def __str__(self):
        return f"PauliGate({self.element}, {self.alpha}) {self.qubits}"


    # fixme docme testme
    def resolve(self, resolver) -> 'Circuit':
        import sympy
        if variable_is_symbolic(self.alpha):
            alpha = float(sympy.N(self.alpha, subs=resolver))
            return PauliGate(self.element, alpha)
        return self

    def ascircuit(self, topology: nx.Graph = None) -> Circuit:
        """
        Returns a Circuit corresponding to the exponential of
        the Pauli algebra element object, i.e. exp[-1.0j * alpha * element]

        If a qubit topology is provided then the returned circuit will
        respect the qubit connectivity, adding swaps as necessary.
        """
        # Kudos: Adapted from pyquil. The topological CNOT network is novel.

        circ = Circuit()
        element = self.element
        alpha = self.alpha

        if element.is_identity() or element.is_zero():
            return circ

        # Check that all terms commute
        groups = pauli_commuting_sets(element)
        if len(groups) != 1:
            raise ValueError("Pauli terms do not all commute")

        for term, coeff in element:
            if not term:
                # scalar
                # TODO: Add phase gate? But Z gate below does not respect phase
                continue

            if not isclose(complex(coeff).imag, 0.0):
                raise ValueError("Pauli term coefficients must be real")
            theta = complex(coeff).real * alpha

            # TODO: 1-qubit terms special case

            active_qubits = set()
            change_to_z_basis = Circuit()
            for qubit, op in term:
                active_qubits.add(qubit)
                if op == 'X':
                    change_to_z_basis += Y(qubit)**-0.5
                elif op == 'Y':
                    change_to_z_basis += X(qubit)**0.5

            if topology is not None:
                if (not nx.is_directed(topology)
                        or not nx.is_arborescence(topology)):
                    # An 'arborescence' is a directed tree
                    active_topology = steiner_tree(topology, active_qubits)
                    center = nx.center(active_topology)[0]
                    active_topology = nx.dfs_tree(active_topology, center)
                else:
                    active_topology = topology
            else:
                active_topology = nx.DiGraph()
                nx.add_path(active_topology, reversed(list(active_qubits)))

            cnot_seq = Circuit()
            order = list(reversed(list(nx.topological_sort(active_topology))))
            for q0 in order[:-1]:
                q1 = list(active_topology.pred[q0])[0]
                if q1 not in active_qubits:
                    cnot_seq += SWAP(q0, q1)
                    active_qubits.add(q1)
                else:
                    cnot_seq += CNOT(q0, q1)

            circ += change_to_z_basis
            circ += cnot_seq
            circ += Z(order[-1]) ** (2*theta/pi)
            circ += cnot_seq.H
            circ += change_to_z_basis.H
        # end term loop

        return circ
# end class PauliGate


# Fin
