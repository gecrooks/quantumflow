# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumcompiler

Gate Modules
############

Larger unitary computational modules that
can be broken up into standard gates.

Danger: Gate's may have many qubits, so explicitly creating
the gate tensor may consume huge amounts of memory. Beware.

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

.. autoclass:: ControlGate
    :members:

.. autoclass:: QFTGate
    :members:

.. autoclass:: InvQFTGate
    :members:
"""

from typing import Iterable, Iterator, List, Mapping, Union

import networkx as nx
import numpy as np
from networkx.algorithms.approximation.steinertree import steiner_tree
from sympy.combinatorics import Permutation

from . import tensors, utils, var
from .circuits import Circuit
from .gates import unitary_from_hamiltonian
from .ops import Gate, Operation
from .paulialgebra import Pauli, pauli_commuting_sets, sX, sY, sZ
from .qubits import Qubits
from .states import Density, State
from .stdgates import CZ, CNot, CZPow
from .stdgates import H as _H  # NB: Workaround for name conflict with Gate.H
from .stdgates import I, Swap, X, XPow, Y, YPow, Z, ZPow
from .tensors import QubitTensor
from .var import Variable

__all__ = (
    "IdentityGate",
    "PauliGate",
    "MultiSwapGate",
    "ReversalGate",
    "CircularShiftGate",
    "ControlGate",
    "QFTGate",
    "InvQFTGate",
)


class IdentityGate(Gate):
    r"""
    The multi-qubit identity gate.
    """
    cv_interchangeable = True
    cv_hermitian = True
    cv_tensor_structure = "identity"

    @property
    def hamiltonian(self) -> Pauli:
        return Pauli.zero()

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor(np.eye(2 ** self.qubit_nb))

    def __pow__(self, t: Variable) -> "IdentityGate":
        return self

    @property
    def H(self) -> "IdentityGate":
        return self

    def decompose(self) -> Iterator[I]:  # noqa: E741
        for q in self.qubits:
            yield I(q)


# end class IdentityGate


# TESTME : axes
# DOCME: axes
# TODO: diagrams
# TODO: Decompose
# ⊖ ⊕ ⊘ ⊗ ● ○
class ControlGate(Gate):
    """A controlled unitary gate. Given C control qubits and a
    gate acting on K qubits, return a gate with C+K qubits
    """

    def __init__(self, control_qubits: Qubits, gate: Gate, axes: str = None) -> None:
        control_qubits = tuple(control_qubits)
        qubits = tuple(control_qubits) + tuple(gate.qubits)
        if len(set(qubits)) != len(qubits):
            raise ValueError("Control and gate qubits overlap")

        if axes is None:
            axes = "Z" * len(control_qubits)
        assert len(axes) == len(control_qubits)

        super().__init__(qubits)
        self.control_qubits = qubits
        self.gate = gate
        self.axes = axes

    @property
    def hamiltonian(self) -> Pauli:
        ctrlham = {
            "X": (1 - sX(0)) / 2,
            "x": sX(0) / 2,
            "Y": (1 - sY(0)) / 2,
            "y": sY(0) / 2,
            "Z": (1 - sZ(0)) / 2,
            "z": sZ(0) / 2,
        }

        ham = self.gate.hamiltonian
        for q, axis in zip(self.control_qubits, self.axes):
            ham *= ctrlham[axis].on(q)

        return ham

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return unitary_from_hamiltonian(self.hamiltonian, self.qubits).tensor


# end class ControlGate


class MultiSwapGate(Gate):
    """A permutation of qubits. A generalized multi-qubit Swap."""

    cv_tensor_structure = "permutation"
    cv_hermitian = True

    def __init__(self, qubits_in: Qubits, qubits_out: Qubits) -> None:
        if set(qubits_in) != set(qubits_out):
            raise ValueError("Incompatible sets of qubits")

        self.qubits_out = tuple(qubits_out)
        self.qubits_in = tuple(qubits_in)
        super().__init__(qubits=qubits_in)

    @classmethod
    def from_gates(cls, gates: Iterable[Operation]) -> "MultiSwapGate":
        """Create a qubit permutation from a circuit of swap gates"""
        qubits_in = Circuit(gates).qubits
        N = len(qubits_in)

        circ: List[Gate] = []
        for gate in gates:
            if isinstance(gate, Swap):
                circ.append(gate)
            elif isinstance(gate, MultiSwapGate):
                circ.extend(gate.decompose())
            elif isinstance(gate, I) or isinstance(gate, IdentityGate):
                continue
            else:
                raise ValueError("Swap gate must be built from swap gates")

        perm = list(range(N))
        for elem in circ:
            q0, q1 = elem.qubits
            i0 = qubits_in.index(q0)
            i1 = qubits_in.index(q1)
            perm[i1], perm[i0] = perm[i0], perm[i1]

        qubits_out = [qubits_in[p] for p in perm]
        return cls(qubits_in, qubits_out)

    @property
    def H(self) -> "MultiSwapGate":
        return MultiSwapGate(self.qubits_out, self.qubits_in)

    def run(self, ket: State) -> State:
        qubits = ket.qubits
        N = ket.qubit_nb

        perm = list(range(N))
        for q0, q1 in zip(self.qubits_in, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        tensor = tensors.permute(ket.tensor, perm)

        return State(tensor, qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        qubits = rho.qubits
        N = rho.qubit_nb

        perm = list(range(N))
        for q0, q1 in zip(self.qubits_in, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)
        perm.extend([idx + N for idx in perm])

        tensor = tensors.permute(rho.tensor, perm)

        return Density(tensor, qubits, rho.memory)

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        N = self.qubit_nb
        qubits = self.qubits

        perm = list(range(2 * N))
        for q0, q1 in zip(qubits, self.qubits_out):
            perm[qubits.index(q0)] = qubits.index(q1)

        U = np.eye(2 ** N)
        U = np.reshape(U, [2] * 2 * N)
        U = np.transpose(U, perm)
        return tensors.asqutensor(U)

    def decompose(self) -> Iterator[Swap]:
        """
        Returns a swap network for this permutation, assuming all-to-all
        connectivity.
        """
        qubits = self.qubits

        perm = [self.qubits.index(q) for q in self.qubits_out]
        for idx0, idx1 in Permutation(perm).transpositions():
            yield Swap(qubits[idx0], qubits[idx1])


# end class MultiSwapGate


class ReversalGate(MultiSwapGate):
    """A qubit permutation that reverses the order of qubits"""

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits, tuple(reversed(qubits)))

    def decompose(self) -> Iterator[Swap]:
        qubits = self.qubits
        for idx in range(self.qubit_nb // 2):
            yield Swap(qubits[idx], qubits[-idx - 1])


# end class ReversalGate


# DOCME Makes a circular buffer
class CircularShiftGate(MultiSwapGate):
    def __init__(self, qubits: Qubits, shift: int = 1) -> None:
        qubits_in = tuple(qubits)
        nshift = shift % len(qubits)
        qubits_out = qubits_in[nshift:] + qubits_in[:nshift]

        super().__init__(qubits_in, qubits_out)
        self.shift = shift


# end class CircularShiftGate


class QFTGate(Gate):
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
    def H(self) -> "InvQFTGate":
        return InvQFTGate(self.qubits)

    def decompose(self) -> Iterator[Union[_H, CZPow, Swap]]:
        qubits = self.qubits
        N = len(qubits)
        for n0 in range(N):
            q0 = qubits[n0]
            yield _H(q0)
            for n1 in range(n0 + 1, N):
                q1 = qubits[n1]
                yield CZ(q1, q0) ** (1 / 2 ** (n1 - n0))
        yield from ReversalGate(qubits).decompose()

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return Circuit(self.decompose()).asgate().on(*self.qubits).tensor


# end class QFTGate


class InvQFTGate(Gate):
    """The inverse Quantum Fourier Transform"""

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits=qubits)

    @property
    def H(self) -> "QFTGate":
        return QFTGate(self.qubits)

    def decompose(self) -> Iterator[Union[_H, CZPow, Swap]]:
        gates = list(QFTGate(self.qubits).decompose())
        yield from (gate.H for gate in gates[::-1])

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return Circuit(self.decompose()).asgate().on(*self.qubits).tensor


# end class InvQFTGate


class PauliGate(Gate):
    """
    A Gate corresponding to the exponential of the Pauli algebra element,
    i.e. exp[-1.0j * alpha * element]
    """

    # Kudos: GEC (2019).

    def __init__(self, element: Pauli, alpha: float) -> None:

        super().__init__(qubits=element.qubits)
        self.element = element
        self.alpha = alpha

    def __str__(self) -> str:
        return f"PauliGate({self.element}, {self.alpha}) {self.qubits}"

    @property
    def H(self) -> "PauliGate":
        return self ** -1

    def __pow__(self, t: Variable) -> "PauliGate":
        return PauliGate(self.element, self.alpha * t)

    @property
    def hamiltonian(self) -> "Pauli":
        return self.alpha * self.element

    def resolve(self, subs: Mapping[str, float]) -> "PauliGate":
        if var.is_symbolic(self.alpha):
            alpha = var.asfloat(self.alpha, subs)
            return PauliGate(self.element, alpha)
        return self

    # TODO: Move main logic to pauli algebra?
    def decompose(
        self, topology: nx.Graph = None
    ) -> Iterator[Union[CNot, XPow, YPow, ZPow]]:
        """
        Returns a Circuit corresponding to the exponential of
        the Pauli algebra element object, i.e. exp[-1.0j * alpha * element]

        If a qubit topology is provided then the returned circuit will
        respect the qubit connectivity, adding swaps as necessary.
        """
        # Kudos: Adapted from pyquil. The topological network is novel.

        circ = Circuit()
        element = self.element
        alpha = self.alpha

        if element.is_identity() or element.is_zero():
            return circ  # pragma: no cover  # TESTME

        # Check that all terms commute
        groups = pauli_commuting_sets(element)
        if len(groups) != 1:
            raise ValueError("Pauli terms do not all commute")

        for qbs, ops, coeff in element:
            if not np.isclose(complex(coeff).imag, 0.0):
                raise ValueError("Pauli term coefficients must be real")
            theta = complex(coeff).real * alpha

            if len(ops) == 0:
                continue

            # TODO: 1-qubit terms special case

            active_qubits = set()
            change_to_z_basis = Circuit()
            for qubit, op in zip(qbs, ops):
                active_qubits.add(qubit)
                if op == "X":
                    change_to_z_basis += Y(qubit) ** -0.5
                elif op == "Y":
                    change_to_z_basis += X(qubit) ** 0.5

            if topology is not None:
                if not nx.is_directed(topology) or not nx.is_arborescence(topology):
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
                    cnot_seq += Swap(q0, q1)
                    active_qubits.add(q1)
                else:
                    cnot_seq += CNot(q0, q1)

            circ += change_to_z_basis
            circ += cnot_seq
            circ += Z(order[-1]) ** (2 * theta / np.pi)
            circ += cnot_seq.H
            circ += change_to_z_basis.H
        # end term loop

        yield from circ  # type: ignore

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return Circuit(self.decompose()).asgate().on(*self.qubits).tensor


# end class PauliGate


# Fin
