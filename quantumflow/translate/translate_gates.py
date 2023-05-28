# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import Iterator, Optional, Union

import networkx as nx
import numpy as np
from networkx.algorithms.approximation.steinertree import steiner_tree
from sympy.combinatorics import Permutation

from ..circuits import Circuit
from ..gates import (
    DiagonalGate,
    IdentityGate,
    InvQFTGate,
    MultiplexedRyGate,
    MultiplexedRzGate,
    MultiSwapGate,
    PauliGate,
    QFTGate,
    ReversalGate,
)
from ..paulialgebra import pauli_commuting_sets
from ..stdgates import (
    CZ,
    V_H,
    CNot,
    CZPow,
    H,
    I,
    Rz,
    Swap,
    V,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
)
from .translations import register_translation


@register_translation
def translate_IdentityGate_to_I(gate: IdentityGate) -> Iterator[I]:  # noqa: E741
    """Translate a multi-qubit identity to a sequence of single qubit identity gates"""
    for q in gate.qubits:
        yield I(q)


@register_translation
def translate_MultiSwapGate_to_swap_network(gate: MultiSwapGate) -> Iterator[Swap]:
    """
    Translate a qubit permutation to a swap network, assuming all-to-all
    connectivity.
    """
    qubits = gate.qubits

    perm = [gate.qubits.index(q) for q in gate.qubits_out]
    for idx0, idx1 in Permutation(perm).transpositions():
        yield Swap(qubits[idx0], qubits[idx1])


@register_translation
def translate_ReversalGate_to_swap_network(gate: ReversalGate) -> Iterator[Swap]:
    """
    Translate a qubit reversal to a swap network, assuming all-to-all
    connectivity.
    """
    qubits = gate.qubits
    for idx in range(gate.qubit_nb // 2):
        yield Swap(qubits[idx], qubits[-idx - 1])


@register_translation
def translate_QFTGate(gate: QFTGate) -> Iterator[Union[H, CZPow, Swap]]:
    qubits = gate.qubits
    N = len(qubits)
    for n0 in range(N):
        q0 = qubits[n0]
        yield H(q0)
        for n1 in range(n0 + 1, N):
            q1 = qubits[n1]
            yield CZ(q1, q0) ** (1 / 2 ** (n1 - n0))
    yield from translate_ReversalGate_to_swap_network(ReversalGate(qubits))


@register_translation
def translate_InvQFTGate(gate: InvQFTGate) -> Iterator[Union[H, CZPow, Swap]]:
    gates = list(translate_QFTGate(QFTGate(gate.qubits)))
    yield from (gate.H for gate in gates[::-1])


@register_translation
def translate_PauliGate(
    gate: PauliGate, topology: Optional[nx.Graph] = None
) -> Iterator[Union[CNot, XPow, YPow, ZPow]]:
    """
    Yields a circuit corresponding to the exponential of
    the Pauli algebra element object, i.e. exp[-1.0j * alpha * element]

    If a qubit topology is provided then the returned circuit will
    respect the qubit connectivity, adding swaps as necessary.
    """
    # Kudos: Adapted from pyquil. The topological network is novel.

    circ = Circuit()
    element = gate.element
    alpha = gate.alpha

    if element.is_identity() or element.is_zero():
        return

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


# end translate_PauliGate


@register_translation
def translate_DiagonalGate(gate: DiagonalGate) -> Iterator[Union[Rz, CNot]]:
    diag_phases = gate.params
    N = gate.qubit_nb
    qbs = gate.qubits

    if N == 1:
        yield Rz((diag_phases[0] - diag_phases[1]), qbs[0])
    else:
        phases = []
        angles = []
        for n in range(0, len(diag_phases), 2):
            phases.append((diag_phases[n] + diag_phases[n + 1]) / 2)
            angles.append(-(diag_phases[n + 1] - diag_phases[n]))

        mux = MultiplexedRzGate(angles, qbs[:-1], qbs[-1])
        yield from translate_MultiplexedRzGate(mux)

        diag = DiagonalGate(phases, qbs[:-1])
        yield from translate_DiagonalGate(diag)


@register_translation
def translate_MultiplexedRzGate(gate: MultiplexedRzGate) -> Iterator[Union[Rz, CNot]]:
    thetas = gate.params
    N = gate.qubit_nb
    controls = gate.controls
    target = gate.targets[0]

    if N == 1:
        yield Rz(thetas[0], target)
    elif N == 2:
        yield Rz((thetas[0] + thetas[1]) / 2, target)
        yield CNot(controls[0], target)
        yield Rz((thetas[0] - thetas[1]) / 2, target)
        yield CNot(controls[0], target)
    else:
        # FIXME: Not quite optimal.
        # There's additional cancellation of CNOTs that could happen
        # See: From "Decomposition of Diagonal Hermitian Quantum Gates Using
        # Multiple-Controlled Pauli Z Gates" (2014).

        # Note that we lop off 2 qubits with each recursion.
        # This allows us to cancel two cnots by reordering the second
        # deke.

        # If we lopped off one at a time the deke would look like this:
        # t0 = thetas[0: len(thetas) // 2]
        # t1 = thetas[len(thetas) // 2:]
        # thetas0 = [(a + b) / 2 for a, b in zip(t0, t1)]
        # thetas1 = [(a - b) / 2 for a, b in zip(t0, t1)]
        # yield from MultiplexedRzGate(thetas0, qbs[1:]).decompose()
        # yield CNot(qbs[0], qbs[-1])
        # yield from MultiplexedRzGate(thetas1, qbs[1:]).decompose()
        # yield CNot(qbs[0], qbs[-1])

        M = len(thetas) // 4
        quarters = list(thetas[i : i + M] for i in range(0, len(thetas), M))

        theta0 = [(t0 + t1 + t2 + t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]
        theta1 = [(t0 - t1 + t2 - t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]
        theta2 = [(t0 + t1 - t2 - t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]
        theta3 = [(t0 - t1 - t2 + t3) / 4 for t0, t1, t2, t3 in zip(*quarters)]

        yield from translate_MultiplexedRzGate(
            MultiplexedRzGate(theta0, controls[2:], target)
        )
        yield CNot(controls[1], target)
        yield from translate_MultiplexedRzGate(
            MultiplexedRzGate(theta1, controls[2:], target)
        )
        yield CNot(controls[0], target)
        yield from translate_MultiplexedRzGate(
            MultiplexedRzGate(theta3, controls[2:], target)
        )
        yield CNot(controls[1], target)
        yield from translate_MultiplexedRzGate(
            MultiplexedRzGate(theta2, controls[2:], target)
        )
        yield CNot(controls[0], target)


# end translate_MultiplexedRzGate


@register_translation
def translate_MultiplexedRyGate(
    gate: MultiplexedRyGate,
) -> Iterator[Union[V, V_H, MultiplexedRzGate]]:
    thetas = gate.params
    controls = gate.controls
    target = gate.targets[0]

    yield V(target)
    yield MultiplexedRzGate(thetas, controls, target)
    yield V(target).H
