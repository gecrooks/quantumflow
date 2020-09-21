# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Directed Acyclic Graph representations of a Circuit.
"""

import itertools
import textwrap
from typing import Any, Dict, Generator, Iterable, Iterator, List, Tuple

import networkx as nx
import numpy as np
import opt_einsum

from . import utils
from .circuits import Circuit
from .config import CIRCUIT_INDENT
from .ops import Channel, Gate, Operation, Unitary
from .qubits import Qubit, Qubits
from .states import Density, State
from .stdops import Moment

__all__ = ("DAGCircuit",)


# DOCME
class In(Operation):
    def __init__(self, q0: Qubit):
        super().__init__([q0])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, In) and other.qubits == self.qubits

    def __hash__(self) -> int:
        return hash(self.qubits)


# DOCME
class Out(Operation):
    def __init__(self, q0: Qubit):
        super().__init__([q0])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Out) and other.qubits == self.qubits

    def __hash__(self) -> int:
        return hash(self.qubits)


# FIXME: Design flaw!?
# DAGCircuit fails if we try to add multi instances of the same gate
# The problem is that we use gates as nodes in the graph, so they have to be
# unique.
# Either wrap all Operations in some QFNode class before adding to circuit,
# Or make sure all Operations occur only once in circuit.
# Clone copies as necessary.


class DAGCircuit(Operation):
    """A Directed Acyclic Graph representation of a Circuit.

    The circuit is converted to a networkx directed acyclic multi-graph,
    stored in the `graph` attribute.

    There are 3 node types, 'in' nodes representing qubits at the start of a
    circuit; operation nodes; and 'out' nodes for qubits at the
    end of a circuit. Edges are directed from 'in' to 'out' via the Operation
    nodes. Each edge is keyed to a qubit.

    A DAGCircuit is considered a mutable object. But it should not be mutated
    if it is part of another Circuit or other immutable composite Operation.

    DAGCircuit is iterable, yielding all of the operation nodes in
    topological sort order.

    Note: Provisional API
    """

    def __init__(self, elements: Iterable[Operation]) -> None:

        self.graph = nx.MultiDiGraph()
        self._qubits_in: Dict[Qubit, In] = {}
        self._qubits_out: Dict[Qubit, Out] = {}

        for elem in elements:
            self.append(elem)  # type: ignore

    def append(self, elem: Operation) -> None:
        G = self.graph
        G.add_node(elem)
        for qubit in elem.qubits:
            if qubit not in self._qubits_in:
                qin = In(qubit)
                qout = Out(qubit)
                self._qubits_in[qubit] = qin
                self._qubits_out[qubit] = qout
                G.add_edge(qin, qout, key=qubit)
            qout = self._qubits_out[qubit]
            prev = list(G.predecessors(qout))[0]
            G.remove_edge(prev, qout)
            G.add_edge(prev, elem, key=qubit)
            G.add_edge(elem, qout, key=qubit)

    @property
    def qubits(self) -> Qubits:
        qbs = list(self._qubits_in.keys())
        qbs = sorted(qbs)
        return tuple(qbs)

    @property
    def qubit_nb(self) -> int:
        return len(self.qubits)

    def on(self, *qubits: Qubit) -> "DAGCircuit":
        return DAGCircuit(Circuit(self).on(*qubits))

    def rewire(self, labels: Dict[Qubit, Qubit]) -> "DAGCircuit":
        return DAGCircuit(Circuit(self).rewire(labels))

    @property
    def H(self) -> "DAGCircuit":
        return DAGCircuit(Circuit(self).H)

    def run(self, ket: State) -> State:
        for elem in self:
            ket = elem.run(ket)
        return ket

    def evolve(self, rho: Density) -> Density:
        for elem in self:
            rho = elem.evolve(rho)
        return rho

    def asgate(self) -> Gate:
        # Note: Experimental
        # Contract entire tensor network graph using einsum
        # TODO: Do same for run() and evolve() and aschannel()
        # Note: Does not work with CTF backend
        tensors = []
        sublists = []

        for elem in self:
            assert isinstance(elem, Gate)
            tensors.append(elem.tensor)
            sublists.append(self.next_edges(elem) + self.prev_edges(elem))

        tensors_sublists = list(itertools.chain(*zip(tensors, sublists)))

        for qout in self._qubits_out.values():
            outsubs = self.prev_edges(qout)
        for qin in self._qubits_in.values():
            outsubs.extend(self.next_edges(qin))
        tensors_sublists.append(tuple(outsubs))

        tensor = opt_einsum.contract(*tensors_sublists)

        return Unitary(tensor, self.qubits)

    def aschannel(self) -> Channel:
        return Circuit(self).aschannel()

    def depth(self, local: bool = True) -> int:
        """Return the circuit depth.

        Args:
            local:  If True include local one-qubit gates in depth
                calculation. Else return the multi-qubit gate depth.
        """
        G = self.graph
        if not local:

            def remove_local(dagc: DAGCircuit) -> Generator[Operation, None, None]:
                for elem in dagc:
                    if dagc.graph.degree[elem] > 2:
                        yield elem

            G = DAGCircuit(remove_local(self)).graph

        return nx.dag_longest_path_length(G) - 1

    def size(self) -> int:
        """Return the number of operations."""
        return self.graph.order() - 2 * self.qubit_nb

    def component_nb(self) -> int:
        """Return the number of independent components that this
        DAGCircuit can be split into."""
        return nx.number_weakly_connected_components(self.graph)

    def components(self) -> List["DAGCircuit"]:
        """Split DAGCircuit into independent components"""
        G = self.graph
        comps = (G.subgraph(c).copy() for c in nx.weakly_connected_components(G))
        return [DAGCircuit(comp) for comp in comps]

    def moments(self) -> Circuit:
        """Split DAGCircuit into Moments, where the operations within each
        moment operate on different qubits (and therefore commute).

        Returns: A Circuit of Moments
        """
        D = self.depth()
        G = self.graph

        node_depth: Dict[Qubit, int] = {}
        node_height: Dict[Qubit, int] = {}

        for elem in self:
            depth = np.max(
                list(node_depth.get(prev, -1) + 1 for prev in G.predecessors(elem))
            )
            node_depth[elem] = depth

        for elem in reversed(list(self)):
            height = np.min(
                list(node_height.get(next, D) - 1 for next in G.successors(elem))
            )
            node_height[elem] = height

        # Place operations in Moment closest to
        # beginning or end of circuit.
        moments = [Circuit() for _ in range(D)]

        # Iterate nodes in reverse seems to preserve original circuit ordering
        for elem in reversed(list(self)):
            depth = node_depth[elem]
            height = node_height[elem]
            if depth <= D - height - 1:
                moments[depth] += elem
            else:
                moments[height] += elem

        circ = Circuit(Moment(moment) for moment in moments)

        return circ

    @utils.deprecated
    def layers(self) -> Circuit:
        return self.moments()

    def __iter__(self) -> Iterator[Operation]:
        for elem in nx.topological_sort(self.graph):
            # Filter in and out nodes
            if not isinstance(elem, In) and not isinstance(elem, Out):
                yield elem

    # DOCME TESTME
    def next_element(self, elem: Operation, qubit: Qubit = None) -> Operation:
        for _, node, key in self.graph.edges(elem, keys=True):
            if qubit is None or key == qubit:
                return node
        assert False  # Insanity check  # FIXME, raise exception

    # DOCME TESTME
    def prev_element(self, elem: Operation, qubit: Qubit = None) -> Operation:
        for node, _, key in self.graph.in_edges(elem, keys=True):
            if qubit is None or key == qubit:
                return node
        assert False  # Insanity check  # FIXME, raise exception

    def next_edges(self, elem: Operation) -> List[Tuple[Any, Any, Qubit]]:
        qubits = elem.qubits
        N = len(qubits)
        edges = [None] * N
        for edge in self.graph.out_edges(elem, keys=True):
            edges[qubits.index(edge[2])] = edge

        return list(edges)  # type: ignore

    def prev_edges(self, elem: Operation) -> List[Tuple[Any, Any, Qubit]]:
        qubits = elem.qubits
        N = len(qubits)
        edges = [None] * N
        for edge in self.graph.in_edges(elem, keys=True):
            edges[qubits.index(edge[2])] = edge

        return list(edges)  # type: ignore

    def __str__(self) -> str:
        circ_str = "\n".join([str(elem) for elem in self])
        circ_str = textwrap.indent(circ_str, " " * CIRCUIT_INDENT)
        return "\n".join([self.name, circ_str])


# End class DAGCircuit
