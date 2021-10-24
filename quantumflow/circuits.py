# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from itertools import chain
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import networkx as nx

from .operations import CompositeOperation, Operation
from .states import Addr, Addrs, Qubit, Qubits

CircuitType = TypeVar("CircuitType", bound="Circuit")
"""Generic type annotations for subtypes of Circuit"""


class Circuit(Sequence, CompositeOperation):

    # DOCME TODO: Adding composite operations is a little weird
    def add(self: CircuitType, other: Iterable[Operation]) -> "CircuitType":
        """Concatenate operations and return new circuit"""
        return type(self)(*chain(self, other))

    def flat(self) -> Iterable[Operation]:
        # DOCME
        for elem in self:
            if isinstance(elem, Circuit):
                yield from elem.flat()
            else:
                yield from elem

    def __add__(self: CircuitType, other: Iterable[Operation]) -> "CircuitType":
        return self.add(other)

    def __contains__(self, key: Any) -> bool:
        return key in self._elements

    def __iadd__(self: CircuitType, other: Iterable[Operation]) -> "CircuitType":
        return self.add(other)

    def __iter__(self) -> Iterator[Operation]:
        yield from self._elements

    @overload
    def __getitem__(self, key: int) -> Operation:
        pass

    @overload  # noqa: F811
    def __getitem__(self: CircuitType, key: slice) -> "CircuitType":
        pass

    def __getitem__(self, key: Union[int, slice]) -> Operation:  # noqa: F811
        if isinstance(key, slice):
            return Circuit(*self._elements[key])
        return self._elements[key]


class Moment(Circuit):
    """
    Represents a collection of Operations that operate on disjoint qubits,
    so that they may be applied at the same moment of time.
    """

    def __init__(
        self, *elements: Operation, qubits: Qubits = None, addrs: Addrs = None
    ) -> None:
        super().__init__(*elements, qubits=qubits, addrs=addrs)

        qbs = list(q for elem in self for q in elem.qubits)
        if len(qbs) != len(set(qbs)):
            raise ValueError("Qubits of operations within Moments must be disjoint.")


class OperationNode:
    def __init__(self, elem: Operation) -> None:
        self.elem = elem

    def __bool__(self) -> bool:
        return self.elem is not None


class DAGCircuit(CompositeOperation):
    def __init__(
        self, *elements: Operation, qubits: Qubits = None, addrs: Addrs = None
    ) -> None:
        super().__init__(*elements, qubits=qubits, addrs=addrs)

        self._graph = nx.MultiDiGraph()

        self._qubits_in: Dict[Qubit, OperationNode] = {}
        self._qubits_out: Dict[Qubit, OperationNode] = {}

        self._addrs_in: Dict[Addr, OperationNode] = {}
        self._addrs_out: Dict[Addr, OperationNode] = {}

        self.add_qubits(self._qubits)
        self.add_addrs(self._addrs)
        self.extend(self._elements)

        self._qubits = None
        self._addrs = None
        self._elements = None

    def __contains__(self, key: Any) -> bool:
        for elem in self:
            if key == elem:
                return True
        return False

    def __iter__(self) -> Iterator[Operation]:
        for node in filter(None, nx.topological_sort(self._graph)):
            yield node.elem

    def __len__(self) -> int:
        return self._graph.order() - 2 * len(self._qubits_in) - 2 * len(self._addrs_in)

    def add_qubits(self, qubits: Qubits) -> None:
        for qubit in qubits:
            if qubit not in self._qubits_in:
                qin = OperationNode(None)
                qout = OperationNode(None)
                self._qubits_in[qubit] = qin
                self._qubits_out[qubit] = qout
                self._graph.add_edge(qin, qout, key=qubit)

    def add_addrs(self, addrs: Addrs) -> None:
        for addr in addrs:  # pragma: no cover  # FIXME
            if addr not in self._addrs_in:
                ain = OperationNode(None)
                aout = OperationNode(None)
                self._addrs_in[addr] = ain
                self._addrs_out[addr] = aout
                self._graph.add_edge(ain, aout, key=addr)

    def append(self, elem: Operation) -> None:
        G = self._graph
        node = OperationNode(elem)
        G.add_node(node)

        self.add_qubits(elem.qubits)
        for qubit in elem.qubits:
            qout = self._qubits_out[qubit]
            prev = list(G.predecessors(qout))[0]
            G.remove_edge(prev, qout)
            G.add_edge(prev, node, key=qubit)
            G.add_edge(node, qout, key=qubit)

        self.add_addrs(elem.addrs)
        for addr in elem.addrs:  # pragma: no cover  # FIXME
            aout = self._addrs_out[addr]
            prev = list(G.predecessors(aout))[0]
            G.remove_edge(prev, aout)
            G.add_edge(prev, node, key=addr)
            G.add_edge(node, aout, key=addr)

    def components(self) -> Tuple["DAGCircuit", ...]:
        G = self._graph
        comps = (G.subgraph(c).copy() for c in nx.weakly_connected_components(G))
        return tuple(
            DAGCircuit(*[c.elem for c in comp if c.elem is not None]) for comp in comps
        )

    def component_nb(self) -> int:
        return nx.number_weakly_connected_components(self._graph)

    def depth(self) -> int:
        return nx.dag_longest_path_length(self._graph) - 1

    def extend(self, elements: Iterable[Operation]) -> None:
        for elem in elements:
            self.append(elem)

    def moments(self) -> Iterator[Moment]:
        G = self._graph

        depth: Dict[Operation, int] = {}
        for node in filter(None, G):
            depth[node] = max([depth.get(n, -1) + 1 for n in G.predecessors(node)])

        height: Dict[Operation, int] = {}
        for node in filter(None, reversed(list(G))):
            height[node] = max([height.get(n, -1) + 1 for n in G.successors(node)])

        moments: List[List[Operation]] = [[] for _ in range(self.depth())]

        for node in filter(None, G):
            if depth[node] < height[node]:
                moments[depth[node]] += node.elem
            else:
                moments[-height[node] - 1] += node.elem

        for moment in moments:
            yield Moment(*moment)

    @property
    def qubits(self) -> Qubits:
        return tuple(self._qubits_in.keys())
