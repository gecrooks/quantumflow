
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Clifford gates
"""


import numpy as np
from numpy import sqrt, pi

# from . import backend as bk
from .qubits import Qubit
from .ops import Gate
from .gates import S, S_H, V, X, Z, I
from .gates import TX, TY, TZ, IDEN, RN
from .circuits import Circuit
# from .variables import variable_is_symbolic
from .measures import gates_close

# from .transform import find_pattern
# from .dagcircuit import DAGCircuit

# from .backends import backend as bk
from .backends import BKTensor

_clifford_gates = (
    I(),

    RN(0.5 * pi, 1., 0., 0.),
    RN(0.5 * pi, 0, 1, 0),
    RN(0.5 * pi, 0, 0, 1),
    RN(pi, 1, 0, 0),
    RN(pi, 0, 1, 0),
    RN(pi, 0, 0, 1),
    RN(-0.5 * pi, 1, 0, 0),
    RN(-0.5 * pi, 0, 1, 0),
    RN(-0.5 * pi, 0, 0, 1),

    RN(pi, 1/sqrt(2), 1/sqrt(2), 0),
    RN(pi, 1/sqrt(2), 0, 1/sqrt(2)),
    RN(pi, 0, 1/sqrt(2), 1/sqrt(2)),
    RN(pi, -1/sqrt(2), 1/sqrt(2), 0),
    RN(pi, 1/sqrt(2), 0, -1/sqrt(2)),
    RN(pi, 0, -1/sqrt(2), 1/sqrt(2)),

    RN(+2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(-2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(+2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(-2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(+2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    RN(-2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    RN(+2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3)),
    RN(-2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3)),
    )
"""
List of all 24 1-qubit Clifford gates. The first gate is the identity.
The rest are given as instances of the generic rotation gate RN
"""


_clifford_circuits = [
    Circuit([I()]),
    Circuit([V()]),
    Circuit([S_H(), V(), S()]),
    Circuit([S()]),
    Circuit([X()]),
    Circuit([S_H(), X(), S()]),
    Circuit([Z()]),
    Circuit([Z(), V(), Z()]),
    Circuit([S(), V(), S_H()]),
    Circuit([S_H()]),
    Circuit([X(), S()]),
    Circuit([S(), V(), S()]),
    Circuit([V(), Z()]),
    Circuit([X(), S_H()]),
    Circuit([S_H(), V(), S_H()]),
    Circuit([Z(), V()]),
    Circuit([V(), S()]),
    Circuit([S(), V(), Z()]),
    Circuit([S_H(), V(), Z()]),
    Circuit([V(), S_H()]),
    Circuit([S(), V()]),
    Circuit([Z(), V(), S()]),
    Circuit([S_H(), V()]),
    Circuit([Z(), V(), S_H()])
    ]


# FIXME: Slow.
def _index_from_gate(gate: Gate) -> int:
    for k, cg in enumerate(_clifford_gates):
        if gates_close(gate.on(0), cg):
            return k
    raise ValueError('Not a Clifford gate')   # pragma: no cover


_clifford_cayley_table = None


def _clifford_mul(index0: int, index1: int) -> int:
    global _clifford_cayley_table
    if _clifford_cayley_table is None:
        _clifford_cayley_table = np.zeros(shape=(24, 24), dtype='int64')
        for i, gate0 in enumerate(_clifford_gates):
            for j, gate1 in enumerate(_clifford_gates):
                res = gate0 @ gate1
                _clifford_cayley_table[i, j] = _index_from_gate(res)

    return _clifford_cayley_table[index0, index1]


class Clifford(Gate):

    text_labels = 'CL{index}'

    def __init__(self, index: int, q0: Qubit = 0) -> None:
        super().__init__(params=dict(index=index), qubits=[q0])
        self.index = index

    @classmethod
    def from_gate(cls, gate: Gate) -> 'Clifford':
        if gate.qubit_nb != 1:
            raise ValueError('Wrong number of qubits')
        index = _index_from_gate(gate)
        return cls(index, *gate.qubits)

    @property
    def tensor(self) -> BKTensor:
        return _clifford_gates[self.index].tensor

    def __matmul__(self, other: 'Gate') -> 'Gate':  # noqa: F811
        if not isinstance(other, Gate):
            raise NotImplementedError()

        if isinstance(other, I) or isinstance(other, IDEN):
            return self

        if not isinstance(other, Clifford):
            return super().__matmul__(other)    # pragma: no cover

        index = _clifford_mul(self.index, other.index)
        return Clifford(index, *self.qubits)

    @property
    def H(self) -> 'Clifford':
        # FIXME: Inelegant
        index = _index_from_gate(_clifford_gates[self.index].H)
        return Clifford(index, *self.qubits)

    # TODO: Rename, deke? decomposition?
    def ascircuit(self) -> Circuit:
        return _clifford_circuits[self.index].on(*self.qubits)


# def merge_cliffords(dagc: DAGCircuit) -> None:
#     for gate0, gate1 in find_pattern(dagc, {Clifford}, {Clifford}):
#         gate = gate1 @ gate0
#         qubit, = gate.qubits

#         prv = dagc.prev_element(gate0)
#         nxt = dagc.next_element(gate1)
#         dagc.graph.add_edge(prv, gate, key=qubit)
#         dagc.graph.add_edge(gate, nxt, key=qubit)

#         dagc.graph.remove_node(gate0)
#         dagc.graph.remove_node(gate1)


# # TESTME
# def progress_cliffords(dagc: DAGCircuit) -> bool:
#     G = dagc.graph
#     again = True
#     changes = False
#     while again:
#         again = False
#         for elem1, elem2 in find_pattern(dagc, {Clifford}, {TX, TY, TZ}):
#             q, = elem2.qubits
#             assert isinstance(elem1, Clifford)
#             sign, gatecls = _from_to[elem1.index, type(elem2)]
#             newgate = gatecls(sign*elem2.params['t'], q)

#             elem0 = dagc.prev_element(elem1, q)
#             elem3 = dagc.next_element(elem2, q)

#             G.remove_edge(elem0, elem1, q)
#             G.remove_edge(elem1, elem2, q)
#             G.remove_edge(elem2, elem3, q)

#             dagc.graph.remove_node(elem2)

#             dagc.graph.add_edge(elem0, newgate, key=q)
#             dagc.graph.add_edge(newgate, elem1, key=q)
#             dagc.graph.add_edge(elem1, elem3, key=q)

#             again = True
#             changes = True
#     return changes


# def retrogress_tx(dagc: DAGCircuit) -> None:
#     """Move TX gates as early in the circuit as possible"""
#     G = dagc.graph
#     again = True
#     while again:
#         again = False
#         for elem1, elem2 in find_pattern(dagc, {XX}, {TX}):
#             q, = elem2.qubits
#             elem0 = dagc.prev_element(elem1, q)
#             elem3 = dagc.next_element(elem2, q)

#             G.remove_edge(elem0, elem1, q)
#             G.remove_edge(elem1, elem2, q)
#             G.remove_edge(elem2, elem3, q)

#             G.add_edge(elem0, elem2, key=q)
#             G.add_edge(elem2, elem1, key=q)
#             G.add_edge(elem1, elem3, key=q)
#             again = True


_from_to = dict()

t = np.exp(1)
for k, cg in enumerate(_clifford_gates):
    for cls0 in TX, TY, TZ:
        for cls1 in TX, TY, TZ:
            for s in +1, -1:
                if gates_close(cls0(t) @ cg, cg @ cls1(s*t)):
                    _from_to[(k, cls0)] = (s, cls1)


# # FIXME: What is this meant to be for?
# def convert_to_cliffords(circ: Circuit) -> Generator[Operation, None, None]:
#     for elem in circ:

#         if not isinstance(elem, TX) \
#                 and not isinstance(elem, TY) \
#                 and not isinstance(elem, TZ):
#             yield elem
#             continue
#         t = elem.params['t']
#         if variable_is_symbolic(t):
#             yield elem
#             continue
#         t %= 2
#         n = int(t*2)
#         t -= n/2
#         if t != 0.0:
#             yield elem.__class__(t, *elem.qubits)
#         if n != 0:
#             Clifford.from_gate(TZ(n/2))
#             yield Clifford.from_gate(elem.__class__(n/2, *elem.qubits))
