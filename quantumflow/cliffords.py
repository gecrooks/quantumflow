
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Clifford gates
"""

from typing import Any
import numpy as np
from numpy import sqrt, pi

# from . import backend as bk
from .qubits import Qubit, Qubits
from .ops import Gate
from .gates import S, S_H, V, X, Z, I, H, V_H, Y
from .gates import TX, TY, TZ, RN
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
    RN(pi, 1, 0, 0),
    RN(-0.5 * pi, 1, 0, 0),

    RN(0.5 * pi, 0, 1, 0),
    RN(pi, 0, 1, 0),
    RN(-0.5 * pi, 0, 1, 0),

    RN(0.5 * pi, 0, 0, 1),
    RN(pi, 0, 0, 1),
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
    Circuit(I()),

    Circuit(V()),
    Circuit(X()),
    Circuit(V_H()),

    Circuit(S_H(), V(), S()),
    Circuit(Y()),
    Circuit(S(), V(), S_H()),

    Circuit(S()),
    Circuit(Z()),
    Circuit(S_H()),

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


from typing import Sequence
from .paulialgebra import Pauli
class CliffordGate(Gate):
    # prototype multiqubit Clifford, to replace Clifford
    # qubits optional?
    def __init__(self,
                 stabalizer: Sequence[Pauli],
                 destabalizer: Sequence[Pauli],
                 qubits: Qubits) -> None:
        qubits_sorted = all(qubits[i] < qubits[i+1] for i in range(0, len(qubits)-1))
        if not qubits_sorted:
            qubits, stabalizer, destabalizer \
                = list(zip(*sorted(zip(qubits, stabalizer, destabalizer))))

        super().__init__(qubits)
        self.stabalizer = stabalizer
        self.destabalizer = destabalizer

        # TODO: Check compatible qubits, check legit tableau

    def on(self, qubits):
        assert self.qubit_nb == len(qubits)
        rewire = {q0: q1 for q0, q1 in zip(self.qubits, qubits)}
        x = [x.relabel(rewire) for x in self.stabalizer]
        z = [z.relabel(rewire) for z in self.destabalizer]
        return CliffordGate(x, z, qubits)

    # TODO: Rewire

    def __str__(self):
        sx = ", ".join(str(x) for x in self.stabalizer)
        sz = ", ".join(str(z) for z in self.destabalizer)
        return f"{self.name}([{sx}], [{sz}], {self.qubits})"

    def conjugate(self, pauli):
        from .paulialgebra import sI, Pauli
        for qbs, ops, coeff in pauli:
            out = sI(0) * coeff
            for q, op in zip(qbs, ops):
                if q in self.qubits:
                    idx = self.qubits.index(q)
                    if op == "X":
                        out *= self.stabalizer[idx]
                    elif op == "Z":
                        out *= self.destabalizer[idx]
                    else:  # Y
                        out *= 1j * self.stabalizer[idx] * self.destabalizer[idx]  # FIXME Sign?
                else:
                    out *= Pauli.sigma(q, op)
        out = round_pauli(out)
        return out

    def __matmul__(self, other):
        qubits = sorted(list(set(other.qubits) | set(self.qubits)))
        # Brain hurts. Why this way around!?
        xs = [other.conjugate(self.conjugate(sX(q))) for q in qubits]
        zs = [other.conjugate(self.conjugate(sZ(q))) for q in qubits]
        # zs = [self.conjugate(z) for z in other.z]
        return CliffordGate(xs, zs, qubits)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CliffordGate):
            return NotImplemented
        return self.stabalizer == other.stabalizer \
            and self.destabalizer == other.destabalizer \
            and self.qubits == other.qubits

    # todo: cache
    def __hash__(self) -> int:
        return hash((self.stabalizer, self.destabalizer, self.qubits))

        # set(other.qubits) | set(self.qubits)
        #     -> list
        #     -> sort
        #     -> some_name

    # def deke(self):
    #     circ = []
    #     cliff = self
    #     for n, q in enumerate(self.qubits):
    #         subterm coeff = cliff.x[n]
    #         for 


from .paulialgebra import sX, sZ, sY
def test_CliffordGate():
    cnot01 = CliffordGate([sX(0) * sX(1), sX(1)],
                          [sZ(0), sZ(0) * sZ(1)],
                          [0, 1])
    print(cnot01)

    cnot10 = cnot01.on([1, 0])
    print('CNOT10', cnot10)

    h0 = CliffordGate([sZ(0)], [sX(0)], [0])
    print(h0)
    h1 = h0.on([1])
    print(h1)

    print(h0.conjugate(sX(0)))
    print(h0.conjugate(sY(0)))
    print(h0.conjugate(sY(0)*sX(1)))

    print(cnot01.conjugate(sX(0)))

    # out = cnot01 @ cnot01
    # print(out)

    # out = cnot01 @ cnot10
    # print("DCNOT", out)

    out = cnot10 @ cnot01
    print("DCNOT", out)

    print(cnot01.conjugate(cnot10.conjugate(sX(0))))

    print("HERE", h0@h1)

    sh2 = CliffordGate([-sY(2)], [sZ(2)], [2])

    cnot12 = cnot01.on([1, 2])
    cnot23 = cnot01.on([2, 3])
    h2 = h0.on([2])

    ladder = cnot01 @ cnot12 @ sh2 @ cnot12 @ cnot01
    print("Ladder", ladder)

    ladder = cnot01 @ cnot12 @ sh2 @ sh2@ sh2  @ cnot12 @ cnot01
    print("Ladder", ladder)

    ladder = cnot01 @ cnot12 @ h2  @ cnot12 @ cnot01
    print("H Ladder", ladder)

    ladder = cnot01 @ cnot12 @ cnot23 @ cnot12 @ cnot01
    print("cnot Ladder", ladder)

    ladder = cnot23 @ cnot12 @ cnot01
    print("cnot single 1", ladder)

    cnot02 = cnot01.on([0, 2])
    cnot03 = cnot01.on([0, 3])
    ladder = cnot03 @ cnot02 @ cnot01
    print("cnot single 2", ladder)
    c = ladder.conjugate(sX(0) * sX(1) * sX(2) * sX(3))
    print(c)


def almost_integer(number, atol=1e-08):
    if isinstance(number, complex):
        if not np.isclose(number.imag, 0, atol=atol):
            return False
        number = number.real
    x = np.isclose(round(number)-number, 0, atol=atol)
    return x

def test_almost_integer():
    assert almost_integer(1)
    assert almost_integer(1.0)    
    assert almost_integer(11239847819349871423)
    assert almost_integer(-4)
    assert almost_integer(-4+0j)
    assert almost_integer(1.0000000000000001)
    assert not almost_integer(1.0000001)

from .paulialgebra import pauli_sum

# To pauli_group
def round_pauli(pauli):
    terms = []
    for qbs, ops, coeff in pauli:
        if not almost_integer(coeff):
            raise ValueError()
        coeff = round(coeff.real)
        assert coeff == -1 or coeff == 0 or coeff == 1
        terms.append(Pauli.term(qbs, ops, round(coeff.real)))
    return pauli_sum(*terms)


def clifford_from_gate(gate: Gate):
    from .paulialgebra import pauli_decompose_hermitian
    x = []
    z = []
    qubits = gate.qubits
    for q in qubits:
        m = Circuit(gate, X(q), gate.H).asgate().asoperator()
        p = pauli_decompose_hermitian(m)
        p = round_pauli(p)
        # Test
        x.append(p)

        m = Circuit(gate, Z(q), gate.H).asgate().asoperator()
        p = pauli_decompose_hermitian(m)
        p = round_pauli(p)
        # Test
        z.append(p)

    return CliffordGate(x, z, qubits)

# _tabs = {
#     CNOT: CliffordGate([+ sX(0) * sX(1), + sX(1)],
#                        [+ sZ(0), + sZ(0) * sZ(1)], (0, 1)),
#     }

# def translate_to_clifford(gate):
#     if type(gate) in _tabs:
#         return _tabs[type(gate)].on(gate.qubits)
#     assert False


# def translate_clifford(cliff):
#     circ = []
#     for n, term in enumerate(cliff):
#         qbs, ops, coeff = term
#         for q, op in zip(qbs, ops):
#             if op == "Z":
                




def test_deke():
    from .paulialgebra import pauli_decompose_hermitian
    from .gates import CNOT, ISWAP, T
    gate = ISWAP(0, 1)
    gate = H(0)
    # x0_out = Circuit(gate, X(0), gate.H).asgate().asoperator()
    # x1_out = Circuit(gate, X(1), gate.H).asgate().asoperator()
    # z0_out = Circuit(gate, Z(0), gate.H).asgate().asoperator()
    # z1_out = Circuit(gate, Z(1), gate.H).asgate().asoperator()
    # x0_deke = pauli_decompose_hermitian(x0_out)
    # x1_deke = pauli_decompose_hermitian(x1_out)
    # z0_deke = pauli_decompose_hermitian(z0_out)
    # z1_deke = pauli_decompose_hermitian(z1_out)    
    # print(x0_deke)
    # print(x1_deke)
    # print(z0_deke)
    # print(z1_deke)


    print(clifford_from_gate(gate))


    for gate in _clifford_gates:
        print(clifford_from_gate(gate))



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

    def __matmul__(self, other: 'Gate') -> 'Gate':
        if not isinstance(other, Gate):
            raise NotImplementedError()

        if other.identity:
            return self

        if not isinstance(other, Clifford):
            return super().__matmul__(other)    # pragma: no cover

        index = _clifford_mul(self.index, other.index)
        return Clifford(index, *self.qubits)

    @property
    def H(self) -> 'Clifford':
        # FIXME: Inelegant, doesn't respect phase
        # index = _index_from_gate(_clifford_gates[self.index].H)
        # return Clifford(index, *self.qubits)
        return self.su().H

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


_cliffords = [
    CliffordGate([+sX(0)], [+sZ(0)], (0,)),
    CliffordGate([+sX(0)], [+sY(0)], (0,)),
    CliffordGate([+sX(0)], [-sZ(0)], (0,)),
    CliffordGate([+sX(0)], [-sY(0)], (0,)),
    CliffordGate([+sZ(0)], [-sX(0)], (0,)),
    CliffordGate([-sX(0)], [-sZ(0)], (0,)),
    CliffordGate([-sZ(0)], [+sX(0)], (0,)),
    CliffordGate([-sY(0)], [+sZ(0)], (0,)),
    CliffordGate([-sX(0)], [+sZ(0)], (0,)),
    CliffordGate([+sY(0)], [+sZ(0)], (0,)),
    CliffordGate([+sY(0)], [-sZ(0)], (0,)),
    CliffordGate([+sZ(0)], [+sX(0)], (0,)),
    CliffordGate([-sX(0)], [+sY(0)], (0,)),
    CliffordGate([-sY(0)], [-sZ(0)], (0,)),
    CliffordGate([-sZ(0)], [-sX(0)], (0,)),
    CliffordGate([-sX(0)], [-sY(0)], (0,)),
    CliffordGate([+sZ(0)], [+sY(0)], (0,)),
    CliffordGate([+sY(0)], [+sX(0)], (0,)),
    CliffordGate([-sY(0)], [-sX(0)], (0,)),
    CliffordGate([-sZ(0)], [+sY(0)], (0,)),
    CliffordGate([-sY(0)], [+sX(0)], (0,)),
    CliffordGate([+sZ(0)], [-sY(0)], (0,)),
    CliffordGate([+sY(0)], [-sX(0)], (0,)),
    CliffordGate([-sZ(0)], [-sY(0)], (0,)),
    ]


def decompose_clifford(clifford):
    if clifford.qubit_nb == 1:
        clifford_circ = {cl: circ for cl, circ in zip(_cliffords, _clifford_circuits)}
        c = clifford.on(range(clifford.qubit_nb))
        return clifford_circ[c].on(clifford.qubits)
    assert False

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
