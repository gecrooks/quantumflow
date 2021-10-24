# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import heapq
from abc import ABC
from functools import reduce
from itertools import groupby, product
from operator import itemgetter, mul
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import sympy as sym

from .operations import STDGATES, Gate, Operation
from .states import Addr, Qubit, Qubits
from .utils.var import ComplexVariable, almost_zero_variable, is_complex_variable


PauliTerm = Tuple[Tuple[Qubit, ...], str, ComplexVariable]
PauliTerms = Tuple[PauliTerm, ...]


# TODO: subs, test with symbolic variables


class PauliElement(ABC):
    """An element of the Pauli algebra.

    A mixin class. Subclasses are the I, X, Y, and Z gates,
    and Pauli, which represents a general element of the algebra.
    """

    _terms: PauliTerms

    def __add__(self, other: Any) -> "Pauli":
        if is_complex_variable(other):
            other = Pauli.scalar(other)
        if isinstance(other, PauliElement):
            return pauli_sum(self, other)
        return NotImplemented

    def __neg__(self) -> "Pauli":
        return self * -1

    def __mul__(self, other: Any) -> "Pauli":
        if is_complex_variable(other):
            other = Pauli.scalar(other)
        if isinstance(other, PauliElement):
            return pauli_product(self, other)
        return NotImplemented

    def __pos__(self) -> "Pauli":
        return 1 * self

    def __sub__(self, other: Any) -> "Pauli":
        return self + -1 * other

    def __radd__(self, other: Any) -> "Pauli":
        return self.__add__(other)

    def __rmul__(self, other: Any) -> "Pauli":
        return self.__mul__(other)

    def __rsub__(self, other: Any) -> "Pauli":
        return (-1 * self) + other

    def __truediv__(self, other: Any) -> "Pauli":
        return self * (1 / other)


# DOCME
class Pauli(PauliElement, Operation):
    def __init__(self, terms: PauliTerms) -> None:
        the_terms: List[PauliTerm] = []
        qubits: Set[Qubit] = set()

        if len(terms) != 0:  # == 0 zero element
            for qbs, ops, value in terms:
                if not all(op in "IXYZ" for op in ops):
                    raise ValueError("Valid PauliElement operators are I, X, Y, and Z")
                if almost_zero_variable(value):
                    continue
                if len(ops) != 0:
                    qops = sorted(zip(qbs, ops))
                    qops = list(filter(lambda x: x[1] != "I", qops))
                    qbs2, ops2 = zip(*qops) if qops else ((), "")
                    qbs = tuple(qbs2)
                    ops = "".join(ops2)

                the_terms.append((tuple(qbs), ops, value))
                qubits.update(qbs)

        super().__init__(qubits=sorted(list(qubits)))

        self._terms = tuple(the_terms)

    @classmethod
    def term(
        cls, qubits: Qubits, ops: str, value: ComplexVariable = 1.0
    ) -> "Pauli":
        return cls(((qubits, ops, value),))

    @classmethod
    def scalar(cls, value: ComplexVariable) -> "Pauli":
        return cls.term((), "", value)

    def __str__(self) -> str:
        out = []
        for qbs, ops, value in self._terms:
            str_value = str(value)
            if str_value[0] != "-":
                str_value = "+" + str_value

            if len(ops) == 0:  # scalar
                out.append(str_value)
            else:
                if value == 1:
                    out.append("+")
                elif value == -1:
                    out.append("-")
                else:
                    out.append(str_value)

                out.append(" ".join(f"{op}({q})" for q, op in zip(qbs, ops)))

        return " ".join(out)

    def __repr__(self) -> str:
        return self.name + "(" + repr(self._terms) + ")"

    @property
    def H(self) -> "Pauli":
        terms = tuple((qbs, ops, np.conj(value)) for qbs, ops, value in self._terms)
        return Pauli(terms)

    @property
    def operator(self) -> np.ndarray:
        from .gates import Identity

        if self._terms == ():  # zero
            return np.zeros(shape=(1, 1))

        res = []
        for qbs, ops, value in self._terms:
            value = complex(sym.N(value))
            gate: Gate = Identity(self.qubits)
            for q, op in zip(qbs, ops):
                gate = STDGATES[op](q) @ gate
            res.append(gate.operator * value)
        return reduce(lambda x, y: x + y, res)

    def relabel(
        self,
        qubit_map: Dict[Qubit, Qubit],
        addr_map: Dict[Addr, Addr] = None,
    ) -> "Pauli":
        terms = []
        for qbs, ops, value in self._terms:
            new_qbs = [qubit_map[q] for q in qbs]
            terms.append((tuple(new_qbs), ops, value))

        return Pauli(tuple(sorted(terms)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Pauli):
            return NotImplemented
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(self._terms)


# End class Pauli


def pauli_sum(*elements: PauliElement) -> Pauli:
    """Return the sum of elements of the PauliElement algebra"""
    terms = []

    key = itemgetter(0, 1)  # (qbs, ops)
    for term, grp in groupby(
        heapq.merge(*(elem._terms for elem in elements), key=key), key=key
    ):
        value = sum(g[2] for g in grp)
        if not almost_zero_variable(value):
            terms.append((term[0], term[1], value))

    return Pauli(tuple(terms))


_PAULI_PROD = {
    "II": ("I", 1.0),
    "IX": ("X", 1.0),
    "IY": ("Y", 1.0),
    "IZ": ("Z", 1.0),
    "XI": ("X", 1.0),
    "XX": ("I", 1.0),
    "XY": ("Z", 1.0j),
    "XZ": ("Y", -1.0j),
    "YI": ("Y", 1.0),
    "YX": ("Z", -1.0j),
    "YY": ("I", 1.0),
    "YZ": ("X", 1.0j),
    "ZI": ("Z", 1.0),
    "ZX": ("Y", 1.0j),
    "ZY": ("X", -1.0j),
    "ZZ": ("I", 1.0),
}


def pauli_product(*elements: PauliElement) -> Pauli:
    """Return the product of elements of the PauliElement algebra"""
    result_terms = []

    for terms in product(*(elem._terms for elem in elements)):
        value = reduce(mul, [term[2] for term in terms])
        ops = (zip(qbs, ops) for qbs, ops, _ in terms)
        out_qubits = []
        out_ops = []
        key = itemgetter(0)
        for qubit, qops in groupby(heapq.merge(*ops, key=key), key=key):
            res = next(qops)[1]  # Operator: X Y Z
            for op in qops:
                pair = res + op[1]
                res, resvalue = _PAULI_PROD[pair]
                value *= resvalue
            if res != "i":
                out_qubits.append(qubit)
                out_ops.append(res)

        term = Pauli.term(out_qubits, "".join(out_ops), value)
        result_terms.append(term)

    return pauli_sum(*result_terms)


def pauli_decompose(matrix: np.ndarray, qubits: Qubits = None) -> Pauli:
    """Decompose a matrix into an element of the Pauli algebra.

    This works because tensor products of Pauli matrices form an orthonormal
    basis in the linear space of all 2^N×2^N matrices under Hilbert-Schmidt
    inner product.
    """
    from .gates import Identity, Unitary

    if not np.ndim(matrix) == 2:
        raise ValueError("Must be square matrix")

    N = int(np.log2(np.size(matrix))) // 2
    if not 2 ** (2 * N) == np.size(matrix):
        raise ValueError("Matrix dimensions must be power of 2")

    if qubits is None:
        qubits = tuple(range(N))
    else:
        qubits = tuple(qubits)
        if not len(qubits) == N:
            raise ValueError("Wrong number of qubits")

    terms = []
    for ops in product("IXYZ", repeat=N):
        op = Pauli.term(qubits, "".join(ops))
        P: Gate = Identity(qubits)
        if op.qubit_nb:
            P = Unitary(op.operator, op.qubits) @ P

        value = np.real_if_close(np.trace(P.operator @ matrix) / (2 ** N))
        term = Pauli.term(qubits, "".join(ops), value)
        terms.append(term)

    return pauli_sum(*terms)


# fin
