# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumflow

QuantumFlow module for working with the Pauli algebra.

.. autoclass:: Pauli
    :members:

.. autofunction:: sX
.. autofunction:: sY
.. autofunction:: sZ
.. autofunction:: sI

.. autofunction:: pauli_sum
.. autofunction:: pauli_product
.. autofunction:: pauli_pow
.. autofunction:: paulis_commute
.. autofunction:: pauli_commuting_sets
.. autofunction:: paulis_close

"""

# Kudos: Adapted from PyQuil's paulis.py, original written by Nick Rubin

import heapq
from cmath import isclose  # type: ignore
from functools import reduce
from itertools import groupby, product
from numbers import Complex
from operator import itemgetter, mul
from typing import Any, Dict, List, Set, Tuple, cast

import numpy as np
import sympy as sym

from .config import ATOL
from .operations import STDGATES, Gate, Operation
from .states import Addr, Qubit, Qubits, State
from .utils import var

# from .tensors import QubitTensor

# from .utils.var import ComplexVariable, almost_zero_variable, is_complex_variable
# from .utils.var import almost_zero_variable as almost_zero


__all__ = [
    "PauliElement",
    "PauliTerm",
    "Pauli",
    "pauli_sum",
    "pauli_product",
    "pauli_pow",
    "paulis_commute",
    "pauli_commuting_sets",
    "paulis_close",
    "pauli_decompose",
]


PauliTerm = Tuple[Qubits, str, var.ComplexVariable]
PauliTerms = Tuple[PauliTerm, ...]


PAULI_OPS = ["X", "Y", "Z", "I"]

PAULI_PROD = {
    "ZZ": ("I", 1.0),
    "YY": ("I", 1.0),
    "XX": ("I", 1.0),
    "II": ("I", 1.0),
    "XY": ("Z", 1.0j),
    "XZ": ("Y", -1.0j),
    "YX": ("Z", -1.0j),
    "YZ": ("X", 1.0j),
    "ZX": ("Y", 1.0j),
    "ZY": ("X", -1.0j),
    "IX": ("X", 1.0),
    "IY": ("Y", 1.0),
    "IZ": ("Z", 1.0),
    "ZI": ("Z", 1.0),
    "YI": ("Y", 1.0),
    "XI": ("X", 1.0),
}


# 2021 GEC
class PauliElement:
    """An element of the Pauli algebra.

    A mixin class. Subclasses are the I, X, Y, and Z gates,
    and Pauli, which represents a general element of the algebra.
    """

    _terms: PauliTerms

    def __add__(self, other: Any) -> "Pauli":
        if var.is_complex_variable(other):
            other = Pauli.scalar(other)
        if isinstance(other, PauliElement):
            return pauli_sum(self, other)
        return NotImplemented

    def __neg__(self) -> "Pauli":
        return self * -1

    def __mul__(self, other: Any) -> "Pauli":
        if var.is_complex_variable(other):
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


# end class PauliElement


class Pauli(PauliElement, Operation):
    """
    An element of the Pauli algebra.

    An element of the Pauli algebra is a sequence of terms, such as

        Y(1) - 0.5 Z(1) X(2) Y(4)

    where X, Y, Z and I are the 1-qubit Pauli operators.

    """

    # Internally, each term is a tuple of a complex coefficient, and a sequence
    # of single qubit Pauli operators. (The coefficient goes last so that the
    # terms sort on the operators).
    #
    # PauliTerm = Tuple[Tuple[Tuple[Qubit, str], ...], complex]
    #
    # Each Pauli operator consists of a tuple of
    # qubits e.g. (0, 1, 3), a tuple of Pauli operators e.g. ('X', 'Y', 'Z').
    # Qubits and Pauli terms are kept in sorted order. This ensures that a
    # Pauli element has a unique representation, and makes summation and
    # simplification efficient. We use Tuples (and not lists) because they are
    # immutable and hashable.

    def __init__(self, *terms: PauliTerm) -> None:
        the_terms = []
        qubits: Set[Qubit] = set()

        if len(terms) != 0:  # == 0 zero element
            # print(terms)
            for qbs, ops, coeff in terms:
                if not all(op in PAULI_OPS for op in ops):
                    raise ValueError("Valid Pauli operators are I, X, Y, and Z")
                if isinstance(coeff, Complex) and isclose(coeff, 0.0):
                    continue
                if len(ops) != 0:
                    qops = sorted(zip(qbs, ops))
                    qops = list(filter(lambda x: x[1] != "I", qops))
                    qbs, ops = zip(*qops) if qops else ((), "")  # type: ignore

                the_terms.append((tuple(qbs), "".join(ops), coeff))
                qubits.update(qbs)

        qbs = sorted(list(qubits))
        super().__init__(qbs)

        self._terms = tuple(the_terms)

    # Rename coeff?
    @classmethod
    def term(
        cls, qubits: Qubits, ops: str, coefficient: var.ComplexVariable = 1.0
    ) -> "Pauli":
        """
        Create an element of the Pauli algebra from a sequence of qubits
        and operators. Qubits must be unique and sortable
        """
        return cls((qubits, ops, coefficient))

    @classmethod
    def sigma(cls, qubit: Qubit, operator: str, coefficient: complex = 1.0) -> "Pauli":
        """Returns a Pauli operator ('I', 'X', 'Y', or 'Z') acting
        on the given qubit"""
        if operator == "I":
            return cls.scalar(coefficient)
        return cls.term([qubit], operator, coefficient)

    @classmethod
    def scalar(cls, coefficient: complex) -> "Pauli":
        """Return a scalar multiple of the Pauli identity element."""
        return cls.term((), "", coefficient)

    @property
    def H(self) -> "Pauli":
        terms = tuple((qbs, ops, np.conj(value)) for qbs, ops, value in self._terms)
        return Pauli(*terms)

    def is_scalar(self) -> bool:
        """Returns true if this object is a scalar multiple of the Pauli
        identity element"""
        if len(self._terms) > 1:
            return False
        if len(self._terms) == 0:
            return True  # Zero element
        if len(self._terms[0][0]) == 0:
            return True
        return False

    @classmethod
    def identity(cls) -> "Pauli":
        """Return the identity element of the Pauli algebra"""
        return cls.scalar(1.0)

    def is_identity(self) -> bool:
        """Returns True if this object is identity Pauli element."""

        if len(self._terms) != 1:
            return False
        if self._terms[0][0] != ():
            return False
        return isclose(self._terms[0][2], 1.0)  # FIXME Variables

    @classmethod
    def zero(cls) -> "Pauli":
        """Return the zero element of the Pauli algebra"""
        return cls()

    def is_zero(self) -> bool:
        """Return True if this object is the zero Pauli element."""
        return len(self._terms) == 0

    def __repr__(self) -> str:
        return "Pauli(" + ", ".join(str(t) for t in self._terms) + ")"

    def __str__(self) -> str:
        out = []
        for qbs, ops, coeff in self._terms:
            str_coeff = str(coeff)
            if str_coeff[0] != "-":
                str_coeff = "+" + str_coeff

            if len(ops) == 0:
                # scalar
                out.append(str_coeff)
            else:
                if coeff == 1:
                    out.append("+")
                elif coeff == -1:
                    out.append("-")
                else:
                    out.append(str_coeff)

                out.append(" ".join(f"{op}({q})" for q, op in zip(qbs, ops)))

        return " ".join(out)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Pauli):
            return NotImplemented
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(self._terms)

    def __pow__(self, exponent: int) -> "Pauli":
        return pauli_pow(self, exponent)

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

    # DEPRECATE
    def asoperator(self, qubits: Qubits = None) -> np.ndarray:
        # DOCME: Use of qubits argument here.

        # Late import to prevent circular imports
        from .gates import Identity as IdentityGate
        from .operations import STDGATES as NAMED_GATES

        qubits = self.qubits if qubits is None else qubits
        if self.is_zero():
            N = len(qubits)
            return np.zeros(shape=(2 ** N, 2 ** N))

        res = []
        for qbs, ops, coeff in self._terms:
            if var.is_symbolic(coeff):
                coeff = complex(coeff)
            gate = IdentityGate(qubits)
            for q, op in zip(qbs, ops):
                gate = NAMED_GATES[op](q) @ gate  # type: ignore
            res.append(gate.operator * coeff)
        return cast(np.ndarray, sum(res))

    # TESTME
    def _run_state(self, ket: State) -> State:
        from .operations import STDGATES as NAMED_GATES

        resultants = []
        for qbs, ops, coeff in self._terms:
            res = State(ket.tensor * coeff, ket.qubits)
            for q, op in zip(qbs, ops):
                res = NAMED_GATES[op](q).run(res)  # type: ignore
            resultants.append(res.tensor)

        out = State(sum(resultants), ket.qubits)
        return out

    def relabel(
        self,
        qubit_map: Dict[Qubit, Qubit],
        addr_map: Dict[Addr, Addr] = None,
    ) -> "Pauli":
        terms = []
        for qbs, ops, value in self._terms:
            new_qbs = [qubit_map[q] for q in qbs]
            terms.append((tuple(new_qbs), ops, value))

        return Pauli(*sorted(terms))


# End class Pauli


def pauli_sum(*elements: PauliElement) -> Pauli:
    """Return the sum of elements of the Pauli algebra"""
    terms = []

    key = itemgetter(0, 1)  # (qbs, str)
    for term, grp in groupby(
        heapq.merge(*(elem._terms for elem in elements), key=key), key=key
    ):
        coeff = sum(g[2] for g in grp)
        if not var.almost_zero(coeff):
            terms.append((term[0], term[1], coeff))

    return Pauli(*terms)


def pauli_product(*elements: PauliElement) -> Pauli:
    """Return the product of elements of the Pauli algebra"""
    result_terms = []

    for terms in product(*(elem._terms for elem in elements)):
        coeff = reduce(mul, [term[2] for term in terms])
        ops = (zip(qbs, ops) for qbs, ops, _ in terms)
        out_qubits = []
        out_ops = []
        key = itemgetter(0)
        for qubit, qops in groupby(heapq.merge(*ops, key=key), key=key):
            res = next(qops)[1]  # Operator: X Y Z
            for op in qops:
                pair = res + op[1]
                res, rescoeff = PAULI_PROD[pair]
                coeff *= rescoeff
            if res != "I":
                out_qubits.append(qubit)
                out_ops.append(res)

        p = Pauli.term(out_qubits, "".join(out_ops), coeff)
        result_terms.append(p)

    return pauli_sum(*result_terms)


def pauli_pow(pauli: PauliElement, exponent: int) -> Pauli:
    """
    Raise an element of the Pauli algebra to a non-negative integer power.
    """

    if not isinstance(exponent, int) or exponent < 0:
        raise ValueError("The exponent must be a non-negative integer.")

    if exponent == 0:
        return Pauli.identity()

    if exponent == 1:
        return Pauli(*pauli._terms)

    # https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    y = Pauli.identity()
    x = pauli
    n = exponent
    while n > 1:
        if n % 2 == 0:  # Even
            x = x * x
            n = n // 2
        else:  # Odd
            y = x * y
            x = x * x
            n = (n - 1) // 2
    return x * y


def paulis_close(
    pauli0: PauliElement, pauli1: PauliElement, atol: float = ATOL
) -> bool:
    """Returns: True if Pauli elements are almost identical."""
    pauli = pauli0 - pauli1
    d = sum(abs(coeff) ** 2 for _, _, coeff in pauli._terms)
    return d <= atol


def paulis_commute(element0: PauliElement, element1: PauliElement) -> bool:
    """
    Return true if the two elements of the Pauli algebra commute.
    i.e. if element0 * element1 == element1 * element0

    Derivation similar to arXiv:1405.5749v2 for the check_commutation step in
    the Raeisi, Wiebe, Sanders algorithm (arXiv:1108.4318, 2011).
    """

    def _coincident_parity(term0: PauliTerm, term1: PauliTerm) -> bool:
        non_similar = 0
        key = itemgetter(0)

        op0 = zip(term0[0], term0[1])
        op1 = zip(term1[0], term1[1])
        for _, qops in groupby(heapq.merge(op0, op1, key=key), key=key):

            listqops = list(qops)
            if len(listqops) == 2 and listqops[0][1] != listqops[1][1]:
                non_similar += 1
        return non_similar % 2 == 0

    for term0, term1 in product(element0._terms, element1._terms):
        if not _coincident_parity(term0, term1):
            return False

    return True


def pauli_commuting_sets(element: PauliElement) -> Tuple[Pauli, ...]:
    """Gather the terms of a Pauli polynomial into commuting sets.

    Uses the algorithm defined in (Raeisi, Wiebe, Sanders,
    arXiv:1108.4318, 2011) to find commuting sets. Except uses commutation
    check from arXiv:1405.5749v2
    """
    if len(element._terms) < 2:
        return (Pauli(*element._terms),)

    groups: List[Pauli] = []

    for term in element._terms:
        pterm = Pauli.term(*term)

        assigned = False
        for i, grp in enumerate(groups):
            if paulis_commute(grp, pterm):
                groups[i] = grp + pterm
                assigned = True
                break
        if not assigned:
            groups.append(pterm)

    return tuple(groups)


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
