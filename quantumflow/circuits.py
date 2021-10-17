# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import textwrap
from itertools import chain
from typing import Iterable, Iterator, Sequence, TypeVar, Union, overload

from .base import BaseGate, BaseOperation
from .bits import Cbits, Qubits, sorted_bits
from .config import CIRCUIT_INDENT
from .gates import Identity

CircuitType = TypeVar("CircuitType", bound="Circuit")
"""Generic type annotations for subtypes of Circuit"""


class Circuit(Sequence, BaseOperation):
    def __init__(self, *elements: BaseOperation, qubits: Qubits = None):

        elements = tuple(elements)
        qbs: Qubits = sorted_bits([q for elem in elements for q in elem.qubits])

        if qubits is not None:
            if not set(qbs).issubset(set(qubits)):
                raise ValueError("Incommensurate qubits")
            qbs = qubits

        super().__init__(qubits=qbs)
        self._circ_qubits = qubits
        self._elements = elements

    def add(self: CircuitType, other: Iterable[BaseOperation]) -> "CircuitType":
        """Concatenate operations and return new circuit"""
        return type(self)(*chain(self, other), qubits=self._circ_qubits)

    def asgate(self) -> BaseGate:
        gate: BaseGate = Identity(self.qubits)
        for elem in self:
            gate = elem.asgate() @ gate
        return gate

    def flat(self: CircuitType) -> Iterable[BaseOperation]:
        # DOCME
        for elem in self:
            if isinstance(elem, Circuit):
                yield from elem.flat()
            else:
                yield from elem

    @property
    def H(self: CircuitType) -> "CircuitType":
        elements = [elem.H for elem in self._elements[::-1]]
        return type(self)(*elements, qubits=self._circ_qubits)

    # def on(self, *qubits: Qubit) -> "Moment":
    #     return Moment(Circuit(self).on(*qubits))

    # def relabel(self, labels: Dict[Qubit, Qubit]) -> "Moment":
    #     return Moment(Circuit(self).relabel(labels))

    def __add__(self: CircuitType, other: Iterable[BaseOperation]) -> "CircuitType":
        return self.add(other)

    @overload
    def __getitem__(self, key: int) -> BaseOperation:
        pass

    @overload  # noqa: F811
    def __getitem__(self: CircuitType, key: slice) -> "CircuitType":
        pass

    def __getitem__(self, key: Union[int, slice]) -> BaseOperation:  # noqa: F811
        if isinstance(key, slice):
            return Circuit(*self._elements[key])
        return self._elements[key]

    def __iadd__(self: CircuitType, other: Iterable[BaseOperation]) -> "CircuitType":
        return self.add(other)

    def __iter__(self) -> Iterator[BaseOperation]:
        yield from self._elements

    def __len__(self) -> int:
        return len(self._elements)

    # # TODO: Support Variable?
    # def __pow__(self, t: float) -> "Repeat":
    #     return Repeat(self, 1) ** t

    # # FIXME
    # def __str__(self) -> str:
    #     circ_str = "\n".join([str(elem) for elem in self])
    #     circ_str = textwrap.indent(circ_str, " " * CIRCUIT_INDENT)
    #     return "\n".join([self.name, circ_str])

    # #FIXMEs
    # def __repr__(self) -> str:
    #     circ_str = ",\n".join([repr(elem) for elem in self])
    #     if self._circ_qubits is not None:
    #         circ_str.append("qubits=" + repr(self._circ_qubits) + "\n")
    #     circ_str = textwrap.indent(circ_str, " " * CIRCUIT_INDENT)
    #     return "\n".join([self.name + "(", circ_str, ")"])


class Moment(Circuit):
    """
    Represents a collection of Operations that operate on disjoint qubits,
    so that they may be applied at the same moment of time.
    """

    def __init__(self, *elements: BaseOperation, qubits: Qubits = None) -> None:
        circ = Circuit(*elements).flat()
        super().__init__(*elements, qubits=qubits)

        qbs = list(q for elem in circ for q in elem.qubits)
        if len(qbs) != len(set(qbs)):
            raise ValueError("Qubits of operations within Moments must be disjoint.")


# # # TODO: Move to ops
# class Repeat(BaseOperation):
#     def __init__(self, op: BaseOperation, repeat: int):
#         super().__init__(op.qubits)
#         self._operation = op
#         self._repeat = repeat

#     def ascircuit(self) -> Circuit:
#         return Circuit(*([self.operation] * self.repeat))

#     def asgate(self) -> BaseGate:
#         return self.ascircuit().asgate()

#     @property
#     def H(self) -> "Repeat":
#         return Repeat(self.operation.H, self.repeat)

#     # TODO: Support Variable?
#     def __pow__(self, t: float) -> "Repeat":
#         if not t.is_integer() or t < 0.0:
#             raise ValueError("Only positive integer powers supported.")
#         return Repeat(self, int(t) * self.repeat)


# # end class Moment
