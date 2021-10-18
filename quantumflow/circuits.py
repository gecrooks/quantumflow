# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# import textwrap
from itertools import chain
from typing import Iterable, Iterator, Sequence, TypeVar, Union, overload

from .base import QuantumComposite, QuantumGate, QuantumOperation
from .bits import Cbits, Qubits

# from .config import CIRCUIT_INDENT
from .gates import Identity

CircuitType = TypeVar("CircuitType", bound="Circuit")
"""Generic type annotations for subtypes of Circuit"""


class Circuit(Sequence, QuantumComposite):

    # DOCME TODO: Adding composite operations is a little weird
    def add(self: CircuitType, other: Iterable[QuantumOperation]) -> "CircuitType":
        """Concatenate operations and return new circuit"""
        return type(self)(*chain(self, other))

    def asgate(self) -> QuantumGate:
        gate: QuantumGate = Identity(self.qubits)
        for elem in self:
            gate = elem.asgate() @ gate
        return gate

    def flat(self) -> Iterable[QuantumOperation]:
        # DOCME
        for elem in self:
            if isinstance(elem, Circuit):
                yield from elem.flat()
            else:
                yield from elem

    @property
    def H(self: CircuitType) -> "CircuitType":
        elements = [elem.H for elem in self._elements[::-1]]
        return type(self)(*elements, qubits=self.qubits, cbits=self.cbits)

    def __add__(self: CircuitType, other: Iterable[QuantumOperation]) -> "CircuitType":
        return self.add(other)

    def __iadd__(self: CircuitType, other: Iterable[QuantumOperation]) -> "CircuitType":
        return self.add(other)

    def __iter__(self) -> Iterator[QuantumOperation]:
        yield from self._elements

    @overload
    def __getitem__(self, key: int) -> QuantumOperation:
        pass

    @overload  # noqa: F811
    def __getitem__(self: CircuitType, key: slice) -> "CircuitType":
        pass

    def __getitem__(self, key: Union[int, slice]) -> QuantumOperation:  # noqa: F811
        if isinstance(key, slice):
            return Circuit(*self._elements[key])
        return self._elements[key]


class Moment(Circuit):
    """
    Represents a collection of Operations that operate on disjoint qubits,
    so that they may be applied at the same moment of time.
    """

    def __init__(
        self, *elements: QuantumOperation, qubits: Qubits = None, cbits: Cbits = None
    ) -> None:
        super().__init__(*elements, qubits=qubits, cbits=cbits)

        qbs = list(q for elem in self for q in elem.qubits)
        if len(qbs) != len(set(qbs)):
            raise ValueError("Qubits of operations within Moments must be disjoint.")
