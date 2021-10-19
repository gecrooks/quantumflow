# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. module:: quantumflow
.. contents:: :local:

States, gates, and various other methods accept a list of qubits labels upon
which the given State or Gate acts. A Qubit label can be any hashable and
sortable python object (most immutable types), but typically an integer or string.
e.g. `[0, 1, 2]`, or `['a', 'b', 'c']`. Similarly labels for classical bit labels.


.. autoclass:: Qubit

.. autoclass:: Cbit



"""
# TODO: Add docs for other members

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, TypeVar, Union

import numpy as np
import sympy as sym

from .config import quantum_dtype
from .utils.future import Protocol

__all__ = (
    "Addr",
    "Addrs",
    "Qubit",
    "Qubits",
    # "Cbit",
    # "Cbits",
    "Variable",
    "Variables",
    "QuantumState",
    "QuantumStateType",
    "State",
    "zero_state",
    "ghz_state",
)


class SortableHashable(Protocol):
    """This Protocol specifies any sortable and hashable python object (i.e. most immutable
    types). Examples include strings, integers, tuples of strings and integers, etc.
    """

    def __lt__(self, other: Any) -> bool:
        pass

    def __hash__(self) -> int:
        pass


class Qubit(SortableHashable, Protocol):
    """Type for qubit labels. Any any sortable and hashable python object."""


# class Cbit(SortableHashable, Protocol):
#     """Type for labels of classical bits. Any sortable and hashable python object."""


class Addr(SortableHashable, Protocol):
    """An address for a chunk of classical data. Any sortable and hashable python
    object."""


Qubits = Sequence[Qubit]
"""Type for sequence of qubits"""


# Cbits = Sequence[Cbit]
# """Type for sequence of classical bits"""

Addrs = Sequence[Addr]
"""Type for sequence of addresses"""

Variable = Union[float, sym.Expr]
"""Type for parameters. Either a float, sympy.Symbol or sympy.Expr"""

Variables = Sequence[Variable]
"""A sequence of Variables"""


# TODO: Frozen Dict
# TODO: QuantumState ABC, Density, QuantumStateType

QuantumStateType = TypeVar("QuantumStateType", bound="QuantumState")
"""Generic type annotations for subtypes of QuantumState"""


class QuantumState(ABC):
    @property
    @abstractmethod
    def addrs(self) -> Addrs:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> Dict[Addr, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def qubits(self) -> Qubits:
        raise NotImplementedError

    @property
    @abstractmethod
    def qubit_nb(self) -> Qubits:
        raise NotImplementedError


class State(QuantumState):
    def __init__(
        self, vector: np.ndarray, qubits: Qubits, data: Dict[Addr, Any] = None
    ):
        qubits = tuple(qubits)
        arr = np.asanyarray(vector).astype(quantum_dtype).flatten()
        assert len(arr) == 2 ** len(qubits)  # FIXME
        self._vector = arr
        self._qubits = qubits
        self._data = data if data is not None else {}

    @property
    def addrs(self) -> Addrs:
        return tuple(self.data.keys())

    @property
    def data(self) -> Dict[Addr, Any]:
        return self._data

    @property
    def qubits(self) -> Qubits:
        return self._qubits

    @property
    def qubit_nb(self) -> Qubits:
        return len(self.qubits)

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    @property
    def tensor(self) -> np.ndarray:
        return np.reshape(self._vector, [2] * self.qubit_nb)


# end class State


def zero_state(qubits: Qubits) -> State:
    """Return the all-zero state on the given qubits"""
    qubits = tuple(qubits)
    N = len(qubits)
    vec = np.zeros(shape=2 ** N)
    vec[0] = 1
    return State(vec, qubits)


def ghz_state(qubits: Union[int, Qubits]) -> State:
    """Return a GHZ state on N qubits"""
    qubits = tuple(qubits)
    N = len(qubits)
    vec = np.zeros(shape=[2] * N)
    vec[(0,) * N] = 1 / np.sqrt(2)
    vec[(1,) * N] = 1 / np.sqrt(2)
    return State(vec.flatten(), qubits)


# fin
