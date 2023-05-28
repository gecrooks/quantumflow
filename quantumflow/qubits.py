# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. module:: quantumflow
.. contents:: :local:
.. currentmodule:: quantumflow

Qubits
======

States, gates, and various other methods accept a list of qubits labels upon
which the given State or Gate acts. A Qubit label can be any hashable and
sortable python object (most immutable types), but typically an integer or string.
e.g. `[0, 1, 2]`, or `['a', 'b', 'c']`.

"""

from typing import Any, Sequence

from .future import Protocol, TypeAlias

__all__ = ("Qubit", "Qubits", "sorted_qubits")


class Qubit(Protocol):
    """Type for qubits. Any sortable and hashable python object.
    e.g. strings, integers, tuples of strings and integers, etc.
    """

    def __lt__(self, other: Any, /) -> bool:
        pass

    def __hash__(self) -> int:
        pass


Qubits: TypeAlias = Sequence[Qubit]
"""Type for sequence of qubits"""


def sorted_qubits(qbs: Qubits) -> Qubits:
    """Return a sorted list of unique qubits in canonical order.

    Qubits can be of different types, so we sort first by type (as a string),
    then within types.
    """
    return tuple(sorted(list(set(qbs)), key=lambda x: (str(type(x)), x)))


# fin
