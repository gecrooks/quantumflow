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


from typing import Any, Sequence

from .utils.future import Protocol

__all__ = (
    "Addr",
    "Addrs",
    "Qubit",
    "Qubits",
    "Cbit",
    "Cbits",
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


class Cbit(SortableHashable, Protocol):
    """Type for labels of classical bits. Any sortable and hashable python object."""


class Addr(SortableHashable, Protocol):
    """An address for a chunk of classical data. Any sortable and hashable python
    object."""


Qubits = Sequence[Qubit]
"""Type for sequence of qubits"""


Cbits = Sequence[Cbit]
"""Type for sequence of classical bits"""

Addrs = Sequence[Addr]
"""Type for sequence of addresses"""


# fin
