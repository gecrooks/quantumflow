# Copyright 2019-, Gavin E. Crooks
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

.. autofunction:: sorted_bits


"""
# TODO: Add docs for other members

from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import numpy as np

from .future import Protocol

__all__ = (
    "qubit_dtype",
    "QubitTensor",
    "asqutensor",
    "Address",
    "Qubit",
    "Qubits",
    "Cbit",
    "Cbits",
    "sorted_bits",
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike  # pragma: no cover


qubit_dtype = np.complex128
"""The complex data type used for quantum data"""

QubitTensor = np.ndarray
"""Type hint for numpy arrays representing quantum data."""


def asqutensor(array: "ArrayLike", ndim: int = None) -> QubitTensor:
    """Converts a tensor like object to a numpy array object with complex data type.

    If rank is given (vectors ndim=1, operators ndim=2, super-operators ndim=4)
    we reshape the array to have than number of axes.
    """
    tensor = np.asarray(array, dtype=qubit_dtype)

    N = np.size(tensor)
    K = int(np.log2(N))
    if 2 ** K != N:
        raise ValueError("Wrong number of elements. Must be 2**N where N is an integer")

    if ndim is not None:
        shape = (2 ** (K // ndim),) * ndim
        tensor = tensor.reshape(shape)

    return tensor


class Address(Protocol):
    """A label for a piece of data. This Protocol specifies any sortable and hashable
    python object (i.e. most immutable types).
    Examples include strings, integers, tuples of strings and integers, etc."""

    def __lt__(self, other: Any) -> bool:
        pass

    def __hash__(self) -> int:
        pass


AddressType = TypeVar("AddressType", bound=Address)


class Qubit(Address, Protocol):
    """Type for qubit labels. Any any sortable and hashable python object."""


class Cbit(Address, Protocol):
    """Type for labels of classical bits. Any sortable and hashable python object."""


Addresses = Sequence[Address]
"""Type for sequence of addresses"""


Qubits = Sequence[Qubit]
"""Type for sequence of qubits"""


Cbits = Sequence[Cbit]
"""Type for sequence of classical bits"""


def sorted_bits(bits: Sequence[AddressType]) -> Sequence[AddressType]:
    """Return a sorted list of unique bit labels in canonical order.

    Data labels (Qubit and Cbit types) can be of different types, so we sort first by
    type (as a string), then within types.
    """
    return tuple(sorted(set(bits), key=lambda x: (str(type(x)), x)))


# fin
