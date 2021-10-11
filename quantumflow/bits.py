# Copyright 2019-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. module:: quantumflow
.. contents:: :local:
.. currentmodule:: quantumflow


States, gates, and various other methods accept a list of qubits labels upon
which the given State or Gate acts. A Qubit label can be any hashable and
sortable python object (most immutable types), but typically an integer or string.
e.g. `[0, 1, 2]`, or `['a', 'b', 'c']`. Similarly labels for classical bit labels.


.. autoclass:: Qubit

.. autofunction:: sorted_qubits

.. autoclass:: Cbit

.. autofunction:: sorted_cbits


"""

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .future import Protocol

__all__ = (
    "qubit_dtype",
    "QubitTensor",
    "asqutensor",
    "Qubit",
    "Qubits",
    "sorted_qubits",
    "Cbit",
    "Cbits",
    "sorted_cbits",
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike  # pragma: no cover


qubit_dtype = np.complex128
"""The complex data type used for quantum data"""

QubitTensor = np.ndarray
"""Type hint for numpy arrays representing quantum data."""


def asqutensor(array: "ArrayLike", rank: int = None) -> QubitTensor:
    """Converts a tensor like object to a numpy array object with complex data type.

    If rank is given (vectors rank 1, operators rank 2, super-operators rank 4)
    we reshape the array to have than number of axes.
    """
    tensor = np.asarray(array, dtype=qubit_dtype)

    N = np.size(tensor)
    K = int(np.log2(N))
    if 2 ** K != N:
        raise ValueError("Wrong number of elements. Must be 2**N where N is an integer")

    if rank is not None:
        shape = (2 ** (K // rank),) * rank
        tensor = tensor.reshape(shape)

    return tensor


class Qubit(Protocol):
    """Type for qubit labels. Any sortable and hashable python object.
    e.g. strings, integers, tuples of strings and integers, etc.
    """

    def __lt__(self, other: Any) -> bool:
        pass

    def __hash__(self) -> int:
        pass


Qubits = Sequence[Qubit]
"""Type for sequence of qubits"""


def sorted_qubits(qbs: Qubits) -> Qubits:
    """Return a sorted list of unique qubits in canonical order.

    Qubits can be of different types, so we sort first by type (as a string),
    then within types.
    """
    return tuple(sorted(list(set(qbs)), key=lambda x: (str(type(x)), x)))


class Cbit(Protocol):
    """Type for labels of classical bits. Any sortable and hashable python object.
    e.g. strings, integers, tuples of strings and integers, etc.
    """

    def __lt__(self, other: Any) -> bool:
        pass

    def __hash__(self) -> int:
        pass


Cbits = Sequence[Cbit]
"""Type for sequence of cbits"""


def sorted_cbits(qbs: Qubits) -> Qubits:
    """Return a sorted list of unique cbits in canonical order.

    Cbit labels can be of different types, so we sort first by type (as a string),
    then within types.
    """
    return tuple(sorted(list(set(qbs)), key=lambda x: (str(type(x)), x)))


# fin
