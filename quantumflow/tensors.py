# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# DOCME

import string
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = ("QubitTensor", "asqutensor")


qubit_dtype = np.complex128
"""The complex data type used by the backend"""

QubitTensor = np.ndarray
"""Type hint for numpy arrays representing quantum data.
QubitTensor arrays have complex data type, and all axes have length 2."""


def asqutensor(array: ArrayLike) -> QubitTensor:
    """Converts a tensor like object to a numpy array object with complex data
    type, and reshapes to [2]*N (So the number of elements must be a power of 2)
    """
    tensor = np.asarray(array, dtype=qubit_dtype)

    N = int(np.log2(np.size(tensor)))
    shape = (2,) * N

    if tensor.shape != shape:  # Only reshape if necessary
        tensor = tensor.reshape(shape)

    return tensor


EINSUM_SUBSCRIPTS = string.ascii_lowercase + string.ascii_uppercase
"""Subscripts that can be used with numpy's einsum"""


def flatten(tensor: QubitTensor, rank: int) -> np.ndarray:
    """Return tensor with with qubit indices flattened"""
    R = rank
    N = np.ndim(tensor) // R
    return np.reshape(tensor, [2 ** N] * R)


def transpose(tensor: QubitTensor, perm: Sequence[int] = None) -> QubitTensor:
    """(Super)-operator transpose. Permutes the meta-indices.
    Default is to invert the meta-index order.
    """
    R = len(perm) if perm is not None else 2
    tensor = flatten(tensor, R)
    tensor = np.transpose(tensor, perm)
    tensor = asqutensor(tensor)

    return tensor


def conj_transpose(tensor: QubitTensor) -> QubitTensor:
    """Return the conjugate transpose of this tensor."""
    tensor = transpose(tensor)
    tensor = np.conj(tensor)
    return tensor


def permute(tensor: QubitTensor, perm: Sequence[int]) -> QubitTensor:
    """Permute the order of the subindexes"""
    N = len(perm)
    R = np.ndim(tensor) // N

    pperm: List[int] = []
    for rr in range(0, R):
        pperm += [rr * N + idx for idx in perm]
    tensor = np.transpose(tensor, pperm)

    return tensor


def inner(tensor0: QubitTensor, tensor1: QubitTensor) -> QubitTensor:
    """Return the inner product between two meta tensors
    (Assuming same meta-shape)"""
    # Note: Relies on fact that vdot flattens arrays
    return np.vdot(tensor0, tensor1)  # type: ignore


def outer(tensor0: QubitTensor, tensor1: QubitTensor, rank: int) -> QubitTensor:
    """Outer (direct) product of qubit tensors"""
    R = rank
    N0 = np.ndim(tensor0) // R
    N1 = np.ndim(tensor1) // R

    tensor = np.outer(tensor0, tensor1)

    # Interleave meta axes
    # R = 1  perm = (0, 1)
    # R = 2  perm = (0, 2, 1, 3)
    # R = 4  perm = (0, 4, 1, 5, 2, 6, 3, 7)
    tensor = np.reshape(tensor, ([2 ** N0] * R) + ([2 ** N1] * R))
    perm = [idx for ij in zip(range(0, R), range(R, 2 * R)) for idx in ij]
    if R != 1:
        tensor = transpose(tensor, perm)

    return tensor


def trace(tensor: QubitTensor, rank: int) -> float:
    """
    Return the trace, the sum of the diagonal elements of the (super)
    operator.
    """
    R = rank
    N = np.ndim(tensor) // R

    if R == 1:
        raise ValueError("Cannot take trace of vector")  # pragma: no cover

    tensor = np.reshape(tensor, [2 ** (N * R // 2)] * 2)
    trace = np.trace(tensor)
    return float(trace)


def norm(tensor: QubitTensor) -> QubitTensor:
    """Return the norm of this vector"""
    return np.absolute(inner(tensor, tensor))


def diag(tensor: QubitTensor) -> QubitTensor:
    """Returns the matrix diagonal of the product tensor"""
    tensor = flatten(tensor, 2)
    tensor = np.diag(tensor)
    tensor = asqutensor(tensor)

    return tensor


# TESTME
# TESTME on channels and density
def partial_trace(
    tensor: QubitTensor, indices: Sequence[int], rank: int = 2
) -> QubitTensor:
    """
    Return the partial trace over some subset of qubits,

    Args:
        tensor: A QubitTensor
        indices: The set of qubits that should be kept.
        rank: Rank of qubit tensor.
    """
    R = rank
    N = np.ndim(tensor) // R

    if R == 1:
        raise ValueError("Cannot take trace of vector")  # pragma: no cover

    subscripts = list(EINSUM_SUBSCRIPTS[0 : N * R])

    print(indices, R, N, subscripts)
    for idx in indices:
        for r in range(1, R):
            subscripts[r * N + idx] = subscripts[idx]
    subscript_str = "".join(subscripts)

    # Only numpy's einsum works with repeated subscripts
    tensor = np.einsum(subscript_str, tensor)

    return tensor


# DOCME
def tensormul(
    tensor0: QubitTensor,
    tensor1: QubitTensor,
    indices: Tuple[int, ...],
    diagonal: bool = False,
) -> QubitTensor:
    N = np.ndim(tensor1)
    K = np.ndim(tensor0) // 2
    assert K == len(indices)

    gate = np.reshape(tensor0, [2 ** K, 2 ** K])

    perm = list(indices) + [n for n in range(N) if n not in indices]
    inv_perm = np.argsort(perm)

    tensor = tensor1
    tensor = np.transpose(tensor, perm)
    tensor = np.reshape(tensor, [2 ** K, 2 ** (N - K)])

    if diagonal:
        tensor = np.transpose(tensor)
        tensor = tensor * np.diag(gate)
        tensor = np.transpose(tensor)
    else:
        tensor = np.matmul(gate, tensor)

    tensor = np.reshape(tensor, [2] * N)
    tensor = np.transpose(tensor, inv_perm)

    return tensor


# fin
