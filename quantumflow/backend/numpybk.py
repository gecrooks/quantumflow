
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow numpy backend
"""


import math
import typing
import string

import numpy as np
from numpy import (  # noqa: F401
    sqrt, pi, conj, minimum,
    arccos, exp, cos, sin, reshape, size,
    real, imag, matmul, absolute, trace, diag,
    outer, tensordot, einsum, transpose, roll, ndim, copy)

from numpy import sum as reduce_sum                 # noqa: F401

from opt_einsum import contract

from ..utils import multi_slice


__all__ = [  # noqa: F405
           'BKTensor', 'CTYPE', 'DEVICE', 'FTYPE', 'MAX_QUBITS', 'TENSOR',
           'TL', 'TensorLike', 'absolute', 'arccos', 'astensor',
           'ccast', 'cis', 'conj', 'cos', 'diag', 'evaluate', 'exp', 'fcast',
           'gpu_available', 'imag', 'inner', 'minimum',
           'outer', 'matmul',
           'ndim', 'real', 'reshape', 'set_random_seed', 'sin',
           'sqrt', 'reduce_sum', 'tensormul', 'trace', 'transpose',
           'getitem', 'astensorproduct', 'productdiag',
           'EINSUM_SUBSCRIPTS', 'einsum',
           'version', 'name', 'size', 'contract', 'tensordot', 'roll', 'copy']


TL = np
"""'TensorLibrary'. The actual imported backend python package
"""

name = TL.__name__
"""The tensor library's name"""

version = TL.__version__
"""The tensor library's version"""


DEVICE = 'cpu'
"""Current device"""
# FIXME DOCME


CTYPE = np.complex128
"""The complex data type used by the backend
"""


FTYPE = np.float64
"""Floating point data type used by the backend
"""


TENSOR = np.ndarray
"""Datatype of the backend tensors.
"""


BKTensor = typing.Any
"""Type hint for backend tensors"""
# Just used for documentation right now. Type checking numpy arrays
# not really supported yet (Jan 2018)


TensorLike = typing.Any
"""Any python object that can be converted into a backend tensor
"""
# Only used for documentation currently. Type checking numpy arrays and
# similar things not really supported yet. (Jan 2018)


MAX_QUBITS = 32
"""
Maximum number of qubits supported by this backend. Numpy arrays can't
have more than 32 dimensions, which limits us to no more than 32 qubits.
Pytorch has a similar problem, leading to a maximum of 24 qubits
"""


EINSUM_SUBSCRIPTS = string.ascii_lowercase + string.ascii_uppercase
"""
A string of all characters that can be used in einsum subscripts in
sorted order
"""


def gpu_available() -> bool:
    """Does the backend support GPU acceleration on current hardware?"""
    return False


def ccast(value: complex) -> TensorLike:
    """Cast value to complex tensor (if necessary)"""
    return value


def fcast(value: float) -> TensorLike:
    """Cast value to float tensor (if necessary)"""
    return value


def astensor(array: TensorLike) -> BKTensor:
    """Converts a numpy array to the backend's tensor object
    """
    array = np.asarray(array, dtype=CTYPE)
    return array


def astensorproduct(array: TensorLike) -> BKTensor:
    """Converts a numpy array to the backend's tensor object, and reshapes
    to [2]*N (So the number of elements must be a power of 2)
    """
    tensor = astensor(array)
    N = int(math.log2(size(tensor)))
    array = tensor.reshape([2]*N)
    return array


def evaluate(tensor: BKTensor) -> TensorLike:
    """:returns: the value of a tensor as an ordinary python object"""
    return tensor


def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two tensors"""
    # Note: Relying on fact that vdot flattens arrays
    return np.vdot(tensor0, tensor1)


def cis(theta: float) -> BKTensor:
    r""":returns: complex exponential

    .. math::
        \text{cis}(\theta) = \cos(\theta)+ i \sin(\theta) = \exp(i \theta)
    """
    return np.exp(theta*1.0j)


def set_random_seed(seed: int) -> None:
    """Reinitialize the random number generator"""
    np.random.seed(seed)


def getitem(tensor: BKTensor, key: typing.Any) -> BKTensor:
    """Get item from tensor"""
    return tensor.__getitem__(key)


def productdiag(tensor: BKTensor) -> BKTensor:
    """Returns the matrix diagonal of the product tensor"""  # DOCME: Explain
    N = ndim(tensor)
    tensor = reshape(tensor, [2**(N//2), 2**(N//2)])
    tensor = np.diag(tensor)
    tensor = reshape(tensor, [2]*(N//2))
    return tensor


def tensormul(tensor0: BKTensor, tensor1: BKTensor,
              indices: typing.Tuple[int, ...],
              diagonal: bool = False) -> BKTensor:
    r"""
    Generalization of matrix multiplication to product tensors.

    A state vector in product tensor representation has N dimension, one for
    each contravariant index, e.g. for 3-qubit states
    :math:`B^{b_0,b_1,b_2}`. An operator has K dimensions, K/2 for
    contravariant indices (e.g. ket components) and K/2 for covariant (bra)
    indices, e.g. :math:`A^{a_0,a_1}_{a_2,a_3}` for a 2-qubit gate. The given
    indices of A are contracted against B, replacing the given positions.

    E.g. ``tensormul(A, B, [0,2])`` is equivalent to

    .. math::

        C^{a_0,b_1,a_1} =\sum_{i_0,i_1} A^{a_0,a_1}_{i_0,i_1} B^{i_0,b_1,i_1}

    Args:
        tensor0: A tensor product representation of a gate
        tensor1: A tensor product representation of a gate or state
        indices: List of indices of tensor1 on which to act.
    Returns:
        Resultant state or gate tensor

    """
    # Note: This method is the critical computational core of QuantumFlow
    # Different implementations kept for edification.

    if diagonal and len(indices) == 1:
        d = np.diag(tensor0)
        tensor = tensor1.copy()
        s0 = multi_slice(indices, [0])
        s1 = multi_slice(indices, [1])
        tensor[s0] *= d[0]
        tensor[s1] *= d[1]
        return tensor

    # return _tensormul_tensordot(tensor0, tensor1, indices)
    # return _tensormul_cirq(tensor0, tensor1, indices)
    return _tensormul_matmul(tensor0, tensor1, indices, diagonal)
    # return _tensormul_contract(tensor0, tensor1, indices)


def _tensormul_matmul(tensor0: BKTensor, tensor1: BKTensor,
                      indices: typing.Tuple[int, ...],
                      diagonal: bool = False) -> BKTensor:
    # About the same speed as tensordot
    N = ndim(tensor1)
    K = ndim(tensor0) // 2
    assert K == len(indices)

    gate = reshape(tensor0, [2**K, 2**K])

    perm = list(indices) + [n for n in range(N) if n not in indices]
    inv_perm = np.argsort(perm)

    tensor = tensor1
    tensor = transpose(tensor, perm)
    tensor = reshape(tensor, [2**K, 2**(N-K)])

    if diagonal:
        tensor = transpose(tensor)
        tensor = tensor * np.diag(gate)
        tensor = transpose(tensor)
    else:
        tensor = matmul(gate, tensor)

    tensor = reshape(tensor, [2]*N)
    tensor = transpose(tensor, inv_perm)

    return tensor


def _tensormul_cirq(tensor0: BKTensor, tensor1: BKTensor,
                    indices: typing.Tuple[int, ...]) -> BKTensor:
    from cirq import targeted_left_multiply
    tensor = targeted_left_multiply(tensor0, tensor1, indices)
    return tensor


def _tensormul_tensordot(tensor0: BKTensor, tensor1: BKTensor,
                         indices: typing.Tuple[int, ...]) -> BKTensor:
    # Significantly faster than using einsum.
    N = ndim(tensor1)
    K = ndim(tensor0) // 2
    assert K == len(indices)

    perm = list(indices) + [n for n in range(N) if n not in indices]
    inv_perm = np.argsort(perm)

    tensor = tensordot(tensor0, tensor1, (range(K, 2*K), indices))
    tensor = transpose(tensor, inv_perm)

    return tensor


def _tensormul_contract(tensor0: BKTensor, tensor1: BKTensor,
                        indices: typing.Tuple[int, ...]) -> BKTensor:

    N = ndim(tensor1)
    K = ndim(tensor0) // 2
    assert K == len(indices)

    left_out = list(indices)
    left_in = list(range(N, N+K))
    right = list(range(0, N))
    for idx, s in zip(indices, left_in):
        right[idx] = s

    tensor = contract(tensor0, tuple(left_out+left_in), tensor1, tuple(right))

    return tensor
