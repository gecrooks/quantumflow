
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

from typing import Tuple

import numpy as np
import jax.numpy as jaxnp
import jax.numpy as jnp
import jax


from jax.numpy import sqrt, pi, conj  # noqa: F401


from jax.numpy import (  # noqa: F401
    minimum,
    arccos, exp, cos, sin, reshape, size,
    real, imag, matmul, absolute, trace, diag,
    einsum, outer, ndim, roll)
from jax.numpy import ndim as rank

from jax.numpy import sum as reduce_sum                 # noqa: F401

from jax.numpy import (tensordot, transpose)

from .numpybk import __all__

TL = jax
"""'TensorLibrary'. The actual imported backend python package
"""

name = TL.__name__
"""The tensor library's name"""

version = TL.__version__
"""The tensor library's version"""


DEVICE = 'cpu'
"""Current device"""
# FIXME DOCME


CTYPE = jaxnp.complex64
"""The complex datatype used by the backend
"""


FTYPE = jaxnp.float64
"""Floating point datatype used by the backend
"""


TENSOR = jaxnp.ndarray
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
    array = jaxnp.asarray(array, dtype=CTYPE)
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
    return np.asarray(tensor)


def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two tensors"""
    # Note: Relying on fact that vdot flattens arrays
    return jaxnp.vdot(tensor0, tensor1)


def cis(theta: float) -> BKTensor:
    r""":returns: complex exponential

    .. math::
        \text{cis}(\theta) = \cos(\theta)+ i \sin(\theta) = \exp(i \theta)
    """
    return jaxnp.exp(theta*1.0j)


def set_random_seed(seed: int) -> None:
    """Reinitialize the random number generator"""
    np.random.seed(seed)


def getitem(tensor: BKTensor, key: typing.Any) -> BKTensor:
    """Get item from tensor"""
    return tensor.__getitem__(key)


def productdiag(tensor: BKTensor) -> BKTensor:
    """Returns the matrix diagonal of the product tensor"""  # DOCME: Explain
    N = rank(tensor)
    tensor = reshape(tensor, [2**(N//2), 2**(N//2)])
    tensor = jaxnp.diag(tensor)
    tensor = reshape(tensor, [2]*(N//2))
    return tensor

import timeit, time

from opt_einsum import contract



def _tensormul(tensor0: BKTensor, tensor1: BKTensor,
                        indices: typing.List[int], diagonal = False) -> BKTensor:
    t0 = time.process_time()
    N = rank(tensor1)
    K = rank(tensor0) // 2
    assert K == len(indices)

    left_out = indices
    left_in = list(range(N, N+K))
    right = list(range(0, N))
    for idx, s in zip(indices, left_in):
        right[idx] = s

    tensor = contract(tensor0, left_out+left_in, tensor1, right)

    return tensor


def func(tensor0, tensor1, idx):
    return jnp.tensordot(tensor0, tensor1, idx)
func = jax.jit(func, static_argnums=[2])




def tensormul(tensor0: BKTensor, tensor1: BKTensor,
              indices: Tuple[int, ...],
              diagonal=False) -> BKTensor:
    N = rank(tensor1)
    K = rank(tensor0) // 2
    assert K == len(indices)

    perm = list(indices) + [n for n in range(N) if n not in indices]
    inv_perm = np.argsort(perm)

    tensor = tensor1
    tensor = tensordot(tensor0, tensor1, (list(range(K, 2*K)), indices))
    #tensor = func(tensor0, tensor1, (tuple(range(K, 2*K)), tuple(indices)))
    tensor = transpose(tensor, inv_perm)

    return tensor

tensormul = jax.jit(tensormul, static_argnums=[2, 3])
