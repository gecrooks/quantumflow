
# Author: G. E. Crooks (2019)

"""
QuantumFlow: Experimental backend for Cyclops Tensor Framework

https://github.com/cyclops-community/ctf/
"""


import math
import typing
import string

import numpy as np
from numpy import pi, cos, sin, arccos, exp                # noqa: F401

import ctf
from ctf import (  # noqa: F401
    # sqrt,
    # pi,
    conj,
    transpose,
    # minimum,
    # arccos,
    # exp,
    # cos,
    # sin,
    reshape,
    # size,
    real,
    imag,
    # matmul,
    # absolute,
    trace,
    diag,
    einsum,
    # outer,
    )


from ctf import abs as absolute                       # noqa: F401
from ctf import sum as reduce_sum                     # noqa: F401


TL = ctf

name = TL.__name__


version = '?.?.?'  # FIXME


DEVICE = 'cpu'


CTYPE = np.complex128
"""The complex datatype used by the backend"""


FTYPE = np.float64
"""Floating point datatype used by the backend"""


TENSOR = ctf.tensor
"""Datatype of the backend tensors."""


BKTensor = typing.Any
"""Type hint for backend tensors"""


TensorLike = typing.Any
"""Any python object that can be converted into a backend tensor"""


# cft can have more tensor indicies than this, but we're currently constrained
# by the limits of numpy. Could probably be worked around.
MAX_QUBITS = 32


EINSUM_SUBSCRIPTS = string.ascii_lowercase + string.ascii_uppercase
# ctf allows more subscripts than this, but not clear what the full set is.


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
    if type(array) == ctf.tensor:
        return array
    return ctf.astensor(array, dtype=CTYPE)


def astensorproduct(array: TensorLike) -> BKTensor:
    """Converts a numpy array to the backend's tensor object, and reshapes
    to [2]*N (So the number of elements must be a power of 2)
    """
    tensor = astensor(array)
    N = int(math.log2(size(tensor)))
    shape = [2]*N
    if tensor.shape != shape:           # Only reshape if necessary
        tensor = tensor.reshape(shape)
    return tensor


def evaluate(tensor: BKTensor) -> TensorLike:
    """Returns the value of a tensor as an ordinary python object"""
    if type(tensor) == ctf.tensor:
        return tensor.to_nparray()
    return tensor


def sqrt(tensor: BKTensor) -> TensorLike:
    # return ctf.power(tensor, 0.5) # Does not work. Bug in cft.
    return ctf.astensor(np.sqrt(evaluate(tensor)), dtype=CTYPE)


def minimum(tensor0: BKTensor, tensor1: BKTensor) -> TensorLike:
    return np.minimum(evaluate(tensor0), evaluate(tensor1))


def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    return len(tensor.shape)


def size(tensor: BKTensor) -> int:
    return np.prod(np.array(tensor.shape))


def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two states"""
    N = rank(tensor0)
    axes = list(range(N))
    return conj(tensor0).tensordot(tensor1, axes=(axes, axes))


def outer(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    return tensor0.tensordot(tensor1, axes=0)


def matmul(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    return tensor0 @ tensor1


def cis(theta: float) -> BKTensor:
    return np.exp(theta*1.0j)


def set_random_seed(seed: int) -> None:
    """Reinitialize the random number generator"""
    ctf.random.seed(seed)


def getitem(tensor: BKTensor, key: typing.Any) -> BKTensor:
    """Get item from tensor"""
    return tensor.__getitem__(key)


def productdiag(tensor: BKTensor) -> BKTensor:
    """Returns the matrix diagonal of the product tensor"""
    N = rank(tensor)
    tensor = reshape(tensor, [2**(N//2), 2**(N//2)])
    tensor = ctf.diag(tensor)
    tensor = reshape(tensor, [2]*(N//2))
    return tensor


def tensormul(tensor0: BKTensor, tensor1: BKTensor,
              indices: typing.List[int]) -> BKTensor:
    N = rank(tensor1)
    K = rank(tensor0) // 2
    assert K == len(indices)

    out = list(EINSUM_SUBSCRIPTS[0:N])
    left_in = list(EINSUM_SUBSCRIPTS[N:N+K])
    left_out = [out[idx] for idx in indices]
    right = list(EINSUM_SUBSCRIPTS[0:N])
    for idx, s in zip(indices, left_in):
        right[idx] = s

    subscripts = ''.join(left_out + left_in + [','] + right + ['->'] + out)

    tensor = einsum(subscripts, tensor0, tensor1)
    return tensor
