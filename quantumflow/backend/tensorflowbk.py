
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow tensorflow backend.
"""

import math
import typing
import string
import numpy as np


import tensorflow as tf
from tensorflow import transpose, minimum, exp, cos, sin        # noqa: F401


from tensorflow import matmul                                   # noqa: F401
from tensorflow import abs as absolute                          # noqa: F401
from tensorflow import einsum, reshape                          # noqa: F401
from tensorflow.python.client import device_lib
from tensorflow import reduce_sum                               # noqa: F401
from tensorflow import roll, tensordot                          # noqa: F401
from tensorflow import identity as copy                         # noqa: F401

from .numpybk import set_random_seed as np_set_random_seed
from .numpybk import TensorLike, BKTensor, __all__              # noqa: F401

from opt_einsum import contract                                 # noqa: F401

# Tensorflow 2.0 is doing something weird with imports so that
# we can't use 'from tensorflow import real, imag, sqrt' any more.
real = tf.math.real
imag = tf.math.imag
sqrt = tf.math.sqrt
conj = tf.math.conj
diag = tf.linalg.diag_part
trace = tf.linalg.trace


TL = tf
name = TL.__name__
version = TL.__version__


# FIXME: Remove?
tf.compat.v1.InteractiveSession()             # TESTME: Is this safe to do?

CTYPE = tf.complex128
FTYPE = tf.float64

TENSOR = tf.Tensor

# Note if we use einsum in tensormul we will be limited to 26 qubits
MAX_QUBITS = 32


# FIXME
def gpu_available() -> bool:
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpus) != 0


DEVICE = 'gpu' if gpu_available() else 'cpu'


EINSUM_SUBSCRIPTS = string.ascii_lowercase
# Tensorflow's einsum only allows 26 indices alas


def ndim(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    return len(tensor.shape)


def ccast(value: complex) -> TensorLike:
    """Cast to complex tensor"""
    return tf.cast(value, CTYPE)


def fcast(value: float) -> TensorLike:
    return tf.cast(value, FTYPE)


def size(tensor: BKTensor) -> int:
    return np.prod(np.array(tensor.get_shape().as_list()))


def astensor(array: TensorLike) -> BKTensor:
    """Covert numpy array to tensorflow tensor"""
    if type(array) == TENSOR:
        return array
    tensor = tf.convert_to_tensor(value=array, dtype=CTYPE)
    return tensor


def astensorproduct(array: TensorLike) -> BKTensor:
    tensor = astensor(array)
    N = int(math.log2(size(tensor)))
    tensor = tf.reshape(tensor, ([2]*N))
    return tensor


def evaluate(tensor: BKTensor) -> TensorLike:
    """Return the value of a tensor"""
    return tensor.numpy()


def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two states"""
    # Note: Relying on fact that vdot flattens arrays
    N = ndim(tensor0)
    axes = list(range(N))
    return tf.tensordot(tf.math.conj(tensor0), tensor1, axes=(axes, axes))


def outer(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    return tf.tensordot(tensor0, tensor1, axes=0)


def cis(theta: float) -> BKTensor:
    """ cis(theta) = cos(theta)+ i sin(theta) = exp(i theta) """
    return tf.exp(theta*1.0j)


def arccos(theta: float) -> BKTensor:
    """Backend arccos"""
    return tf.acos(theta)


# def sum(tensor: BKTensor,
#         axis: typing.Union[int, typing.Tuple[int]] = None,
#         keepdims: bool = None) -> BKTensor:
#     return tf.reduce_sum(input_tensor=tensor, axis=axis, keepdims=keepdims)


def set_random_seed(seed: int) -> None:
    np_set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)


def getitem(tensor: BKTensor, key: typing.Any) -> BKTensor:
    return tensor.__getitem__(key)


def productdiag(tensor: BKTensor) -> BKTensor:
    N = ndim(tensor)
    tensor = reshape(tensor, [2**(N//2), 2**(N//2)])
    tensor = tf.linalg.tensor_diag_part(tensor)
    tensor = reshape(tensor, [2]*(N//2))
    return tensor


def tensormul(tensor0: BKTensor, tensor1: BKTensor,
              indices: typing.Tuple[int, ...],
              diagonal: bool = False) -> BKTensor:
    N = ndim(tensor1)
    K = ndim(tensor0) // 2
    assert K == len(indices)

    out = list(EINSUM_SUBSCRIPTS[0:N])
    left_in = list(EINSUM_SUBSCRIPTS[N:N+K])
    left_out = [out[idx] for idx in indices]
    right = list(EINSUM_SUBSCRIPTS[0:N])
    for idx, s in zip(indices, left_in):
        right[idx] = s

    subscripts = ''.join(left_out + left_in + [','] + right + ['->'] + out)
    # print('>>>', K, N, subscripts)

    tensor = einsum(subscripts, tensor0, tensor1)
    return tensor
