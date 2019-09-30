
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow numpy backend with 64 bit complex numbers
"""
# Experimental. Causes lots of tests to fail.

import numpy as np


from .numpybk import __all__                # noqa: F401

from .numpybk import (                      # noqa: F401
       BKTensor, DEVICE, MAX_QUBITS, TENSOR,
       TL, TensorLike, absolute, arccos,
       ccast, cis, conj, cos, diag, evaluate, exp, fcast,
       gpu_available, imag, inner, minimum,
       outer, matmul,
       ndim, real, reshape, set_random_seed, sin,
       sqrt, reduce_sum, tensormul, trace, transpose,
       getitem, productdiag,
       EINSUM_SUBSCRIPTS, einsum,
       version, name, size, contract, tensordot, roll)


CTYPE = np.complex64

FTYPE = np.float32


def astensor(array: TensorLike) -> BKTensor:    # noqa: F405
    """Converts a numpy array to the backend's tensor object
    """
    array = np.asarray(array, dtype=CTYPE)
    return array


def astensorproduct(array: TensorLike) -> BKTensor:     # noqa: F405
    """Converts a numpy array to the backend's tensor object, and reshapes
    to [2]*N (So the number of elements must be a power of 2)
    """
    tensor = astensor(array)
    N = int(np.log2(size(tensor)))                      # noqa: F405
    array = tensor.reshape([2]*N)
    return array
