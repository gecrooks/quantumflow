
"""
QuantumFlow numpy backend with 64 bit complex numbers
"""
# Experimental. Causes lots of tests to fail.

import numpy as np

from .numpybk import *                  # noqa: F403
from .numpybk import __all__            # noqa: F401

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
