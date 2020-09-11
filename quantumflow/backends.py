
# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
=======
Backend
=======

.. module:: quantumflow.backends
.. contents:: :local:


Tensor Library Backends
#######################
QuantumFlow is designed to use a modern tensor library as a backend.
The current default is numpy.


Configuration
#############

The default backend can be set in the configuration file, and can be
overridden with the QUANTUMFLOW_BACKEND environment variable. e.g.  ::

  > QUANTUMFLOW_BACKEND=numpy make test

You can also set the environment variable in python before quantumflow is
imported.

    >>> import os
    >>> os.environ["QUANTUMFLOW_BACKEND"] = "tensorflow"
    >>> import quantumflow as qf


Standard Backends
#################
.. autoclass:: QFBackend
.. autoclass:: NumpyBackend
.. autoclass:: Numpy64Backend
.. autoclass:: TensorflowBackend
.. autoclass:: CtfBackend
"""
# Kudos: Backend infrastructure inspired by TensorNetwork by Chase Roberts
# https://github.com/google/TensorNetwork

from abc import ABC
from typing import Any, Union, Tuple
import string
import math
import os

import opt_einsum
import numpy as np
import sympy

from .utils import multi_slice
from .config import ENV_PREFIX, SEED

from types import ModuleType

Variable = Union[sympy.Expr, float]
"""Type for parameters. Either a float, sympy.Symbol or sympy.Expr"""

BKTensor = Any
"""Type hint for backend tensors"""
# Only used for documentation at present.

TensorLike = Any
"""Any python object that can be converted into a backend tensor"""
# Only used for documentation currently. Type checking numpy arrays and
# similar things not really supported yet.


class QFBackend(ABC):
    """Base class for QuantumFlow backends

    Each backend implements various tensor methods with semantics that match
    numpy. (For instance, tensorflow's acos() method is adapted to match
    numpy's arccos())

    """

    def __init__(self,
                 lib: ModuleType,
                 name: str,
                 version: str,
                 float_type: type,
                 complex_type: type,
                 tensor_type: type,
                 max_ndim: int,
                 einsum_subscripts: str,
                 ) -> None:

        self.lib = lib
        """'TensorLibrary'. The actual imported backend python package"""

        self.name = name
        """The tensor library's name"""

        self.version = float_type
        """The tensor library's version"""

        self.float_type = float_type
        """Floating point data type used by the backend"""

        self.complex_type = complex_type
        """The complex data type used by the backend"""

        self.tensor_type = tensor_type
        """The tensor data type used by the backend"""

        self.max_ndim = max_ndim
        """Maximum number of dimensions supported by this backend."""

        self.einsum_subscripts = einsum_subscripts
        """A string of all characters that can be used in einsum subscripts in
        sorted order"""

        self.symlib = sympy
        """Library for symbolic math"""

        self.symbolic_type = sympy.Expr
        """The symbolic data type"""

        self.pi = np.pi
        """Numerical constant pi"""

        self.PI = sympy.pi
        """Symbolic constant pi"""

        # Deprecated / legacy
        self.TL = self.lib
        self.MAX_QUBITS = self.max_ndim
        self.BACKEND = self.name
        self.BKTensor = BKTensor
        self.TensorLike = TensorLike

    # QF Backend functions

    def is_symbolic(self, x: Variable) -> bool:
        return isinstance(x, self.symbolic_type)

    def fcast(self, value: float) -> TensorLike:
        """Cast value to float tensor (if necessary)"""
        return value

    def ccast(self, value: complex) -> TensorLike:
        """Cast value to complex tensor (if necessary)"""
        return value

    # TODO: Make abstract method
    def astensor(self, array: TensorLike, dtype: type = None) -> BKTensor:
        """Converts a tensor like object to the backend's tensor object
        """
        dtype = self.complex_type if dtype is None else dtype
        array = self.lib.asarray(array, dtype=dtype)   # type: ignore
        return array

    def astensorproduct(self, array: TensorLike) -> BKTensor:
        """Converts a numpy array to the backend's tensor object, and reshapes
        to [2]*N (So the number of elements must be a power of 2)
        """
        tensor = self.astensor(array)
        N = int(math.log2(self.size(tensor)))
        shape = [2]*N
        # TODO: Check this works. shape is list or tuple or what?
        if tensor.shape != shape:           # Only reshape if necessary
            tensor = tensor.reshape(shape)
        return tensor

    def numpy(self, tensor: BKTensor) -> TensorLike:
        return tensor

    def inner(self, tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
        """Return the inner product between two tensors"""
        # Note: Relying on fact that vdot flattens arrays
        return self.lib.vdot(tensor0, tensor1)  # type: ignore

    def cis(self, theta: float) -> BKTensor:
        r""":returns: complex exponential

        .. math::
            \text{cis}(\theta) = \cos(\theta)+ i \sin(\theta) = \exp(i \theta)
        """
        return self.exp(theta*1.0j)

    # DOCME: Explain
    def productdiag(self, tensor: BKTensor) -> BKTensor:
        """Returns the matrix diagonal of the product tensor"""
        N = self.ndim(tensor)
        tensor = self.reshape(tensor, [2**(N//2), 2**(N//2)])
        tensor = self.diag(tensor)
        tensor = self.reshape(tensor, [2]*(N//2))
        return tensor

    def contract(self, *args: Any, **kwargs: Any) -> BKTensor:
        return opt_einsum.contract(*args, **kwargs)

    # Backend method adaption

    def absolute(self, x: BKTensor) -> BKTensor:
        return self.lib.absolute(x)        # type: ignore

    def arccos(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.acos(x)
        return self.lib.arccos(x)           # type: ignore

    def arcsin(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.asin(x)
        return self.lib.arcsin(x)           # type: ignore

    def arctan(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.atan(x)
        return self.lib.arctan(x)           # type: ignore

    def arctan2(self, x1: Variable, x2: Variable) -> Variable:
        if self.is_symbolic(x1) or self.is_symbolic(x2):
            return self.symlib.atan2(x1, x2)
        return self.lib.arctan2(x1, x2)     # type: ignore

    def conj(self, x: BKTensor) -> BKTensor:
        return self.lib.conj(x)             # type: ignore

    def cos(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.cos(x)
        return self.lib.cos(x)              # type: ignore

    def copy(self, x: BKTensor) -> Variable:
        return self.lib.copy(x)             # type: ignore

    def diag(self, x: BKTensor) -> BKTensor:
        return self.lib.diag(x)             # type: ignore

    def einsum(self, *args: Any, **kwargs: Any) -> BKTensor:
        return self.lib.einsum(*args, **kwargs)   # type: ignore

    def exp(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.exp(x)
        return self.lib.exp(x)             # type: ignore

    def imag(self, x: BKTensor) -> BKTensor:
        if self.is_symbolic(x):
            return self.symlib.im(x)
        return self.lib.imag(x)                # type: ignore

    def matmul(self, tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
        return tensor0 @ tensor1

    def minimum(self, t1: BKTensor, t2: BKTensor) -> BKTensor:
        return self.lib.minimum(t1, t2)     # type: ignore

    def ndim(self, x: Variable) -> int:
        return self.lib.ndim(x)             # type: ignore

    def outer(self, x1: BKTensor, x2: BKTensor) -> BKTensor:
        return self.lib.outer(x1, x2)          # type: ignore

    def real(self, x: BKTensor) -> BKTensor:
        if self.is_symbolic(x):
            return self.symlib.re(x)
        return self.lib.real(x)                 # type: ignore

    def reshape(self, x: BKTensor, newshape: Any) \
            -> BKTensor:
        return self.lib.reshape(x, newshape)    # type: ignore

    def seed(self, seed: int) -> None:
        """Reinitialize the random number generator"""
        self.lib.random.seed(seed)                 # type: ignore

    def sign(self, x: Variable) -> bool:
        if self.is_symbolic(x):
            return self.symlib.sign(x)
        return self.lib.sign(x)                 # type: ignore

    def sin(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.sin(x)
        return self.lib.sin(x)                  # type: ignore

    def size(self, x: BKTensor) -> BKTensor:
        return self.lib.size(x)                 # type: ignore

    def sqrt(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.sqrt(x)
        return self.lib.sqrt(x)                    # type: ignore

    def sum(self, x: BKTensor, *args: Any, **kwargs: Any) -> BKTensor:
        return self.lib.sum(x, *args, **kwargs)        # type: ignore

    def tan(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.tan(x)
        return self.lib.tan(x)                     # type: ignore

    def tensordot(self, x1: BKTensor, x2: BKTensor,
                  axes: Any = 2) -> BKTensor:
        return self.lib.tensordot(x1, x2, axes)          # type: ignore

    def trace(self, x: BKTensor) -> BKTensor:
        return self.lib.trace(x)              # type: ignore

    def transpose(self, x: BKTensor, axes: Any = None) -> BKTensor:
        return self.lib.transpose(x, axes)            # type: ignore

    # Deprecated methods

    # deprecated
    def set_random_seed(self, seed: int) -> None:
        self.seed(seed)

    # deprecated
    def reduce_sum(self, x: BKTensor, *args: Any, **kwargs: Any) -> BKTensor:
        return self.sum(x, *args, **kwargs)

    # deprecated
    def evaluate(self, tensor: BKTensor) -> TensorLike:
        """:returns: the value of a tensor as an ordinary python object"""
        return self.numpy(tensor)

    # TODO: Replace diagonal flag with tensor_structure
    def tensormul(self,
                  tensor0: BKTensor, tensor1: BKTensor,
                  indices: Tuple[int, ...],
                  diagonal: bool = False) -> BKTensor:
        r"""
        Generalization of matrix multiplication to product tensors.

        A state vector in product tensor representation has N dimension, one
        for
        each contravariant index, e.g. for 3-qubit states
        :math:`B^{b_0,b_1,b_2}`. An operator has K dimensions, K/2 for
        contravariant indices (e.g. ket components) and K/2 for covariant (bra)
        indices, e.g. :math:`A^{a_0,a_1}_{a_2,a_3}` for a 2-qubit gate. The
        given
        indices of A are contracted against B, replacing the given positions.

        E.g. ``tensormul(A, B, [0,2])`` is equivalent to

        .. math::

            C^{a_0,b_1,a_1} =\sum_{i_0,i_1} A^{a_0,a_1}_{i_0,i_1}
                                            B^{i_0,b_1,i_1}

        Args:
            tensor0: A tensor product representation of a gate
            tensor1: A tensor product representation of a gate or state
            indices: List of indices of tensor1 on which to act.
        Returns:
            Resultant state or gate tensor

        """
        N = self.ndim(tensor1)
        K = self.ndim(tensor0) // 2
        assert K == len(indices)

        out = list(self.einsum_subscripts[0:N])
        left_in = list(self.einsum_subscripts[N:N+K])
        left_out = [out[idx] for idx in indices]
        right = list(self.einsum_subscripts[0:N])
        for idx, s in zip(indices, left_in):
            right[idx] = s

        subscripts = ''.join(left_out + left_in + [','] + right + ['->'] + out)

        tensor = self.einsum(subscripts, tensor0, tensor1)
        return tensor


class NumpyBackend(QFBackend):
    def __init__(self) -> None:
        import numpy as np
        super().__init__(
            lib=np,
            name=np.__name__,
            version=np.__version__,
            complex_type=np.complex128,
            float_type=np.float64,
            tensor_type=np.ndarray,
            max_ndim=32,
            einsum_subscripts=(string.ascii_lowercase
                               + string.ascii_uppercase),
            )

    def tensormul(self,
                  tensor0: BKTensor, tensor1: BKTensor,
                  indices: Tuple[int, ...],
                  diagonal: bool = False) -> BKTensor:

        # Note: This method is the critical computational core of QuantumFlow
        # Different implementations kept for edification.

        if diagonal and len(indices) == 1:
            d = self.diag(tensor0)
            tensor = tensor1.copy()
            s0 = multi_slice(indices, [0])
            s1 = multi_slice(indices, [1])
            tensor[s0] *= d[0]
            tensor[s1] *= d[1]
            return tensor

        #return self._tensormul_tensordot(tensor0, tensor1, indices)
        return self._tensormul_cirq(tensor0, tensor1, indices)
        #return self._tensormul_matmul(tensor0, tensor1, indices, diagonal)
        #return self._tensormul_contract(tensor0, tensor1, indices)

    def _tensormul_matmul(self, tensor0: BKTensor, tensor1: BKTensor,
                          indices: Tuple[int, ...],
                          diagonal: bool = False) -> BKTensor:
        # About the same speed as tensordot
        N = self.ndim(tensor1)
        K = self.ndim(tensor0) // 2
        assert K == len(indices)

        gate = self.reshape(tensor0, [2**K, 2**K])

        perm = list(indices) + [n for n in range(N) if n not in indices]
        inv_perm = np.argsort(perm)

        tensor = tensor1
        tensor = self.transpose(tensor, perm)
        tensor = self.reshape(tensor, [2**K, 2**(N-K)])

        if diagonal:
            tensor = self.transpose(tensor)
            tensor = tensor * np.diag(gate)
            tensor = self.transpose(tensor)
        else:
            tensor = self.matmul(gate, tensor)

        tensor = self.reshape(tensor, [2]*N)
        tensor = self.transpose(tensor, inv_perm)

        return tensor

    def _tensormul_cirq(self, tensor0: BKTensor, tensor1: BKTensor,
                        indices: Tuple[int, ...]) -> BKTensor:
        from cirq import targeted_left_multiply
        tensor = targeted_left_multiply(tensor0, tensor1, indices)
        return tensor

    def _tensormul_tensordot(self, tensor0: BKTensor, tensor1: BKTensor,
                             indices: Tuple[int, ...]) -> BKTensor:
        # Significantly faster than using einsum.
        N = self.ndim(tensor1)
        K = self.ndim(tensor0) // 2
        assert K == len(indices)

        perm = list(indices) + [n for n in range(N) if n not in indices]
        inv_perm = np.argsort(perm)

        tensor = self.tensordot(tensor0, tensor1, (range(K, 2*K), indices))
        tensor = self.transpose(tensor, inv_perm)

        return tensor

    def _tensormul_contract(self, tensor0: BKTensor, tensor1: BKTensor,
                            indices: Tuple[int, ...]) -> BKTensor:

        N = self.ndim(tensor1)
        K = self.ndim(tensor0) // 2
        assert K == len(indices)

        left_out = list(indices)
        left_in = list(range(N, N+K))
        right = list(range(0, N))
        for idx, s in zip(indices, left_in):
            right[idx] = s

        tensor = self.contract(tensor0, tuple(left_out+left_in),
         tensor1, tuple(right))

        return tensor


class Numpy64Backend(NumpyBackend):
    """
    QuantumFlow numpy backend with 64 bit complex numbers.

    Experimental since it causes lots of tests to fail due to floating point
    roundoff errors
    """
    def __init__(self) -> None:
        import numpy as np
        super().__init__()

        self.complex_type = np.complex64
        self.float_type = np.float32


class TensorflowBackend(QFBackend):
    """

    """
    def __init__(self) -> None:
        import tensorflow as tf

        # # TESTME: Is this safe to do? Necessary?
        # tf.compat.v1.InteractiveSession()

        super().__init__(
            lib=tf,
            name=tf.__name__,
            version=tf.__version__,
            complex_type=tf.complex128,
            float_type=tf.float64,
            tensor_type=tf.Tensor,
            max_ndim=26,
            einsum_subscripts=string.ascii_lowercase,
            )

        self.tf = tf

    def ndim(self, tensor: BKTensor) -> int:
        return len(tensor.shape)

    def ccast(self, value: complex) -> TensorLike:
        """Cast to complex tensor"""
        if self.is_symbolic(value):
            value = complex(value)
        return self.tf.cast(value, self.complex_type)

    def fcast(self, value: float) -> TensorLike:
        if self.is_symbolic(value):
            value = float(value)
        return self.tf.cast(value, self.float_type)

    def size(self, tensor: BKTensor) -> int:
        return np.prod(np.array(tensor.get_shape().as_list()))

    def astensor(self, array: TensorLike, dtype: type = None) -> BKTensor:
        """Covert numpy array to tensorflow tensor"""
        dtype = self.complex_type if dtype is None else dtype
        tensor = self.tf.convert_to_tensor(value=array, dtype=dtype)
        return tensor

    def astensorproduct(self, array: TensorLike) -> BKTensor:
        tensor = self.astensor(array)
        N = int(math.log2(self.size(tensor)))
        tensor = self.tf.reshape(tensor, ([2]*N))
        return tensor

    def numpy(self, tensor: BKTensor) -> TensorLike:
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.numpy()

    def inner(self, tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
        N = self.ndim(tensor0)
        axes = list(range(N))
        return self.tensordot(self.tf.math.conj(tensor0), tensor1,
                              axes=(axes, axes))

    def outer(self, tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
        return self.tensordot(tensor0, tensor1, axes=0)

    def arccos(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.acos(x)
        return self.tf.acos(x)

    def arcsin(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.asin(x)
        return self.tf.asin(x)

    def arctan(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.atan(x)
        return self.tf.atan(x)

    def arctan2(self, x1: Variable, x2: Variable) -> Variable:
        if self.is_symbolic(x1) or self.is_symbolic(x2):
            return self.symlib.atan2(x1, x2)
        return self.tf.atan2(x1, x2)

    def imag(self, x: BKTensor) -> BKTensor:
        if self.is_symbolic(x):
            return self.symlib.im(x)
        return self.tf.math.imag(x)

    def real(self, x: BKTensor) -> BKTensor:
        if self.is_symbolic(x):
            return self.symlib.re(x)
        return self.tf.math.real(x)

    def conj(self, x: BKTensor) -> BKTensor:
        return self.tf.math.conj(x)

    def diag(self, x: BKTensor) -> BKTensor:
        return self.tf.linalg.diag_part(x)

    def trace(self, x: BKTensor) -> BKTensor:
        return self.tf.linalg.trace(x)

    def absolute(self, x: BKTensor) -> BKTensor:
        return self.tf.abs(x)

    def sqrt(self, x: Variable) -> BKTensor:
        if self.is_symbolic(x):
            return self.symlib.sqrt(x)
        x = self.fcast(x)
        return self.tf.math.sqrt(x)

    def sum(self, x: BKTensor, *args: Any, **kwargs: Any) -> BKTensor:
        return self.tf.reduce_sum(x, *args, **kwargs)

    def seed(self, seed: int) -> None:
        # FIXME
        self.tf.compat.v1.set_random_seed(seed)

    def copy(self, x: BKTensor) -> BKTensor:
        return self.tf.identity(x)

# End TensorflowBackend


class CtfBackend(QFBackend):  # pragma: no cover
    """
    QuantumFlow: Experimental backend for Cyclops Tensor Framework

    https://github.com/cyclops-community/ctf/
    """
    def __init__(self) -> None:
        import ctf

        super().__init__(
            lib=ctf,
            name=ctf.__name__,
            version="?.?.?",
            complex_type=np.complex128,
            float_type=np.float64,
            tensor_type=ctf.tensor,
            # Limited by numpy
            max_ndim=32,
            # ctf allows more subscripts than this?
            einsum_subscripts=string.ascii_lowercase + string.ascii_uppercase
            )

        self.ctf = ctf

    def sign(self, var: BKTensor) -> bool:
        if self.is_symbolic(var):
            return self.symlib.sign(var)
        return np.sign(self.numpy(var))

    def astensor(self, array: TensorLike, dtype: type = None) -> BKTensor:
        dtype = self.complex_type if dtype is None else dtype

        if type(array) == self.tensor_type:
            # TODO: Check dtype?
            return array

        return self.ctf.astensor(array, dtype)

    def numpy(self, tensor: BKTensor) -> TensorLike:
        if type(tensor) == self.tensor_type:
            return tensor.to_nparray()
        return tensor

    def sqrt(self, tensor: BKTensor) -> TensorLike:
        if self.is_symbolic(tensor):
            return self.symlib.sqrt(tensor)
        # return ctf.power(tensor, 0.5) # Does not work. Bug in cft.
        return self.astensor(np.sqrt(self.numpy(tensor)))

    def minimum(self, tensor0: BKTensor, tensor1: BKTensor) -> TensorLike:
        return self.astensor(np.minimum(self.numpy(tensor0),
                             self.numpy(tensor1)))

    def ndim(self, tensor: BKTensor) -> int:
        return len(tensor.shape)

    def size(self, tensor: BKTensor) -> int:
        return np.prod(np.array(tensor.shape))

    def inner(self, tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
        """Return the inner product between two states"""
        N = self.ndim(tensor0)
        axes = list(range(N))
        return self.conj(tensor0).tensordot(tensor1, axes=(axes, axes))

    def outer(self, tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
        return tensor0.tensordot(tensor1, axes=0)

    def cis(self, theta: float) -> BKTensor:
        return np.exp(theta*1.0j)

    def arccos(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.acos(x)
        return np.arccos(x)

    def arcsin(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.asin(x)
        return np.arcsin(x)

    def arctan(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.atan(x)
        return np.arctan(x)

    def arctan2(self, x1: Variable, x2: Variable) -> Variable:
        if self.is_symbolic(x1) or self.is_symbolic(x2):
            return self.symlib.atan2(x1, x2)
        return np.arctan2(x1, x2)

    def cos(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.cos(x)
        return np.cos(x)

    def sin(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.sin(x)
        return np.sin(x)

    def tan(self, x: Variable) -> Variable:
        if self.is_symbolic(x):
            return self.symlib.tan(x)
        return np.tan(x)

    def absolute(self, x: BKTensor) -> BKTensor:
        return self.ctf.abs(x)

    def imag(self, x: BKTensor) -> BKTensor:
        if self.is_symbolic(x):
            return self.symlib.im(x)
        if isinstance(x, complex):
            return x.imag
        return self.ctf.imag(x)

    def real(self, x: BKTensor) -> BKTensor:
        if self.is_symbolic(x):
            return self.symlib.re(x)
        if isinstance(x, complex):
            return x.real
        return self.ctf.real(x)

# End CtfBackend


# Set backend
DEFAULT_BACKEND = 'numpy'
BACKENDS = {'numpy': NumpyBackend,
            'numpy64': Numpy64Backend,
            'tensorflow': TensorflowBackend,
            'ctf': CtfBackend,
            }

# TODO: Delegate reading environment variables to config
# Environment variable override
_BACKEND_EV = ENV_PREFIX + 'BACKEND'
BACKEND = os.getenv(_BACKEND_EV, DEFAULT_BACKEND)


def get_backend(backend_name: str = None) -> QFBackend:
    if backend_name is None:
        backend_name = BACKEND
    if backend_name not in BACKENDS:  # pragma: no cover
        raise ValueError(f'Unknown backend: {backend_name}')

    return BACKENDS[backend_name]()


backend = get_backend(BACKEND)


if SEED is not None:               # pragma: no cover
    NumpyBackend().seed(SEED)
    backend.seed(SEED)
