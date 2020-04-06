
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
=======
Backend
=======

.. module:: quantumflow.backend
.. contents:: :local:


Tensor Library Backends
#######################
QuantumFlow is designed to use a modern tensor library as a backend.
The current options are tensorflow, eager, pytorch, and numpy (default).

- numpy (Default)
    Python classic. Relatively fast on a single CPU, but no GPU
    acceleration, and no backprop.


- tensorflow
    Tensorflow backend

- ctf (Cyclops Tensor Framework)
    Experimental prototype. Potentially fast for large qubit states.


Configuration
#############

The default backend can be set in the configuration file, and can be
overridden with the QUANTUMFLOW_BACKEND environment variable. e.g.  ::

  > QUANTUMFLOW_BACKEND=numpy pytest tests/test_backend.py

You can also set the environment variable in python before quantumflow is
imported.

    >>> import os
    >>> os.environ["QUANTUMFLOW_BACKEND"] = "tensorflow"
    >>> import quantumflow as qf


Backend API
###########

.. autofunction:: quantumflow.backend.tensormul

Each backend is expected to implement the following methods, with semantics
that match numpy. (For instance, tensorflow's acos() method is adapted to match
numpy's arccos())

- absolute
- arccos
- conj
- cos
- diag
- exp
- matmul
- minimum
- real
- reshape
- sin
- reduce_sum
- transpose

Note that numpy's sum() is imported as reduce_sum, to avoid conflicts with
python's builtin sum() function.

- pi  Numerical constant 3.14..
- PI  Symbolic pi (sympy.pi) where supported.

"""


import os

from ..config import ENV_PREFIX, SEED
from .numpybk import set_random_seed as np_set_random_seed

DEFAULT_BACKEND = 'numpy'
BACKENDS = ('tensorflow', 'ctf', 'numpy64', 'numpy')

# Environment variable override
_BACKEND_EV = ENV_PREFIX + 'BACKEND'
BACKEND = os.getenv(_BACKEND_EV, DEFAULT_BACKEND)
if BACKEND not in BACKENDS:  # pragma: no cover
    raise ValueError(f'Unknown backend: {_BACKEND_EV}={BACKEND}')

if BACKEND == 'tensorflow':                          # pragma: no cover
    from quantumflow.backend.tensorflowbk import *   # noqa: F403
elif BACKEND == 'ctf':                               # pragma: no cover
    from quantumflow.backend.ctfbk import *          # noqa: F403
elif BACKEND == 'numpy64':                           # pragma: no cover
    from quantumflow.backend.numpy64bk import *      # noqa: F403
else:                                                # pragma: no cover
    from quantumflow.backend.numpybk import *        # noqa: F403

__all__ = [  # noqa: F405
           'BKTensor', 'CTYPE', 'DEVICE', 'FTYPE', 'MAX_QUBITS', 'TENSOR',
           'TL', 'TensorLike', 'absolute', 'arccos', 'astensor',
           'ccast', 'conj', 'cos', 'diag', 'evaluate', 'exp', 'fcast',
           'gpu_available', 'imag', 'inner', 'minimum',
           'outer', 'matmul',
           'rank', 'real', 'reshape', 'set_random_seed', 'sin',
           'sqrt', 'reduce_sum', 'tensormul', 'trace', 'transpose',
           'getitem', 'astensorproduct', 'productdiag',
           'EINSUM_SUBSCRIPTS', 'einsum',
           '__version__', '__name__', 'pi', 'PI', 'sign']


if SEED is not None:               # pragma: no cover
    np_set_random_seed(SEED)
    set_random_seed(SEED)          # noqa: F405
