
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

- eager
    Tensorflow eager mode. Tensorflow can automatically figure out
    back-propagated gradients, so we can efficiently optimize quantum networks
    using stochastic gradient descent.

- tensorflow
    Regular tensorflow. Eager mode recommened.

- tensorflow2
    Tensorflow 2.x backend. Eager is now the default operation mode.

- torch (Experimental)
    Experimental prototype. Fast on CPU and GPU. Unfortunately stochastic
    gradient descent not available due to pytorch's lack of support for
    complex math. Pytorch is not installed by default. See the pytorch website
    for installation instructions.

- ctf (Cyclops Tensor Framework)
    Experimental prototype. Potentially fast for large qubit states.


Configuration
#############

The default backend can be set in the configuration file, and can be
overridden with the QUANTUMFLOW_BACKEND environment variable. e.g.  ::

  > QUANTUMFLOW_BACKEND=numpy pytest tests/test_flow.py

Options are tensorflow, eager, numpy, and torch.

You can also set the environment variable in python before quantumflow is
imported.

    >>> import os
    >>> os.environ["QUANTUMFLOW_BACKEND"] = "numpy"
    >>> import quantumflow as qf


GPU
###

Unfortunately, tensorflow does not fully supports complex numbers,
so we cannot run with eager or tensofrlow mode on GPUs at present.
The numpy backend does not have GPU acceleration either.

The torch backened can run with GPU acceleration, which can lead to
significant speed increase for simulation of large quantum states.
Note that the main limiting factor is GPU memory. A single state uses 16 x 2^N
bytes. We need to be able to place 2 states (and a bunch of smaller tensors)
on a single GPU. Thus a 16 GiB GPU can simulate a 28 qubit system.

    > QUANTUMFLOW_DEVICE=gpu QUANTUMFLOW_BACKEND=torch ./benchmark.py 24
    > QUANTUMFLOW_DEVICE=cpu QUANTUMFLOW_BACKEND=torch ./benchmark.py 24


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

In addition each backend implements the following methods and variables.


.. automodule:: quantumflow.backend.numpybk
   :members:
"""


import os

from ..config import ENV_PREFIX, SEED
from .numpybk import set_random_seed as np_set_random_seed

DEFAULT_BACKEND = 'numpy'
BACKENDS = ('tensorflow', 'tensorflow2', 'eager', 'torch', 'ctf', 'numpy')

# Environment variable override
_BACKEND_EV = ENV_PREFIX + 'BACKEND'
BACKEND = os.getenv(_BACKEND_EV, DEFAULT_BACKEND)
if BACKEND not in BACKENDS:  # pragma: no cover
    raise ValueError('Unknown backend: {}={}'.format(_BACKEND_EV, BACKEND))

if BACKEND == 'tensorflow':                          # pragma: no cover
    from quantumflow.backend.tensorflowbk import *   # noqa: F403
elif BACKEND == 'eager':                             # pragma: no cover
    from quantumflow.backend.eagerbk import *        # noqa: F403
elif BACKEND == 'tensorflow2':                       # pragma: no cover
    from quantumflow.backend.tensorflow2bk import *  # noqa: F403
elif BACKEND == 'ctf':                             # pragma: no cover
    from quantumflow.backend.ctfbk import *        # noqa: F403
else:                                                # pragma: no cover
    from quantumflow.backend.numpybk import *        # noqa: F403

__all__ = [  # noqa: F405
           'BKTensor', 'CTYPE', 'DEVICE', 'FTYPE', 'MAX_QUBITS', 'TENSOR',
           'TL', 'TensorLike', 'absolute', 'arccos', 'astensor',
           'ccast', 'cis', 'conj', 'cos', 'diag', 'evaluate', 'exp', 'fcast',
           'gpu_available', 'imag', 'inner', 'minimum',
           'outer', 'matmul',
           'rank', 'real', 'reshape', 'set_random_seed', 'sin',
           'sqrt', 'reduce_sum', 'tensormul', 'trace', 'transpose',
           'getitem', 'astensorproduct', 'productdiag',
           'EINSUM_SUBSCRIPTS', 'einsum',
           '__version__', '__name__']


if SEED is not None:               # pragma: no cover
    np_set_random_seed(SEED)
    set_random_seed(SEED)          # noqa: F405
