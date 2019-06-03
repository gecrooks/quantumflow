
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow's Tensor Library Backend
"""

import os

from ..config import ENV_PREFIX, SEED
from .numpybk import set_random_seed as np_set_random_seed

DEFAULT_BACKEND = 'numpy'
BACKENDS = ('tensorflow', 'tensorflow2', 'eager', 'torch', 'numpy')

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
elif BACKEND == 'torch':                             # pragma: no cover
    from quantumflow.backend.torchbk import *        # noqa: F403
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

# Warning: Using 'sum' as function is weirdly problamatic because of python's
# built in sum(). We use 'sum' for the backend interface becuase it is part of
# numpy's conventions. (Tensoflow uses reduce_sum() instead)
# One problem that can show up randomly is that if you use python's sum()
# anywhere else in quantuflow, then the line above will start causing
# typecheck errors. Workaround is to always use np.sum(), and never
# python's sum. A better solution might be to rename bk.sum().


if SEED is not None:               # pragma: no cover
    np_set_random_seed(SEED)
    set_random_seed(SEED)          # noqa: F405
