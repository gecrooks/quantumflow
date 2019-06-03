# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Configuration
"""

import os
import random


ENV_PREFIX = 'QUANTUMFLOW_'  # Environment variable prefix


# ==== Version number ====
try:
    from quantumflow.version import version
except ImportError:                           # pragma: no cover
    # package is not installed
    version = "?.?.?"


# ==== TOLERANCE ====
TOLERANCE = 1e-6
"""Tolerance used in various floating point comparisons"""


# ==== Random Seed ====
_ENVSEED = os.getenv(ENV_PREFIX + 'SEED', None)
SEED = int(_ENVSEED) if _ENVSEED is not None else None
if SEED is not None:
    random.seed(SEED)  # pragma: no cover
