
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow pyQuil imports

This is a temporary workaround to simplify pyquil importation. Instead
of `from pyquil.gates import SWAP`, use `from quantumflow import pyquil`,
then `pyquil.SWAP(...)`.

Importation should be simplified in pyQuil 2.0
(https://github.com/rigetti/pyquil/issues/302)
"""

from pyquil import *                                        # noqa: F401,F403


from pyquil.quilbase import (                               # noqa: F401
    Halt,
    ClassicalEqual,
    ClassicalLessThan,
    ClassicalLessEqual,
    ClassicalGreaterThan,
    ClassicalGreaterEqual,
    ClassicalAdd,
    ClassicalMul,
    ClassicalDiv,
    ClassicalSub,
    ClassicalAnd,
    ClassicalExchange,
    ClassicalInclusiveOr,
    ClassicalMove,
    ClassicalNeg,
    ClassicalNot,
    ClassicalExclusiveOr,
    Jump,
    JumpTarget,
    JumpWhen,
    JumpUnless,
    Pragma,
    Measurement,
    Nop,
    Wait,
    Reset,
    ResetQubit,
    ClassicalStore,
    ClassicalLoad,)


from pyquil.gates import (                                  # noqa: F401
    CCNOT, CNOT, CPHASE, CPHASE00, CPHASE01, CPHASE10, CSWAP, CZ, H, I, ISWAP,
    PHASE, PSWAP, RX, RY, RZ, S, SWAP, T, X, Y, Z, STANDARD_GATES)

from pyquil.gates import (                                  # noqa: F401
    MEASURE, HALT, NOP, WAIT, RESET, TRUE, FALSE, NOT, AND, OR, MOVE,
    EXCHANGE)

from pyquil.quil import Program                             # noqa: F401
from pyquil.quilatom import Qubit, MemoryReference          # noqa: F401
from pyquil.wavefunction import Wavefunction                # noqa: F401
from pyquil.device import AbstractDevice, NxDevice, gates_in_isa  # noqa: F401
from pyquil.quilbase import *                               # noqa: F401,F403
from pyquil.quilbase import Declare, Gate                   # noqa: F401

from pyquil import api                                      # noqa: F401
from pyquil.api import *                                    # noqa: F401,F403
from pyquil.api._qac import AbstractCompiler                # noqa: F401
from pyquil.api import QuantumComputer                      # noqa: F401
from pyquil.api import ForestConnection                     # noqa: F401
from pyquil.api import QVM                                  # noqa: F401
from pyquil.api import local_qvm                            # noqa: F401

from pyquil.noise import decoherence_noise_with_asymmetric_ro  # noqa: F401

import pyquil.parser as parser                              # noqa: F401
