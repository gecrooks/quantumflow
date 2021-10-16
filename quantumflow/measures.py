# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np

from .base import BaseGate
from .config import ATOL
from .gates import Identity


def fubini_study_fidelity(vector0: np.ndarray, vector1: np.ndarray) -> float:
    inner01 = np.vdot(vector0, vector1)
    inner00 = np.vdot(vector0, vector0)
    inner11 = np.vdot(vector1, vector1)
    ratio = np.absolute(inner01) / np.sqrt(np.absolute(inner00 * inner11))
    fid = np.minimum(ratio, 1.0)
    return float(np.real(fid))


def fubini_study_close(
    vector0: np.ndarray, vector1: np.ndarray, atol: float = ATOL
) -> bool:
    return 1 - fubini_study_fidelity(vector0, vector1) <= atol


def gates_close(gate0: BaseGate, gate1: BaseGate, atol: float = ATOL) -> bool:
    gate1 = gate1.permute(gate0.qubits)
    return fubini_study_close(gate0.operator, gate1.operator, atol)


def almost_identity(gate: BaseGate, atol: float = ATOL) -> bool:
    return gates_close(gate, Identity(gate.qubits), atol)
