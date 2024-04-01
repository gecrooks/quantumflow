# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
.. contents:: :local:
.. currentmodule:: gateset

Collections of gates and operations
###################################

.. autodata:: CD_DRIVE
   :annotation:


"""

from typing import Set, Type

from .gates import ControlGate
from .ops import Gate, Operation
from .stdgates import (
    CCZ,
    CH,
    CS,
    CT,
    CU3,
    CV,
    CV_H,
    CY,
    CZ,
    ECP,
    S_H,
    T_H,
    U2,
    U3,
    V_H,
    XX,
    XY,
    YY,
    ZZ,
    B,
    Barenco,
    Can,
    CCiX,
    CCNot,
    CCXPow,
    CNot,
    CNotPow,
    CPhase,
    CPhase00,
    CPhase01,
    CPhase10,
    CRz,
    CSwap,
    CYPow,
    CZPow,
    Deutsch,
    FSim,
    H,
    HPow,
    I,
    ISwap,
    Ph,
    PhaseShift,
    PSwap,
    Rx,
    Ry,
    Rz,
    Rzz,
    S,
    SqrtISwap,
    SqrtISwap_H,
    SqrtSwap,
    SqrtY,
    SqrtY_H,
    Swap,
    Sycamore,
    T,
    V,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
)
from .stdops import Measure, Project0, Project1, Reset

__all__ = (
    "CIRQ_GATES",
    "LATEX_OPERATIONS",
    "TERMINAL_GATES",
    "QSIM_GATES",
    "QUIL_GATES",
    "QISKIT_GATES",
    "QUIRK_GATES",
)


BRAKET_GATES: Set[Type[Gate]] = set(
    [
        CCNot,
        CNot,
        CPhase,
        CPhase00,
        CPhase01,
        CPhase10,
        CSwap,
        CY,
        CZ,
        H,
        I,
        ISwap,
        PSwap,
        PhaseShift,
        Rx,
        Ry,
        Rz,
        S,
        S_H,
        Swap,
        T,
        T_H,
        V,
        V_H,
        X,
        XX,
        XY,
        Y,
        YY,
        Z,
        ZZ,
    ]
)
"""Set of QuantumFlow gates that are supported by BraKet"""


CIRQ_GATES: Set[Type[Gate]] = set(
    [
        I,
        X,
        Y,
        Z,
        S,
        T,
        H,
        XPow,
        YPow,
        ZPow,
        CZ,
        Swap,
        ISwap,
        CNot,
        XX,
        YY,
        ZZ,
        CCNot,
        CSwap,
        CCZ,
        FSim,
    ]
)
"""Set of QuantumFlow gates that are supported by Cirq"""


LATEX_OPERATIONS: Set[Type[Operation]] = set(
    [
        I,
        X,
        Y,
        Z,
        H,
        T,
        S,
        T_H,
        S_H,
        V,
        V_H,
        Rx,
        Ry,
        Rz,
        SqrtY,
        SqrtY_H,
        XPow,
        YPow,
        ZPow,
        HPow,
        CNot,
        CZ,
        Swap,
        ISwap,
        PSwap,
        CV,
        CS,
        CV_H,
        CPhase,
        CH,
        Can,
        CCNot,
        CSwap,
        CCZ,
        CCiX,
        Deutsch,
        CCXPow,
        XX,
        YY,
        ZZ,
        Can,
        Project0,
        Project1,
        Reset,
        #     NoWire,  # FIXME
        Measure,
        Ph,
        ECP,
        SqrtISwap_H,
        Barenco,
        CNotPow,
        CYPow,
        CZPow,
        B,
        CY,
        ECP,
        Sycamore,
        XY,
        ControlGate,
    ]
)
"""Set of QuantumFlow operations that can be converted to LaTeX circuit diagrams"""


TERMINAL_GATES: Set[Type[Gate]] = set([I, Ph, X, Y, Z, S, T, H, XPow, YPow, ZPow, CNot])
"""Default set of standard gates for gate decompositions"""

# Note missing Swap, ...
QSIM_GATES: Set[Type[Gate]] = set(
    [I, X, Y, Z, S, T, H, XPow, YPow, ZPow, CZ, ISwap, CNot, FSim]
)
"""Set of QuantumFlow gates that are supported by QSIM"""


QUIL_GATES: Set[Type[Gate]] = set(
    [
        I,
        X,
        Y,
        Z,
        H,
        S,
        T,
        PhaseShift,
        Rx,
        Ry,
        Rz,
        CZ,
        CNot,
        Swap,
        ISwap,
        CPhase00,
        CPhase01,
        CPhase10,
        CPhase,
        PSwap,
        CCNot,
        CSwap,
    ]
)
"""Set of QuantumFlow gates that are supported by quil"""


QUIRK_GATES: Set[Type[Gate]] = set(
    [
        I,
        H,
        X,
        Y,
        Z,
        V,
        V_H,
        SqrtY,
        SqrtY_H,
        S,
        S_H,
        T,
        T_H,
        CNot,
        CY,
        CZ,
        Swap,
        CSwap,
        CCNot,
        CCZ,
        Rx,
        Ry,
        Rz,
        XPow,
        YPow,
        ZPow,
    ]
)

QUTIP_GATES: Set[Type[Gate]] = set(
    [
        I,
        Ph,
        Rx,
        Ry,
        Rz,
        V,
        H,
        PhaseShift,
        CY,
        CZ,
        S,
        T,
        X,
        Y,
        Z,
        CS,
        CT,
        CNot,
        Swap,
        ISwap,
        SqrtSwap,
        SqrtISwap,
        CCNot,
        CSwap,
        CPhase,
    ]
)

# QUTIP_GATES: Tuple[Type[Gate], ...] = tuple(_QUTIP_GATE_NAMES.keys())
# """List of QuantumFlow gates that we know how to convert directly to and from QuTiP"""


QISKIT_GATES: Set[Type[Gate]] = set(
    [
        CCNot,
        CH,
        CRz,
        CSwap,
        CPhase,
        CU3,
        CV,
        CNot,
        CY,
        CZ,
        H,
        I,
        Rx,
        Ry,
        Rz,
        Rzz,
        S,
        S_H,
        Swap,
        V,
        T,
        T_H,
        PhaseShift,
        U2,
        U3,
        X,
        Y,
        Z,
    ]
)
"""Set of QuantumFlow gates that are supported by qiskit and QASM"""


# fin
