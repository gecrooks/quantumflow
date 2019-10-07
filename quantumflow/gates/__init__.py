
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


# QuantumFlow Gates and actions on gates.

"""
.. contents:: :local:
.. currentmodule:: quantumflow


Gate objects
############
.. autoclass:: Gate
    :members:

Actions on gates
#################
.. autofunction:: join_gates
.. autofunction:: control_gate
.. autofunction:: conditional_gate
.. autofunction:: almost_unitary
.. autofunction:: print_gate


Standard gates
##############

Standard one-qubit gates
************************
.. autoclass:: I
.. autoclass:: X
.. autoclass:: Y
.. autoclass:: Z
.. autoclass:: H
.. autoclass:: S
.. autoclass:: T
.. autoclass:: PHASE
.. autoclass:: RX
.. autoclass:: RY
.. autoclass:: RZ


Standard two-qubit gates
************************
.. autoclass:: CZ
.. autoclass:: CNOT
.. autoclass:: SWAP
.. autoclass:: ISWAP
.. autoclass:: CPHASE00
.. autoclass:: CPHASE01
.. autoclass:: CPHASE10
.. autoclass:: CPHASE
.. autoclass:: PSWAP


Standard three-qubit gates
**************************
.. autoclass:: CCNOT
.. autoclass:: CSWAP



Additional gates
################


One-qubit gates
***************
.. autoclass:: S_H
.. autoclass:: T_H
.. autoclass:: TX
.. autoclass:: TY
.. autoclass:: TZ
.. autoclass:: TH
.. autoclass:: ZYZ
.. autoclass:: P0
.. autoclass:: P1
.. autoclass:: V
.. autoclass:: V_H
.. autoclass:: W
.. autoclass:: TW



Two-qubit gates
***************
.. autoclass:: CAN
.. autoclass:: XX
.. autoclass:: YY
.. autoclass:: ZZ
.. autoclass:: PISWAP
.. autoclass:: EXCH
.. autoclass:: BARENCO
.. autoclass:: CY
.. autoclass:: CH
.. autoclass:: CTX


Multi-qubit gates
*****************
.. autofunction:: identity_gate
.. autofunction:: random_gate


QASM gates
**********
.. autoclass::  U3
.. autoclass::  U2
.. autoclass::  CU3
.. autoclass::  CRZ
.. autoclass::  RZZ


Mapping between APIs
####################
Each of the main quantum computing python APIs (QuantumFlow, Cirq, qsikit
(QASM), and pyQuil) have different gates available and different naming
conventions. The following table maps gate names between these APIs.

==========================  =========== =============== =========== ===========
Description                 QF          Cirq            QASM/qsikit PyQuil
==========================  =========== =============== =========== ===========
Identity                    I           I               id or iden  I
Pauli-X                     X           X               x           X
Pauli-Y                     Y           Y               y           Y
Pauli-Z                     Z           Z               z           Z
Hadamard                    H           H               h           H
X-rotations                 RX          Rx              rx          RX
Y-rotations                 RY          Ry              ry          RY
Z-rotations                 RZ          Rz              rz          RZ
Sqrt of Z                   S           S               s           S
Sqrt of S                   T           T               t           T
Phase-gate                  PHASE       .               u1           PHASE
Bloch rotations             RN          .               .           .
Powers of X                 TX          XPowGate        .           .
Powers of Y                 TY          YPowGate        .           .
Powers of Z                 TZ          ZPowGate        .           .
Powers of Hadamard          TH          HPowGate        .           .
Inv. of S                   S_H         .               sdg         .
Inv. of T                   T_H         .               tdg         .
Sqrt of X                   V           .               .           .
Inv. sqrt of X              V_H         .               .           .
Phased-X gate               W           .               .           .
Powers of W gate            TW          PhasedXPowGate  .           .

ZZ^t                        XX          XXPowGate       .           .
YY^t                        YY          YYPowGate       .           .
ZZ^t                        ZZ          ZZPowGate       .           .
Canonical                   CAN         .               .           .
Controlled-Not              CNOT        CNOT            cx          CNOT
Controlled-Z                CZ          CZ              cz          CZ
Controlled-Y                CY          .               cy          .
Controlled-Hadamard         CH          .               ch          .
Controlled-V                CV          .               .           .
Controlled-inv-V            CV_H        .               .           .
Swap                        SWAP        SWAP            swap        .
Exchange                    EXCH        SwapPowGate (*) .           .
iSwap                       ISWAP       ISWAP           .           ISWAP
parametric-iSwap            PISWAP      ISwapPowGate(*) .           PISWAP
Parametric Swap             PSWAP       .               .           PSWAP
Controlled-phase on 00      CPHASE00    .               .           CPHASE00
Controlled-phase on 01      CPHASE01    .               .           CPHASE01
Controlled-phase on 10      CPHASE10    .               .           CPHASE10
Controlled-phase            CPHASE      CCZPowGate      cu1         CPHASE
Barenco                     BARENCO     .               .           .
Fermionic-Simulation        FSIM        FSimGate        .           .

Toffoli                     CCNOT       CCX             ccx         CCNOT
Fredkin                     CSWAP       CSWAP           cswap       CSWAP
Controlled-Controlled-Z     CCZ         CCZ             .           .
Powers of CCNOT             .           CCXPowGate      .           .
Powers of CCZ               .           CCZPowGate      .           .

QASM's U3 gate              U3          .               u3          .
QASM's U2 gate              U2          .               u2          .
QASM's controlled-U3        CU3         .               cu3         .
QASM's Controlled-RZ        CRZ         .               crz         .
QASM's ZZ-rotations         RZZ         .               rzz         .
==========================  =========== =============== =========== ===========

(*) Modulo differences in parametization

(**) Cirq defines XX, YY, and ZZ gates, as XX^1, YY^1, ZZ^1 which are direct
products of single qubit gates.
(e.g. XX(t,0,1) is the same as X(0) X(1) when t=1)


"""
# Note: see comments in gates_qasm.py to understand the QASM gate mappings.
# TODO: Check that all gates documentated

# Kudos: Standard gate definitions adapted from Nick Rubin's reference-qvm


from .gates_utils import (  # noqa: F401
    identity_gate,
    random_gate,
    join_gates,
    control_gate,
    conditional_gate,
    P0, P1,
    almost_unitary,
    almost_identity,
    almost_hermitian,
    print_gate)
from .gates_one import (   # noqa: F401
    IDEN, I, X, Y, Z, H, S, T, PHASE, RX, RY, RZ,
    RN, TX, TY, TZ, TH, ZYZ, S_H, T_H, V, V_H, W, TW,
    cliffords)
from .gates_two import (CZ, CNOT, SWAP, ISWAP, CPHASE00, CPHASE01, CPHASE10,
                        CPHASE, PSWAP, CAN, XX, YY, ZZ, PISWAP, EXCH, CTX,
                        BARENCO, CV, CV_H, CY, CH, FSIM)
from .gates_three import (CCNOT, CSWAP, CCZ)
from .gates_qasm import (U3, U2, CU3, CRZ, RZZ)

# TODO: Add __all__ ?

# DOCME
GATESET = frozenset([
    # one qubit
    I, X, Y, Z, H, S, T, PHASE, RX, RY, RZ,
    RN, TX, TY, TZ, TH, S_H, T_H, V, V_H, W, TW, ZYZ,
    # two qubit
    CZ, CNOT, SWAP, ISWAP, CPHASE00, CPHASE01, CPHASE10,
    CPHASE, PSWAP, CAN, XX, YY, ZZ, PISWAP, EXCH, CTX,
    BARENCO, CV, CV_H, CY, CH, FSIM,
    # three+ qubit
    CCNOT, CSWAP, CCZ, IDEN,
    # qasm gates
    U3, U2, CU3, CRZ, RZZ
    ])


# DOCME
# TODO: FrozenDict?
NAMED_GATES = {gate_class.__name__: gate_class for gate_class in GATESET}


STD_GATES = NAMED_GATES.values()

QUIL_GATES = {'I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'PHASE',
              'RX', 'RY', 'RZ', 'CZ', 'CNOT', 'SWAP',
              'ISWAP', 'CPHASE00', 'CPHASE01', 'CPHASE10',
              'CPHASE', 'PSWAP', 'CCNOT', 'CSWAP', 'PISWAP'}


# TODO: QASM gate set

# TOOD: Cirq gate set
