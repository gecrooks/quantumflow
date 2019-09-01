
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


QASM GATES
**********
.. autoclass::  U3
.. autoclass::  U2
.. autoclass::  U1
.. autoclass::  CU3
.. autoclass::  CRZ
.. autoclass::  RZZ


"""
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
    I, X, Y, Z, H, S, T, PHASE, RX, RY, RZ,
    RN, TX, TY, TZ, TH, ZYZ, S_H, T_H, V, V_H, W, TW,
    cliffords)
from .gates_two import (CZ, CNOT, SWAP, ISWAP, CPHASE00, CPHASE01, CPHASE10,
                        CPHASE, PSWAP, CAN, XX, YY, ZZ, PISWAP, EXCH, CTX,
                        BARENCO, CV, CV_H, CY, CH, FSIM)
from .gates_three import (CCNOT, CSWAP, CCZ)
from .gates_qasm import (U3, U2, U1, CU3, CRZ, RZZ)

# TODO: Move cliffords to gate_utils?

# DOCME
GATESET = frozenset([
    # one qubit
    I, X, Y, Z, H, S, T, PHASE, RX, RY, RZ,
    RN, TX, TY, TZ, TH, ZYZ, S_H, T_H, V, V_H, W, TW,
    # two qubit
    CZ, CNOT, SWAP, ISWAP, CPHASE00, CPHASE01, CPHASE10,
    CPHASE, PSWAP, CAN, XX, YY, ZZ, PISWAP, EXCH, CTX,
    BARENCO, CV, CV_H, CY, CH, FSIM,
    # three qubit
    CCNOT, CSWAP, CCZ,
    # qasm gates
    U3, U2, U1, CU3, CRZ, RZZ
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
