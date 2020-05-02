
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
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
.. autofunction:: print_gate
.. autofunction:: almost_unitary
.. autofunction:: almost_identity
.. autofunction:: almost_hermitian
.. autodata:: NAMED_GATES
.. autodata:: STD_GATESET


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
.. autoclass:: PhaseShift
.. autoclass:: RX
.. autoclass:: RY
.. autoclass:: RZ
.. autoclass:: Ph

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
.. autoclass:: V
.. autoclass:: V_H
.. autoclass:: PhasedX
.. autoclass:: PhasedXPow


Two-qubit gates
***************
.. autoclass:: CAN
.. autoclass:: XX
.. autoclass:: YY
.. autoclass:: ZZ
.. autoclass:: XY
.. autoclass:: EXCH
.. autoclass:: Barenco
.. autoclass:: CY
.. autoclass:: CH
.. autoclass:: CNotPow
.. autoclass:: SqrtISwap
.. autoclass:: SqrtISwap_H
.. autoclass:: SqrtSwap
.. autoclass:: SqrtSwap_H


Three-qubit gates
*****************
.. autoclass:: CZ
.. autoclass:: Deutsch
.. autoclass:: CCiX



Multi-qubit gates
*****************
.. autoclass:: IDEN
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

==========================  =========== =============== =========== =========== ===========
Description                 QF          Cirq            QASM/qsikit PyQuil      Pennylane
==========================  =========== =============== =========== =========== ===========
Identity  (single qubit)    I           I               id or iden  I
Identity  (multi-qubit)     Iden        IdentityGate
Pauli-X                     X           X               x           X           PauliX
Pauli-Y                     Y           Y               y           Y           PauliY
Pauli-Z                     Z           Z               z           Z           PauliZ
Hadamard                    H           H               h           H           Hadamard
X-rotations                 RX          rx              rx          RX          RX
Y-rotations                 RY          ry              ry          RY          RY
Z-rotations                 RZ          rz              rz          RZ          RZ
Sqrt of Z                   S           S               s           S           S
Sqrt of S                   T           T               t           T           T
Phase shift                 PhaseShift  .               u1          PHASE       PhaseShift
Bloch rotations             RN          .               .           .           .
Powers of X                 XPow        XPowGate        .           .           .
Powers of Y                 YPow        YPowGate        .           .           .
Powers of Z                 ZPow        ZPowGate        .           .           .
Powers of Hadamard          HPow        HPowGate        .           .           .
Inv. of S                   S_H         .               sdg         .           .
Inv. of T                   T_H         .               tdg         .           .
Sqrt of X                   V           .               .           .           .
Inv. sqrt of X              V_H         .               .           .           .

Powers of X⊗X               XX          XXPowGate       .           .           .
Powers of Y⊗Y               YY          YYPowGate       .           .           .
Powers of Z⊗Z               ZZ          ZZPowGate       .           .           .
Canonical                   Can         .               .           .           .
Controlled-Not              CNOT        CNOT            cx          CNOT        CNOT
Controlled-Z                CZ          CZ              cz          CZ          CZ
Controlled-Y                CY          .               cy          .           .
Controlled-Hadamard         CH          .               ch          .           .
Controlled-V                CV          .               .           .           .
Controlled-inv-V            CV_H        .               .           .           .
Powers of CNOT				CXPow		CNotPowGate		.			.			.
Powers of CY				CYPow			.				.			.			.
Powers of CZ				CZPow		CZPowGate		.			.			.
Swap                        Swap        SWAP            swap        .           SWAP
Exchange                    Exch        SwapPowGate (*) .           .
iSwap                       ISwap       ISWAP           .           ISWAP       .
XY (powers of iSwap)        XY          ISwapPowGate(*) .           XY(*)       .
Givens rotation				Givens		givens			.			.			.
Barenco                     Barenco     .               .           .           .
B (Berkeley)                B           .               .           .           .
Sqrt-iSWAP                  SqrtISwap   .               .           .           .
Inv. of sqrt-iSWAP          SqrtISwap_H .               .           .           .
Sqrt-SWAP                   SqrtSwap    .               .           .           .
Inv. of sqrt-SWAP           SqrtSwap_H  .               .           .           .
ECP                         ECP         .               .           .           .
W (Dual-rail Hadamard)      W           .               .           .           .

Toffoli                     CCNOT       CCX             ccx         CCNOT       Toffoli
Fredkin                     CSwap       CSWAP           cswap       CSWAP       CSWAP
Controlled-Controlled-Z     CCZ         CCZ             .           .           .
Deutsch                     Deutsch     .               .           .           .
Powers of CCNOT             CCXPow    	CCXPowGate      .           .           .
Powers of CCZ               .           CCZPowGate      .           .           .

* Forest specific gates
Controlled-phase            CPhase      CZPowGate(*)    cu1         CPHASE      .
Controlled-phase on 00      CPhase00    .               .           CPHASE00    .
Controlled-phase on 01      CPhase01    .               .           CPHASE01    .
Controlled-phase on 10      CPhase10    .               .           CPHASE10    .
Parametric Swap             PSwap       .               .           PSWAP       .

* Cirq specific gates
Fermionic-Simulation        FSim        FSimGate        .           .           .
Phased-X gate               PhasedX     .               .           .           .
Powers of Phased-X gate     PhasedXPow  PhasedXPowGate  .           .           .
Sycamore                    Sycamore    Sycamore        .           .           .
Fermionic swap              FSwap       FSwap           .           .           .
Powers of fermionic swap    FSwapPow    FSwapPow        .           .           .

* QASM/qiskit gates
QASM's U3 gate              U3          .               u3          .           Rot
QASM's U2 gate              U2          .               u2          .
QASM's controlled-U3        CU3         .               cu3         .           .
QASM's ZZ-rotations         RZZ         .               rzz         .           CRot
Controlled-RX               .           .               .           .           CRX
Controlled-RY               .           .               .           .           CRY
Controlled-RZ               CRZ         .               crz                     CRZ
==========================  =========== =============== =========== =========== ===========

(*) Modulo differences in parametrization

(**) Cirq defines XX, YY, and ZZ gates, as XX^1, YY^1, ZZ^1 which are direct
products of single qubit gates.
(e.g. XX(t,0,1) is the same as X(0) X(1) when t=1)


"""  # noqa: E501
# Note: see comments in gates_qasm.py to understand the QASM gate mappings.


from .gates_utils import *      # noqa: F401, F403

from .gates_one import *        # noqa: F401, F403
from .gates_two import *        # noqa: F401, F403
from .gates_three import *      # noqa: F401, F403

from .gates_cirq import *       # noqa: F401, F403
from .gates_forest import *     # noqa: F401, F403
from .gates_qasm import *       # noqa: F401, F403

from .gates_one import __all__ as _gates_one
from .gates_two import __all__ as _gates_two
from .gates_three import __all__ as _gates_three

from .gates_cirq import __all__ as _gates_cirq
from .gates_forest import __all__ as _gates_forest
from .gates_qasm import __all__ as _gates_qasm


_gates = (_gates_one + _gates_two + _gates_three + _gates_qasm
          + _gates_cirq + _gates_forest)

NAMED_GATES = {name: globals()[name] for name in _gates}
del NAMED_GATES['IDEN']  # FIXME: Removed because not a StdGate 
"""
A mapping between gate names and gate classes
"""

STD_GATESET = frozenset(NAMED_GATES.values())
"""
A set of all the standard gates classes.
"""
