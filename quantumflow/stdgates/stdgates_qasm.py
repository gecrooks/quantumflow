# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates specific to QASM

Qiskit has two names for each gate: the name of the class object (with a "Gate" suffix)
and the lower cased abbreviations that added as methods to the QuantumCircuit class and
QASM files.

## Standard gates (as of qiskit v0.26.0)
https://qiskit.org/documentation/apidoc/circuit_library.html

============    ============    ============    ============================================================================================================
Qiskit          QASM            QF              Comments
============    ============    ============    ============================================================================================================

C3X                                             A triply-controlled X gate
C3SX                                            A triply controlled V gate
C4X                                             A quadruply-controlled not gate
CCX             ccx             CCNot
DCX             dcx                             A Double-CNOT
CH              ch              CH
CPhase          cp              CPhase
CRX             crz             CRx
CRY             cry             CRy
CRZ             crz             CRz
CSwap           cswap           CSwap
CSX             csx             CV
CU
CU1             cu1             CPhase          Replaced by CPhase as of qiskit 0.16.0
CU3             cu3             CU3             Deprecated.
CX              cx              CNot
CY              cy              CY
CZ              cz              CZ
H               h               H
I               id              I
MS              ms              -               Deprecated. Essentially an Rxx gate
Phase           p               PhaseShift
RCCX                            -               Claims to be a Margolus gate, but it ain't. Some other generic "simplified" Toffoli gate.
RC3X                            -               A triply controlled "simplified Toffoli" style gate.
R
RX              rx              Rx
RXX             rxx             Rxx
RY              ry              Ry
RYY             ryy             Ryy
RZ              rz              Rz
RZZ             rzz             Rzz
RZX             rzx                             Special case of the cross resonance gate
ECR             ecr                             Echoed cross-resonance gate
S               s               S
Sdg             sdg             S_H
Swap            swap            Swap
iSwap           iswap           iswap
SX              sx              V
SXdg            sxdg            V_H
T               t               T
Tdg             tdg             T_H
U               u                               A Z-Y-Z Euler decomposition of a 1-qubit gate with different arguments
U1              u1              PhaseShift      Replaced by Phase as of qiskit 0.16.0
U2              u2              U2              Deprecated. Replaced by 'U'
U3              u3              U3
X               x               X
Y               y               Y
Z               z               Z
============    ============    ============    ============================================================================================================


## Multi-qubit gates
MCPhase
MCXGate
MCXGrayCode
NCXRecursive
NCXVChain

## Standard gate like operations
Barrier
Measure
Reset


"""  # noqa: E501
# Is QASM's U1 gate a PhaseShift gate or an RZ gate?
# This is very confusing. In the QASM paper U1(lam) is defined as both
# PhaseShift(lam) (Eq. 3) and as U3(0,0, lam), which is RZ(lam) (Bottom of
# page 10). https://arxiv.org/pdf/1707.03429.pdf
# The same happens in the code.
# https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/u1.py
# U1._define() assumes U1 is RZ, but U1._matrix() gives the PhaseShift gate.
# In code and paper, RZ is defined as U1, so it's even not clear if QASM's RZ
# is our PhaseShift or RZ!
# Phase shift and RZ only differ by a global phase, so this doesn't matter most
# of the time. An exception is constructing controlled gates. Looking at QASM's
# control gates, crz is in fact a controlled-RZ, and cu1 is a
# controlled-PhaseShift! (up to global phase).
# Therefore, QASM's u1 is a PhaseShift gate, and rz is an RZ gate,
# cu1 is a CPHASE.
# u3 and cu3 are even more of a nightmare. See notes below.

# Update: qiskit 0.16.0 renames and clarifies some of these issues.
# U1 is now Phase or in circuits 'p' (i.e. PhaseShift gate)
# CU1 is now CPhase or in circuits 'cp'

from typing import List

import numpy as np

from .. import tensors
from ..config import CTRL
from ..future import cached_property
from ..qubits import Qubit
from ..tensors import QubitTensor
from ..var import Variable
from .stdgates import StdCtrlGate, StdGate
from .stdgates_1q import PhaseShift, Rx, Ry, Rz
from .stdgates_2q import XX, YY, ZZ, CNot
from .stdgates_forest import CPhase

__all__ = ("U3", "U2", "CU3", "CRZ", "RZZ", "CRx", "CRy", "CRz", "Rxx", "Ryy", "Rzz")


U1 = PhaseShift
CU1 = CPhase


class U3(StdGate):
    r"""The U3 single qubit gate from QASM.
    The U2 gate is the U3 gate with theta=pi/2. The U1 gate has theta=phi=0,
    which is the same as a PhaseShift gate.

    ..math::
        \text{U3}(\theta, \phi, \lambda) = R_z(\phi) R_y(\theta) R_z(\lambda)

    Refs:
        https://arxiv.org/pdf/1707.03429.pdf (Eq. 2)
        https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/u3.py
    """

    def __init__(
        self, theta: Variable, phi: Variable, lam: Variable, q0: Qubit
    ) -> None:
        super().__init__(params=[theta, phi, lam], qubits=[q0])

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = self.float_param("theta")
        phi = self.float_param("phi")
        lam = self.float_param("lam")

        unitary = [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0) * np.exp(1j * lam)],
            [
                np.sin(theta / 2.0) * np.exp(1j * phi),
                np.cos(theta / 2.0) * np.exp(1j * (phi + lam)),
            ],
        ]
        return tensors.asqutensor(unitary)

        # Alternative definition from paper
        # Differs by a phase from definition above, but this matches
        # definition of CU3 in qsikit (but not definition of CU3 in paper.)
        # circ = Circuit()
        # circ += RZ(lam)
        # circ += RY(theta)
        # circ += RZ(phi)
        # return circ.asgate().tensor

    @property
    def H(self) -> "U3":
        theta, phi, lam = self.params
        return U3(-theta, -lam, -phi, *self.qubits)


# end class U3


class U2(StdGate):
    """A 'single pulse' 1-qubit gate defined in QASM"""

    def __init__(self, phi: Variable, lam: Variable, q0: Qubit) -> None:
        super().__init__(params=[phi, lam], qubits=[q0])

    @cached_property
    def tensor(self) -> QubitTensor:
        phi, lam = self.params
        return U3(np.pi / 2, phi, lam, q0=0).tensor

    @property
    def H(self) -> "U3":
        phi, lam = self.params
        return U3(-np.pi / 2, -lam, -phi, *self.qubits)


# end class U2


class CU3(StdGate):
    r"""The controlled U3 gate, as defined by QASM.

    Ref:
        https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/cu3.py
    """

    def __init__(
        self,
        theta: Variable,
        phi: Variable,
        lam: Variable,
        q0: Qubit,
        q1: Qubit,
    ) -> None:
        super().__init__(params=[theta, phi, lam], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = self.float_param("theta")
        phi = self.float_param("phi")
        lam = self.float_param("lam")

        # Note: Gate is defined via this circuit in QASM
        # Except for first line, which was added to qsikit to make the
        # definitions of cu3 and u3 in qsikit consistent.
        # https://github.com/Qiskit/qiskit-terra/pull/2755
        # That seems silly. They should have fixed the phase of u3 to match
        # the definition in the QASM paper, not change the cu3 gate to
        # something entirely different.
        from ..circuits import Circuit

        circ = Circuit(
            [
                PhaseShift((lam + phi) / 2, 0),
                PhaseShift((lam - phi) / 2, 1),
                CNot(0, 1),
                U3(-theta / 2, 0.0, -(phi + lam) / 2, 1),
                CNot(0, 1),
                U3(theta / 2, phi, 0.0, 1),
            ]
        )

        return circ.asgate().tensor

    @property
    def H(self) -> "CU3":
        theta, phi, lam = self.params
        return CU3(-theta, -lam, -phi, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [CTRL, "U3({theta}, {phi}, {lam})"]


# end class class CU3


class CRx(StdCtrlGate):
    r"""A controlled Rx gate."""

    cv_target = Rx

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def H(self) -> "CRx":
        return self**-1

    def __pow__(self, t: Variable) -> "CRx":
        theta = self.param("theta")
        return CRx(theta * t, *self.qubits)


# end class CRx


class CRy(StdCtrlGate):
    r"""A controlled Ry gate."""

    cv_target = Ry

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def H(self) -> "CRy":
        return self**-1

    def __pow__(self, t: Variable) -> "CRy":
        theta = self.param("theta")
        return CRy(theta * t, *self.qubits)


# end class CRy


class CRz(StdCtrlGate):
    r"""A controlled Rz gate."""

    cv_target = Rz
    cv_tensor_structure = "diagonal"

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def H(self) -> "CRz":
        return self**-1

    def __pow__(self, t: Variable) -> "CRz":
        theta = self.param("theta")
        return CRz(theta * t, *self.qubits)


# end class CRz


# Legacy
CRZ = CRz


class Rxx(StdGate):
    """A two-qubit XX-rotation gate, as defined by QASM.
    Same as XX(theta/pi), up to phase.
    """

    cv_interchangeable = True

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        q0, q1 = self.qubits
        (theta,) = self.params
        return XX(theta / np.pi, q0, q1).tensor

    @property
    def H(self) -> "Rxx":
        return self**-1

    def __pow__(self, e: Variable) -> "Rxx":
        (theta,) = self.params
        return Rxx(theta * e, *self.qubits)


# end class Ryy


class Ryy(StdGate):
    """A two-qubit YY-rotation gate, as defined by QASM.
    Same as YY(theta/pi)
    """

    cv_interchangeable = True

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        q0, q1 = self.qubits
        (theta,) = self.params
        return YY(theta / np.pi, q0, q1).tensor

    @property
    def H(self) -> "Ryy":
        return self**-1

    def __pow__(self, e: Variable) -> "Ryy":
        (theta,) = self.params
        return Ryy(theta * e, *self.qubits)


# end class Rxx


# How is this different from CRZ?
class Rzz(StdGate):
    """A two-qubit ZZ-rotation gate, as defined by QASM.
    Same as ZZ(theta/pi).
    """

    cv_interchangeable = True
    cv_tensor_structure = "diagonal"

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        q0, q1 = self.qubits
        (theta,) = self.params
        return ZZ(theta / np.pi, q0, q1).tensor

    @property
    def H(self) -> "Rzz":
        return self**-1

    def __pow__(self, e: Variable) -> "Rzz":
        (theta,) = self.params
        return Rzz(theta * e, *self.qubits)


# end class Rzz

# Legacy
RZZ = Rzz


# fin
