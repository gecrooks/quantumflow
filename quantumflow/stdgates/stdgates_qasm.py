# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates specific to QASM
"""

import numpy as np

from .. import tensors
from ..config import CTRL
from ..ops import StdGate
from ..paulialgebra import Pauli, sZ
from ..qubits import Qubit
from ..tensors import QubitTensor
from ..utils import cached_property
from ..var import Variable
from .stdgates_1q import PhaseShift, Rz
from .stdgates_2q import CNot
from .stdgates_forest import CPhase

__all__ = ("U3", "U2", "CU3", "CRZ", "RZZ")


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
    _diagram_labels = [CTRL, "U3({theta}, {phi}, {lam})"]

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


# end class class CU3


class CRZ(StdGate):
    r"""A controlled RZ gate."""
    cv_tensor_structure = "diagonal"
    _diagram_labels = [CTRL, "Rz({theta})"]

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        (theta,) = self.params
        q0, q1 = self.qubits
        return Rz(theta, q1).hamiltonian * (1 - sZ(q0)) / 2

    @cached_property
    def tensor(self) -> QubitTensor:
        (theta,) = self.params
        q0, q1 = self.qubits
        gate = Rz(theta, q1)
        from ..modules import ControlGate

        return ControlGate([q0], gate).tensor

    @property
    def H(self) -> "CRZ":
        return self ** -1

    def __pow__(self, t: Variable) -> "CRZ":
        theta = self.param("theta")
        return CRZ(theta * t, *self.qubits)


# end class CRZ


# TODO: Check proper phase, so can add Hamiltonian.
# How is this different from CRZ?
class RZZ(StdGate):
    """A two-qubit ZZ-rotation gate, as defined by QASM.
    Same as ZZ(theta/pi), up to phase.
    """

    cv_interchangeable = True
    cv_tensor_structure = "diagonal"

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        (theta,) = self.params
        q0, q1 = self.qubits
        from ..circuits import Circuit

        circ = Circuit([CNot(q0, q1), PhaseShift(theta, q1), CNot(q0, q1)])
        return circ.asgate().tensor

    @property
    def H(self) -> "RZZ":
        return self ** -1

    def __pow__(self, e: Variable) -> "RZZ":
        (theta,) = self.params
        return RZZ(theta * e, *self.qubits)


# end class RZZ


# fin
