
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates specific to QASM
"""


from numpy import pi

from .. import backend as bk
from ..ops import Gate
from ..qubits import Qubit

from ..utils import cached_property
from ..variables import Variable
from ..config import CTRL
from ..paulialgebra import Pauli, sZ

from .gates_utils import control_gate
from .gates_one import RZ, PhaseShift
from .gates_two import CNOT
from .gates_forest import CPHASE


__all__ = ('U3', 'U2', 'CU3', 'CRZ', 'RZZ')


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
CU1 = CPHASE


class U3(Gate):
    r"""The U3 single qubit gate from QASM.
    The U2 gate is the U3 gate with theta=pi/2. The U1 gate has theta=phi=0,
    which is the same as a PhaseShift gate.

    ..math::
        \text{U3}(\theta, \phi, \lambda) = R_z(\phi) R_y(\theta) R_z(\lambda)

    Refs:
        https://arxiv.org/pdf/1707.03429.pdf (Eq. 2)
        https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/u3.py
    """
    def __init__(self,
                 theta: Variable,
                 phi: Variable,
                 lam: Variable,
                 q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta, phi=phi, lam=lam),
                         qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, phi, lam = self.parameters()

        ctheta = bk.ccast(theta)
        unitary = [[bk.cos(ctheta / 2.0),
                    -bk.sin(ctheta / 2.0) * bk.exp(1j*lam)],
                   [bk.sin(ctheta / 2.0) * bk.exp(1j*phi),
                    bk.cos(ctheta / 2.0) * bk.exp(1j*(phi+lam))]]
        return bk.astensorproduct(unitary)

        # Alternative definition from paper
        # Differs by a phase from definition above, but this matches
        # definition of CU3 in qsikit (but not definition of CU3 in paper.)
        # circ = Circuit()
        # circ += RZ(lam)
        # circ += RY(theta)
        # circ += RZ(phi)
        # return circ.asgate().tensor

    @property
    def H(self) -> 'U3':
        theta, phi, lam = self.parameters()
        return U3(-theta, -lam, -phi, *self.qubits)

    # TODO: __pow__


class U2(Gate):
    """A 'single pulse' 1-qubit gate defined in QASM"""
    def __init__(self, phi: Variable, lam: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(phi=phi, lam=lam),
                         qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        phi, lam = self.parameters()
        return U3(pi/2, phi, lam).tensor

    @property
    def H(self) -> 'U3':
        phi, lam = self.parameters()
        return U3(-pi/2, -lam, -phi, *self.qubits)

    # TODO: __pow__


class CU3(Gate):
    r"""The controlled U3 gate, as defined by QASM.

    Ref:
        https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/cu3.py
    """
    _diagram_labels = [CTRL, 'U3({theta}, {phi}, {lam})']

    def __init__(self,
                 theta: Variable,
                 phi: Variable,
                 lam: Variable,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta, phi=phi, lam=lam),
                         qubits=[q0, q1])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        q0, q1 = self.qubits
        theta, phi, lam = self.parameters()
        # Note: Gate is defined via this circuit in QASM
        # Except for first line, which was added to qsikit to make the
        # definitions of cu3 and u3 in qsikit consistent.
        # https://github.com/Qiskit/qiskit-terra/pull/2755
        # That's silly. They should have fixed the phase of u3 to match
        # the definition in the QASM paper, not change the cu3 gate to
        # something entirely different.
        from ..circuits import Circuit
        circ = Circuit([RZ((lam+phi)/2, q0),
                        RZ((lam-phi)/2, q1),
                        CNOT(q0, q1),
                        U3(-theta / 2, 0, -(phi+lam)/2, q1),
                        CNOT(q0, q1),
                        U3(theta / 2, phi, 0, q1)])
        return circ.asgate().tensor

    @property
    def H(self) -> 'CU3':
        theta, phi, lam = self.parameters()
        return CU3(-theta, -lam, -phi, *self.qubits)


class CRZ(Gate):
    r"""A controlled RZ gate.
    """
    diagonal = True
    _diagram_labels = [CTRL, 'Rz({theta})']

    def __init__(self,
                 theta: Variable,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        theta, = self.parameters()
        q0, q1 = self.qubits
        return RZ(theta, q1).hamiltonian * (1-sZ(q0))/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, = self.parameters()
        q0, q1 = self.qubits
        gate = RZ(theta, q1)
        return control_gate(q0, gate).tensor

    @property
    def H(self) -> 'CRZ':
        return self ** -1

    def __pow__(self, t: Variable) -> 'CRZ':
        theta, = self.parameters()
        return CRZ(theta * t, *self.qubits)


# TODO: Check proper phase, so can add Hamiltonian.
# How's this different from CRZ!?
class RZZ(Gate):
    """A two-qubit ZZ-rotation gate, as defined by QASM.
    Same as ZZ(theta/pi), up to phase.
    """
    interchangeable = True
    diagonal = True

    def __init__(self,
                 theta: Variable,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, = self.parameters()
        q0, q1 = self.qubits
        from ..circuits import Circuit
        circ = Circuit([CNOT(q0, q1), PhaseShift(theta, q1), CNOT(q0, q1)])
        return circ.asgate().tensor

    @property
    def H(self) -> 'RZZ':
        return self ** -1

    def __pow__(self, e: Variable) -> 'RZZ':
        theta, = self.parameters()
        return RZZ(theta * e, *self.qubits)
