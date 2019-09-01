
# QuantumFlow: Gates specific to QASM

from numpy import pi

from .. import backend as bk
from ..ops import Gate
from ..qubits import Qubit
from ..circuits import Circuit

from .gates_utils import control_gate
from .gates_one import RZ, RY
from .gates_two import CNOT

__all__ = ['U3', 'U2', 'U1', 'CU3', 'CRZ', 'RZZ']

# TODO: CU1 which is different from CRZ
# https://arxiv.org/pdf/1707.03429.pdf p12


class U3(Gate):
    r"""The U3 single qubit gate from QASM.
    The U2 gate is the U3 gate with theta=pi/2. The U1 gate has theta=phi=0,
    which is the same as an RZ gate.

    ..math::
        \text{U3}(\theta, \phi, \lambda) = R_z(\phi) R_y(\theta) R_z(\lambda)

    Refs:
        https://arxiv.org/pdf/1707.03429.pdf
        https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/u3.py
    """
    def __init__(self,
                 theta: float,
                 phi: float,
                 lam: float,
                 q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta, phi=phi, lam=lam),
                         qubits=[q0])

    @property
    def tensor(self) -> bk.BKTensor:
        theta, phi, lam = self.params.values()
        q0 = self.qubits[0]
        circ = Circuit()
        circ += RZ(lam, q0)
        circ += RY(theta, q0)
        circ += RZ(phi, q0)
        return circ.asgate().tensor

    @property
    def H(self) -> Gate:
        theta, phi, lam = self.params.values()
        return U3(-theta, -lam, -phi, *self.qubits)


class U2(Gate):
    """A 'single pulse' 1-qubit gate defined in QASM"""
    def __init__(self, phi: float, lam: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(phi=phi, lam=lam),
                         qubits=[q0])

    @property
    def tensor(self) -> bk.BKTensor:
        phi, lam = self.params.values()
        return U3(pi/2, phi, lam).tensor

    @property
    def H(self) -> Gate:
        phi, lam = self.params.values()
        return U3(-pi/2, -lam, -phi, *self.qubits)


class U1(RZ):
    """A diagonal 1-qubit gate defined in QASM. Equivalent to RZ"""
    def __init__(self, lam: float, q0: Qubit = 0) -> None:
        super().__init__(lam, q0)


class CU3(Gate):
    r"""The controlled U3 gate, as defined by QASM.

    Ref:
        https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/cu3.py
    """
    def __init__(self,
                 theta: float,
                 phi: float,
                 lam: float,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta, phi=phi, lam=lam),
                         qubits=[q0, q1])

    @property
    def tensor(self) -> bk.BKTensor:
        q0, q1 = self.qubits
        theta, phi, lam = self.params.values()
        # Note: Gate is defined via this circuit in QASM
        # This does not seem to be the same as a controlled-U3 gate?
        circ = Circuit([U1((lam+phi)/2, q0),
                        U1((lam-phi)/2, q1),
                        CNOT(q0, q1),
                        U3(-theta / 2, 0, -(phi+lam)/2, q1),
                        CNOT(q0, q1),
                        U3(theta / 2, phi, 0, q1)])
        return circ.asgate().tensor

    @property
    def H(self) -> Gate:
        theta, phi, lam = self.params.values()
        return CU3(-theta, -lam, -phi, *self.qubits)


class CRZ(Gate):
    r"""A controlled RZ gate.
    """
    def __init__(self,
                 theta: float,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def tensor(self) -> bk.BKTensor:
        theta, = self.params.values()
        q0, q1 = self.qubits
        gate = RZ(theta, q1)
        return control_gate(q0, gate).tensor

    @property
    def H(self) -> Gate:
        theta, = self.params.values()
        return CRZ(-theta, *self.qubits)


class RZZ(Gate):
    """A two-qubit ZZ-rotation gate, as defined by QASM"""
    # Same as ZZ(theta/pi), up to phase. (TESTME)

    interchangeable = True

    def __init__(self,
                 theta: float,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def tensor(self) -> bk.BKTensor:
        theta, = self.params.values()
        q0, q1 = self.qubits
        circ = Circuit([CNOT(q0, q1), RZ(theta, q1), CNOT(q0, q1)])
        return circ.asgate().tensor

    @property
    def H(self) -> Gate:
        theta = -self.params['theta']
        return RZZ(theta, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta = self.params['theta'] * t
        return RZZ(theta, *self.qubits)
