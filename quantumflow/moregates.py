
# QuantumFlow: Additional gates

import numpy as np
from numpy import pi

from . import backend as bk
from .ops import Gate
from .gates import control_gate
from .qubits import Qubit
from .stdgates import I, TX, RZ, RY, CNOT
from .circuits import Circuit

__all__ = ['BARENCO', 'SX', 'SX_H', 'CY', 'CH',
           'U3', 'U2', 'U1', 'U0', 'CU3', 'CRZ', 'RZZ']


class SX(Gate):
    r"""
    Principal square root of the X gate, X-PLUS-90 gate.
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def tensor(self) -> bk.BKTensor:
        return TX(0.5).tensor

    @property
    def H(self) -> Gate:
        return SX_H(*self.qubits)

    def __pow__(self, t: float) -> Gate:
        return TX(0.5*t)


class SX_H(Gate):
    r"""
    Complex conjugate of the SX gate, X-MINUS-90 gate.
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def tensor(self) -> bk.BKTensor:
        return TX(-0.5).tensor

    @property
    def H(self) -> Gate:
        return SX(*self.qubits)

    def __pow__(self, t: float) -> Gate:
        return TX(-0.5*t)


class BARENCO(Gate):
    """A universal two-qubit gate:

    Ref:
        A Universal Two–Bit Gate for Quantum Computation, A. Barenco (1996)
        https://arxiv.org/pdf/quant-ph/9505016.pdf
    """

    def __init__(self,
                 alpha: float,
                 phi: float,
                 theta: float,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        params = dict(alpha=alpha, phi=phi, theta=theta)
        qubits = [q0, q1]
        super().__init__(params=params, qubits=qubits)

    @property
    def tensor(self) -> bk.BKTensor:
        alpha, phi, theta = self.params.values()

        calpha = bk.ccast(alpha)
        cphi = bk.ccast(phi)
        ctheta = bk.ccast(theta)

        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, bk.cis(calpha) * bk.cos(ctheta),
                    -1j * bk.cis(calpha - cphi) * bk.sin(ctheta)],
                   [0, 0, -1j * bk.cis(calpha + cphi) * bk.sin(ctheta),
                    bk.cis(calpha) * bk.cos(ctheta)]]
        return bk.astensorproduct(unitary)


class CY(Gate):
    r"""A controlled-Y gate

    Equivalent to ``controlled_gate(Y())`` and locally equivalent to
    ``CANONICAL(1/2,0,0)``

    .. math::
        \text{CY}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & i & 0
            \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, -1j],
                              [0, 0, 1j, 0]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> Gate:
        return self  # Hermitian


class CH(Gate):
    r"""A controlled-Hadamard gate

    Equivalent to ``controlled_gate(H())`` and locally equivalent to
    ``CANONICAL(1/2, 0, 0)``

    .. math::
        \text{CH}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \tfrac{1}{\sqrt{2}} &  \tfrac{1}{\sqrt{2}} \\
                0 & 0 & \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}
            \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                              [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> Gate:
        return self  # Hermitian


class U3(Gate):
    r"""The U3 single qubit gate from QASM.
    The U2 gaet is the U3 gate with theta=pi/2. The U1 gate has theta=phi=0,
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


class U2(U3):
    """A 'single pulse' 1-qubit gate defined in QASM"""
    def __init__(self, phi: float, lam: float, q0: Qubit = 0) -> None:
        super().__init__(pi/2, phi, lam, q0)


class U1(RZ):
    """A diagonal 1-qubit gate defined in QASM. Equivalent to RZ"""
    def __init__(self, lam: float, q0: Qubit = 0) -> None:
        super().__init__(lam, q0)


class U0(I):
    """The U0 gate from QASM. Wait a given length of time.
    """
    def __init__(self, m: float, q0: Qubit = 0):
        super().__init__(q0)
        self._params = dict(m=m)


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
        gate = U3(theta, phi, lam, q1)
        return control_gate(q0, gate).tensor

    @property
    def H(self) -> Gate:
        theta, phi, lam = self.params.values()
        return CU3(-theta, -lam, -phi, *self.qubits)


class CRZ(Gate):
    r"""A controlled RZ gate.

    QASM calls this a CU1 gate (controlled-U1).
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
