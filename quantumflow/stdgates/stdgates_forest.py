# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates peculiar to Rigetti's Forest
"""

import numpy as np

from .. import tensors, var
from ..config import CTRL, NCTRL
from ..ops import StdGate
from ..paulialgebra import Pauli, sZ
from ..qubits import Qubit
from ..tensors import QubitTensor
from ..utils import cached_property
from ..var import Variable

__all__ = ("CPhase", "CPhase00", "CPhase01", "CPhase10", "PSwap")


class CPhase(StdGate):
    r"""A 2-qubit 11 phase-shift gate

    .. math::
        \text{CPhase}(\theta) \equiv \text{diag}(1, 1, 1, e^{i \theta})
    """
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"
    _diagram_labels = [CTRL + "({theta})", CTRL + "({theta})"]

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        theta = self.param("theta")
        return -theta * (1 + sZ(q0) * sZ(q1) - sZ(q0) - sZ(q1)) / 4

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, np.exp(1j * theta)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CPhase":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase":
        theta = self.param("theta") * t
        return CPhase(theta, *self.qubits)


# end class CPhase


class CPhase00(StdGate):
    r"""A 2-qubit 00 phase-shift gate

    .. math::
        \text{CPhase00}(\theta) \equiv \text{diag}(e^{i \theta}, 1, 1, 1)
    """
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"
    _diagram_labels = [NCTRL + "({theta})", NCTRL + "({theta})"]

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        theta = self.param("theta")
        return -theta * (1 + sZ(q0) * sZ(q1) + sZ(q0) + sZ(q1)) / (4)

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [
            [np.exp(1j * theta), 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CPhase00":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase00":
        theta = self.param("theta")
        return CPhase00(theta * t, *self.qubits)


# end class CPhase00


class CPhase01(StdGate):
    r"""A 2-qubit 01 phase-shift gate

    .. math::
        \text{CPhase01}(\theta) \equiv \text{diag}(1, e^{i \theta}, 1, 1)
    """
    cv_tensor_structure = "diagonal"
    _diagram_labels = [NCTRL + "({theta})", CTRL + "({theta})"]

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return -self.param("theta") * (1 - sZ(q0) * sZ(q1) + sZ(q0) - sZ(q1)) / (4)

    @cached_property
    def tensor(self) -> QubitTensor:
        unitary = [
            [1.0, 0, 0, 0],
            [0, np.exp(1j * var.asfloat(self.param("theta"))), 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CPhase01":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase01":
        theta = self.param("theta")
        return CPhase01(theta * t, *self.qubits)


# end class CPhase01


class CPhase10(StdGate):
    r"""A 2-qubit 10 phase-shift gate

    .. math::
        \text{CPhase10}(\theta) \equiv \text{diag}(1, 1, e^{i \theta}, 1)
    """
    cv_tensor_structure = "diagonal"
    _diagram_labels = [CTRL + "({theta})", NCTRL + "({theta})"]

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        theta = self.param("theta")
        return -theta * (1 - sZ(q0) * sZ(q1) - sZ(q0) + sZ(q1)) / (4)

    @cached_property
    def tensor(self) -> QubitTensor:
        unitary = [
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, np.exp(1j * var.asfloat(self.param("theta"))), 0],
            [0, 0, 0, 1.0],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CPhase10":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase10":
        theta = self.param("theta")
        return CPhase10(theta * t, *self.qubits)


# end class CPhase10


class PSwap(StdGate):
    r"""A 2-qubit parametric-swap gate, as defined by Quil.
    Interpolates between SWAP (theta=0) and iSWAP (theta=pi/2).

    Locally equivalent to ``CAN(1/2, 1/2, 1/2 - theta/pi)``

    .. math::
        \text{PSwap}(\theta) \equiv \begin{pmatrix} 1&0&0&0 \\
        0&0&e^{i\theta}&0 \\ 0&e^{i\theta}&0&0 \\ 0&0&0&1 \end{pmatrix}
    """
    cv_interchangeable = True
    cv_tensor_structure = "monomial"

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [
            [[[1, 0], [0, 0]], [[0, 0], [np.exp(theta * 1.0j), 0]]],
            [[[0, np.exp(theta * 1.0j)], [0, 0]], [[0, 0], [0, 1]]],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "PSwap":
        theta = self.param("theta")
        theta = 2.0 * np.pi - theta % (2.0 * np.pi)
        return PSwap(theta, *self.qubits)


# end class PSwap


# fin
