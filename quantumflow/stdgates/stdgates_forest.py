# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates peculiar to Rigetti's Forest
"""

from typing import List

import sympy as sym
from sympy.abc import theta as sym_theta

from ..operations import OperatorStructure, StdGate

# from ..config import CTRL, NCTRL
from ..paulialgebra import Pauli
from ..states import Qubit, Variable
from .stdgates_1q import Z

__all__ = ("CPhase", "CPhase00", "CPhase01", "CPhase10", "PSwap")


class CPhase(StdGate):
    r"""A 2-qubit 11 phase-shift gate

    .. math::
        \text{CPhase}(\theta) \equiv \text{diag}(1, 1, 1, e^{i \theta})
    """
    cv_interchangeable = True
    cv_operator_structure = OperatorStructure.diagonal
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, sym.exp(sym.I * sym_theta)],
        ]
    )

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(theta, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return -self.theta * (1 + Z(q0) * Z(q1) - Z(q0) - Z(q1)) / 4

    @property
    def H(self) -> "CPhase":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase":
        return CPhase(t * self.theta, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [CTRL, "P({theta})"]


# end class CPhase


class CPhase00(StdGate):
    r"""A 2-qubit 00 phase-shift gate

    .. math::
        \text{CPhase00}(\theta) \equiv \text{diag}(e^{i \theta}, 1, 1, 1)
    """
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"
    cv_operator_structure = OperatorStructure.diagonal
    cv_sym_operator = sym.Matrix(
        [
            [sym.exp(sym.I * sym_theta), 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(theta, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return -self.theta * (1 + Z(q0) * Z(q1) + Z(q0) + Z(q1)) / 4

    @property
    def H(self) -> "CPhase00":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase00":
        return CPhase00(self.theta * t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [NCTRL + "({theta})", NCTRL + "({theta})"]


# end class CPhase00


class CPhase01(StdGate):
    r"""A 2-qubit 01 phase-shift gate

    .. math::
        \text{CPhase01}(\theta) \equiv \text{diag}(1, e^{i \theta}, 1, 1)
    """
    cv_tensor_structure = "diagonal"
    cv_operator_structure = OperatorStructure.diagonal
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, sym.exp(sym.I * sym_theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(theta, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return -self.theta * (1 - Z(q0) * Z(q1) + Z(q0) - Z(q1)) / (4)

    @property
    def H(self) -> "CPhase01":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase01":
        return CPhase01(self.theta * t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [NCTRL + "({theta})", CTRL + "({theta})"]


# end class CPhase01


class CPhase10(StdGate):
    r"""A 2-qubit 10 phase-shift gate

    .. math::
        \text{CPhase10}(\theta) \equiv \text{diag}(1, 1, e^{i \theta}, 1)
    """
    cv_operator_structure = OperatorStructure.diagonal
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, sym.exp(sym.I * sym_theta), 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(theta, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return -self.theta * (1 - Z(q0) * Z(q1) - Z(q0) + Z(q1)) / (4)

    @property
    def H(self) -> "CPhase10":
        return self ** -1

    def __pow__(self, t: Variable) -> "CPhase10":
        return CPhase10(self.theta * t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [CTRL + "({theta})", NCTRL + "({theta})"]


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
    cv_operator_structure = OperatorStructure.monomial
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 0, sym.exp(sym_theta * sym.I), 0],
            [0, sym.exp(sym_theta * sym.I), 0, 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(theta, q0, q1)

    @property
    def H(self) -> "PSwap":
        theta = 2.0 * sym.pi - self.theta % (2 * sym.pi)
        return PSwap(theta, *self.qubits)


# end class PSwap


# fin
