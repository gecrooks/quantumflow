# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Standard three qubit gates
"""
from typing import List

import sympy as sym
from sympy.abc import theta as sym_theta

from .. import utils
from ..gates import Unitary
from ..operations import StdCtrlGate, StdGate

# from ..config import CTRL
from ..paulialgebra import Pauli
from ..states import Qubit, State, Variable
from .stdgates_1q import X, XPow, Z, ZPow
from .stdgates_2q import ISwap, Swap, SwapPow

# 3-qubit gates, in alphabetic order

__all__ = (
    "CCiX",
    "CCX",
    "CCXPow",
    "CCZ",
    "CCZPow",
    "CISwap",
    "CSwap",
    "CSwapPow",
    "Deutsch",
    "Margolus",
    "CCNot",
    "Tofolli",
    "Fredkin",
)

# NB: Not all these control gates are StdCtrlGate subclasses since the target gate
# may not exist in QF.


class CCiX(StdGate):
    r"""
    A doubly controlled iX gate.

    .. math::
        \text{CCiX}() \equiv \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & i \\
                0 & 0 & 0 & 0 & 0 & 0 & i & 0
            \end{pmatrix}

    Refs:
          http://arxiv.org/abs/1210974
    """
    # Kudos: Adapted from 1210974, via Quipper
    # See also https://arxiv.org/pdf/1508.03273.pdf, fig 3

    cv_tensor_structure = "monomial"
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, sym.I],
            [0, 0, 0, 0, 0, 0, sym.I, 0],
        ]
    )

    def __init__(self, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(q0, q1, q2)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return -X(q2) * (1 - Z(q1)) * (1 - Z(q0)) * sym.pi / 8

    def _diagram_labels_(self) -> List[str]:
        return ["─" + CTRL + "─", "─" + CTRL + "─", "iX─"]


# end class CCiX


class CCX(StdCtrlGate):
    r"""
    A 3-qubit Toffoli gate. A controlled, controlled-not.

    Equivalent to ``controlled_gate(cnot())``

    .. math::
        \text{CCNot}() \equiv \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
            \end{pmatrix}

    """
    cv_target = X

    def __init__(self, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(q0, q1, q2)

    @property
    def H(self) -> "CCX":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CCXPow":
        return CCXPow(t, *self.qubits)


# end class CCX


class CCXPow(StdCtrlGate):
    r"""
    Powers of the Toffoli gate.

    args:
        t:  turns (powers of the CNOT gate, or controlled-powers of X)
        q0: control qubit
        q1: control qubit
        q2: target qubits
    """
    cv_target = XPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(t, q0, q1, q2)

    @property
    def H(self) -> "CCXPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "CCXPow":
        return CCXPow(t * self.t, *self.qubits)


# end class CCXPow


class CCZ(StdCtrlGate):
    r"""
    A controlled, controlled-Z.

    Equivalent to ``controlled_gate(CZ())``

    .. math::
        \text{CSWAP}() \equiv \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
            \end{pmatrix}
    """
    cv_target = Z
    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(q0, q1, q2)

    @property
    def H(self) -> "CCZ":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CCZPow":
        return CCZPow(t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [CTRL, CTRL, CTRL]


# end class CCZ


class CCZPow(StdCtrlGate):
    r"""
    Powers of the Toffoli gate.

    Args:
        t:  Powers of the Z gate
        q0: control qubit
        q1: control qubit
        q2: target qubits
    """
    cv_target = ZPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(t, q0, q1, q2)

    @property
    def H(self) -> "CCZPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "CCZPow":
        return CCZPow(t * self.t, *self.qubits)


# end class CCZPow


class CISwap(StdCtrlGate):
    r"""
    A controlled iSwap gate.

    .. math::
        \text{CISwap}() \equiv \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & i & 0 \\
                0 & 0 & 0 & 0 & 0 & i & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
            \end{pmatrix}


    Refs:
          https://arxiv.org/abs/2002.11728
    """
    cv_target = ISwap

    def __init__(self, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(q0, q1, q2)


# end class CISwap


class CSwap(StdCtrlGate):
    r"""
    A 3-qubit Fredkin gate. A controlled swap.

    Equivalent to ``controlled_gate(swap())``

    .. math::
        \text{CSwap}() \equiv \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
            \end{pmatrix}
    """
    cv_target = Swap

    def __init__(self, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(q0, q1, q2)

    @property
    def H(self) -> "CSwap":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CSwapPow":
        return CSwapPow(t, *self.qubits)


# end class CSWAP


class CSwapPow(StdCtrlGate):
    r"""
    Powers of the Controlled Swap gate.

    Args:
        t:  Powers of the controlled-swap gate
        q0: control qubit
        q1: control qubit
        q2: target qubits
    """
    cv_target = SwapPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(t, q0, q1, q2)

    @property
    def H(self) -> "CSwapPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "CSwapPow":
        return CSwapPow(t * self.t, *self.qubits)


# end class CSwapPow


# TODO: hamiltonian
class Deutsch(StdGate):
    r"""
    The Deutsch gate, a 3-qubit universal quantum gate.

    A controlled-controlled-i*R_x(2*theta) gate.
    Note that Deutsch(pi/2) is the CCNot gate.

    .. math::
        \text{Deutsch}(\theta) \equiv \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & i \cos(\theta) & \sin(\theta) \\
                0 & 0 & 0 & 0 & 0 & 0 & \sin(\theta)& i \cos(\theta)
            \end{pmatrix}

    Ref:
        D. Deutsch, Quantum Computational Networks,
            Proc. R. Soc. Lond. A 425, 73 (1989).
    """
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, sym.I * sym.cos(sym_theta), sym.sin(sym_theta)],
            [0, 0, 0, 0, 0, 0, sym.sin(sym_theta), sym.I * sym.cos(sym_theta)],
        ]
    )

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(theta, q0, q1, q2)

    # TODO: Specializes to Toffoli

    def _diagram_labels_(self) -> List[str]:
        return [CTRL, CTRL, "iRx({theta})^2"]


# end class Deutsch


class Margolus(StdGate):
    r"""
    A "simplified" Toffoli gate.

    Differs from the Toffoli only in that the |101> state picks up a -1 phase.
    Can be implemented with 3 CNot gates.

    .. math::
        \text{CCiX}() \equiv \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & -1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
            \end{pmatrix}

    Refs:
        https://arxiv.org/pdf/quant-ph/0312225.pdf
    """

    cv_hermitian = True
    cv_tensor_structure = "monomial"

    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )

    def __init__(self, q0: Qubit, q1: Qubit, q2: Qubit) -> None:
        super().__init__(q0, q1, q2)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return (
            (1 - Z(q0))
            * (-2 - Z(q1) * X(q2) + Z(q1) * Z(q2) + X(q2) + Z(q2))
            * sym.pi
            / 8
        )

    @property
    def H(self) -> "Margolus":
        return self  # Hermitian


# end class Margolus


CCNot = CCX
Tofolli = CCX
Fredkin = CSwap

# fin
