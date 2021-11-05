# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates peculiar to Cirq
"""

from typing import List

# import numpy as np
import sympy as sym
from sympy.abc import theta as sym_theta
from sympy.abc import phi as sym_phi
from sympy.abc import t as sym_t
from sympy.abc import p as sym_p

# from ..config import SWAP_TARGET
from ..states import Qubit, Variable
from ..paulialgebra import Pauli
from ..operations import StdGate, OperatorStructure
from .stdgates_1q import I, X, XPow, ZPow
from .stdgates_2q import CZ, Swap

__all__ = ("PhasedX", "PhasedXPow", "FSim", "FSwap", "FSwapPow", "Syc")


# Kudos: Phased X gate and powers of phased X gates adapted from Cirq


class PhasedX(StdGate):
    r"""A phased X gate, equivalent to the circuit
    ───Z^-p───X───Z^p───
    """
    # Kudos: Adapted from Cirq

    cv_operator_structure = OperatorStructure.monomial

    cv_sym_operator = ZPow(sym_p, 0).sym_operator @ X(0).sym_operator @ ZPow(-sym_p, 0).sym_operator

    def __init__(self, p: Variable, q0: Qubit) -> None:
        super().__init__(p, q0)

    @property
    def H(self) -> "PhasedX":
        return self

    def __pow__(self, t: Variable) -> "PhasedXPow":
        return PhasedXPow(self.p, t, *self.qubits)

# end class PhasedX


class PhasedXPow(StdGate):
    """A phased X gate raised to a power.

    Equivalent to the circuit ───Z^-p───X^t───Z^p───

    """
    # Kudos: Adapted from Cirq

    cv_sym_operator = ZPow(sym_p, 0).sym_operator @ XPow(sym_t, 0).sym_operator @ ZPow(-sym_p, 0).sym_operator

    def __init__(self, p: Variable, t: Variable, q0: Qubit) -> None:
        super().__init__(p, t, q0)

    @property
    def H(self) -> "PhasedXPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "PhasedXPow":
        return PhasedXPow(self.p, self.t * t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["PhX({p})^{t}"]


# end class PhasedXPow


class FSim(StdGate):
    r"""Fermionic simulation gate family.

    Contains all two qubit interactions that preserve excitations, up to
    single-qubit rotations and global phase.

    Locally equivalent to ``Can(theta/pi, theta/pi, phi/(2*pi))``

    .. math::
        \text{FSim}(\theta, \phi) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\theta) & -i \sin(\theta) & 0 \\
                0 & -i \sin(\theta)  & \cos(\theta) & 0 \\
                0 & 0 & 0 & e^{-i\phi)}
            \end{pmatrix}
    """
    # Kudos: Adapted from Cirq

    cv_interchangeable = True
    cv_sym_operator = sym.Matrix([
        [1, 0, 0, 0],
        [0, sym.cos(sym_theta), -sym.I * sym.sin(sym_theta), 0],
        [0, -sym.I * sym.sin(sym_theta), sym.cos(sym_theta), 0],
        [0, 0, 0, sym.exp(-sym.I * sym_phi)],
    ])


    def __init__(self, theta: Variable, phi: Variable, q0: Qubit, q1: Qubit):
        """
        Args:
            theta: Swap angle on the span(|01⟩, |10⟩) subspace, in radians.
            phi: Phase angle, in radians, applied to |11⟩ state.
        """
        super().__init__(theta, phi, q0, q1)

    @property
    def H(self) -> "FSim":
        return self ** -1

    def __pow__(self, t: Variable) -> "FSim":
        return FSim(self.theta * t, self.phi * t, *self.qubits)


# end class FSim


# Kudos: Adapted from OpenFermion-Cirq
# https://github.com/quantumlib/OpenFermion-Cirq/blob/master/openfermioncirq/gates/common_gates.py
class FSwap(StdGate):
    r"""Fermionic swap gate. It swaps adjacent fermionic modes in
    the Jordan-Wigner representation.
    Locally equivalent to iSwap and ``Can(1/2, 1/2, 0)``

    .. math::
        \text{FSwap} = \begin{pmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0 \\
                        0 & 1 & 0 & 0 \\
                        0 & 0 & 0 & -1
                       \end{pmatrix}

    """
    cv_sym_operator = sym.Matrix([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1],
            ])


    cv_interchangable = True
    cv_operator_structure = OperatorStructure.monomial

    def __init__(self, q0: Qubit, q1: Qubit):
        super().__init__(q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return Swap(q0, q1).hamiltonian + CZ(q0, q1).hamiltonian

    @property
    def H(self) -> "FSwapPow":
        return FSwapPow(-1, *self.qubits)

    def __pow__(self, t: Variable) -> "FSwapPow":
        return FSwapPow(t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [SWAP_TARGET + "ᶠ"] * 2


# End class FSwap


# Kudos: Adapted from OpenFermion-Cirq
# https://github.com/quantumlib/OpenFermion-Cirq/blob/master/openfermioncirq/gates/common_gates.py
class FSwapPow(StdGate):
    """Powers of the fermionic swap (FSwap) gate.
    Locally equivalent to ``Can(t/2, t/2, 0)``
    """
    cv_sym_operator = sym.Matrix([
            [1, 0, 0, 0],
            [0, sym.exp(sym.I * sym.pi * sym_t/2) * sym.cos(sym.pi * sym_t / 2), -sym.I * sym.exp(sym.I * sym.pi * sym_t/2) * sym.sin(sym.pi * sym_t / 2), 0],
            [0, -sym.I * sym.exp(sym.I * sym.pi * sym_t/2) * sym.sin(sym.pi * sym_t / 2), sym.exp(sym.I * sym.pi * sym_t/2) * sym.cos(sym.pi * sym_t / 2), 0],
            [0, 0, 0, sym.exp(sym.I * sym.pi * sym_t)],
        ])


    cv_interchangable = True

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit):
        super().__init__(t, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        return self.t * FSwap(*self.qubits).hamiltonian

    @property
    def H(self) -> "FSwapPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "FSwapPow":
        return FSwapPow(self.t * t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [SWAP_TARGET + "ᶠ^{t}"] * 2


# End class FSwapPow

class Syc(StdGate):
    r"""The Sycamore gate is a two-qubit gate equivalent to
    ``FSim(π/2, π/6)``, and locally equivalent to ``Can(1/2, 1/2, 1/12)``.

    This gate was used to demonstrate quantum on Google's Sycamore chip.

    .. math::
        \operatorname{Sycamore}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0& -i  & 0 \\
                0 & -i  & 0& 0 \\
                0 & 0 & 0 & \exp(- i \pi/6)
            \end{pmatrix}

    Ref:
         https://www.nature.com/articles/s41586-019-1666-5
    """
    # Kudos: Adapted from Cirq

    cv_interchangeable = True
    cv_operator_structure = OperatorStructure.monomial

    cv_sym_operator = sym.Matrix([
            [1, 0, 0, 0],
            [0, 0, -sym.I, 0],
            [0, -sym.I, 0, 0],
            [0, 0, 0, sym.exp(- sym.I * sym.pi /6)],
            ])

    def __init__(self, q0: Qubit, q1: Qubit):
        super().__init__(q0, q1)

    @property
    def H(self) -> "FSim":
        return self ** -1

    def __pow__(self, t: Variable) -> "FSim":
        return FSim(t * sym.pi / 2, t * sym.pi / 6, *self.qubits)


# end class Syc


# fin
