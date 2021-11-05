# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Common two qubit gates
======================

# FIXME
.. autoclass:: CX
.. autoclass:: CXPow
.. autoclass:: CY
.. autoclass:: CYPow
.. autoclass:: CZ
.. autoclass:: CZPow

"""
from typing import List

import numpy as np
import sympy as sym

# symbols
from sympy.abc import alpha as sym_alpha
from sympy.abc import phi as sym_phi
from sympy.abc import t as sym_t
from sympy.abc import theta as sym_theta

from ..gates import Unitary

# from ..config import CTRL, SWAP_TARGET, TARGET
from ..operations import OperatorStructure, StdCtrlGate, StdGate, Variable
from ..paulialgebra import Pauli
from ..states import Qubit
from ..utils.future import TypeAlias
from .stdgates_1q import V_H, H, HPow, I, S, T, V, X, XPow, Y, YPow, Z, ZPow

__all__ = (
    "XX",
    "YY",
    "ZZ",
    "Can",
    "A",
    "B",
    "Barenco",
    "Can",
    "CH",
    "CHPow",
    "CS",
    "CT",
    "CV",
    "CV_H",
    "CX",
    "CXPow",
    "CY",
    "CYPow",
    "CZ",
    "CZPow",
    "ECP",
    "Exch",
    "Givens",
    "ISwap",
    "SqrtISwap",
    "SqrtISwap_H",
    "SqrtSwap",
    "SqrtSwap_H",
    "Swap",
    "SwapPow",
    "W",
    "XY",
    # Aliases
    "CNot",  # Alias for CX  # DOCME
    "CNotPow",  # Alias for CXPow
    # "CrossResonance",
)


sym_tx = sym.Symbol("tx")
sym_ty = sym.Symbol("ty")
sym_tz = sym.Symbol("tz")


class XX(StdGate):
    r"""A parametric 2-qubit gate generated from an XX interaction,

    Equivalent to ``Can(t, 0, 0)``.

    XX(1/2) is the Mølmer-Sørensen gate.

    Ref: Sørensen, A. & Mølmer, K. Quantum computation with ions in thermal
    motion. Phys. Rev. Lett. 82, 1971–1974 (1999)

    Args:
        t:
    """
    # TODO: Is XX(1/2) MS gate, or is it XX(-1/2)???
    cv_interchangeable = True
    cv_sym_operator = sym.Matrix(
        [
            [sym.cos(sym.pi * sym_t / 2), 0, 0, -sym.I * sym.sin(sym.pi * sym_t / 2)],
            [0, sym.cos(sym.pi * sym_t / 2), -sym.I * sym.sin(sym.pi * sym_t / 2), 0],
            [0, -sym.I * sym.sin(sym.pi * sym_t / 2), sym.cos(sym.pi * sym_t / 2), 0],
            [-sym.I * sym.sin(sym.pi * sym_t / 2), 0, 0, sym.cos(sym.pi * sym_t / 2)],
        ]
    )

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return self.t * X(q0) * X(q1) * sym.pi / 2

    @property
    def H(self) -> "XX":
        return self ** -1

    def __pow__(self, t: Variable) -> "XX":
        return XX(t * self.t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["XX^{t}"] * 2


# end class XX


class YY(StdGate):
    r"""A parametric 2-qubit gate generated from a YY interaction.

    Equivalent to ``Can(0, t, 0)``, and locally equivalent to
    ``Can(t, 0, 0)``

    Args:
        t:
    """
    cv_interchangeable = True
    cv_sym_operator = sym.Matrix(
        [
            [sym.cos(sym.pi * sym_t / 2), 0, 0, sym.I * sym.sin(sym.pi * sym_t / 2)],
            [0, sym.cos(sym.pi * sym_t / 2), -sym.I * sym.sin(sym.pi * sym_t / 2), 0],
            [0, -sym.I * sym.sin(sym.pi * sym_t / 2), sym.cos(sym.pi * sym_t / 2), 0],
            [sym.I * sym.sin(sym.pi * sym_t / 2), 0, 0, sym.cos(sym.pi * sym_t / 2)],
        ]
    )

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return self.t * Y(q0) * Y(q1) * sym.pi / 2

    @property
    def H(self) -> "YY":
        return self ** -1

    def __pow__(self, t: Variable) -> "YY":
        return YY(t * self.t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["YY^{t}"] * 2


# end class XX


class ZZ(StdGate):
    r"""A parametric 2-qubit gate generated from a ZZ interaction.

    Equivalent to ``Can(0,0,t)``, and locally equivalent to
    ``Can(t,0,0)``

    Args:
        t:
    """
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"
    cv_sym_operator = sym.Matrix(
        [
            [sym.exp(-sym.I * sym.pi * sym_t / 2), 0, 0, 0],
            [0, sym.exp(sym.I * sym.pi * sym_t / 2), 0, 0],
            [0, 0, sym.exp(sym.I * sym.pi * sym_t / 2), 0],
            [0, 0, 0, sym.exp(-sym.I * sym.pi * sym_t / 2)],
        ]
    )

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return self.t * Z(q0) * Z(q1) * sym.pi / 2

    @property
    def H(self) -> "ZZ":
        return self ** -1

    def __pow__(self, t: Variable) -> "ZZ":
        return ZZ(t * self.t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["ZZ^{t}"] * 2


# end class ZZ


# TODO: Add references and explanation
# DOCME: Comment on sign conventions.
# NB: Must be defined after XX, YY, ZZ
class Can(StdGate):
    r"""The canonical 2-qubit gate

    The canonical decomposition of 2-qubits gates removes local 1-qubit
    rotations, and leaves only the non-local interactions.

    .. math::
        \text{Can}(t_x, t_y, t_z) \equiv
            \exp\Big\{-i\frac{\pi}{2}(t_x X\otimes X
            + t_y Y\otimes Y + t_z Z\otimes Z)\Big\}

    """
    cv_interchangeable = True
    cv_sym_operator = (
        XX.cv_sym_operator.subs(sym_t, sym_tx)
        @ YY.cv_sym_operator.subs(sym_t, sym_ty)
        @ ZZ.cv_sym_operator.subs(sym_t, sym_tz)
    )

    def __init__(
        self, tx: Variable, ty: Variable, tz: Variable, q0: Qubit, q1: Qubit
    ) -> None:
        super().__init__(tx, ty, tz, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        tx, ty, tz = self.args
        q0, q1 = self.qubits
        return (
            (tx * X(q0) * X(q1) + ty * Y(q0) * Y(q1) + tz * Z(q0) * Z(q1)) * sym.pi / 2
        )

    @property
    def H(self) -> "Can":
        return self ** -1

    def __pow__(self, t: Variable) -> "Can":
        tx, ty, tz = self.args
        return Can(tx * t, ty * t, tz * t, *self.qubits)


# end class Can


class A(StdGate):
    r"""The A gate. A 2-qubit, 2-parameter gate that is locally
    equivalent to Can(1/2, t, t)

    .. math::
        {autogenerated_latex}

    Args:
        theta (Variable):
        q0 (Qubit):
        q1 (Qubit):

    Refs:
        :cite:`Barkoutsos2018a`
        :cite:`Gard2020a`

    [1]: https://threeplusone.com/gates#A
    """
    cv_hermitian = True
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, sym.cos(sym_theta), sym.exp(sym.I * sym_phi) * sym.sin(sym_theta), 0],
            [0, sym.exp(-sym.I * sym_phi) * sym.sin(sym_theta), -sym.cos(sym_theta), 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, theta: Variable, phi: Variable, q0: Qubit, q1: Qubit):
        super().__init__(theta, phi, q0, q1)

    @property
    def H(self) -> "A":
        return self


# end class A


class B(StdGate):
    """The B (Berkeley) gate. Equivalent to Can(-1/2, -1/4, 0)

    .. math::
        {autogenerated_latex}

    [1]: https://threeplusone.com/gates#B
    """

    cv_interchangeable = True
    cv_sym_operator = (
        sym.Matrix(
            [
                [1 + sym.sqrt(2), 0, 0, sym.I],
                [0, 1, sym.I * (1 + sym.sqrt(2)), 0],
                [0, sym.I * (1 + sym.sqrt(2)), 1, 0],
                [sym.I, 0, 0, 1 + sym.sqrt(2)],
            ]
        )
        * sym.sqrt(2 - sym.sqrt(2))
        / 2
    )

    def __init__(self, q0: Qubit, q1: Qubit):
        super().__init__(q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        return Can(-1 / 2, -1 / 4, 0, *self.qubits).hamiltonian

    @property
    def H(self) -> "Can":
        return self ** -1

    def __pow__(self, t: Variable) -> "Can":
        return Can(-t / 2, -t / 4, 0, *self.qubits)


# end class B


class Barenco(StdGate):
    """A universal two-qubit gate:

    Ref:
        A Universal Two–Bit Gate for Quantum Computation, A. Barenco (1996)
        https://arxiv.org/pdf/quant-ph/9505016.pdf
    """

    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [
                0,
                0,
                sym.exp(sym.I * sym_alpha) * sym.cos(sym_theta),
                -sym.I * sym.exp(sym.I * (sym_alpha - sym_phi)) * sym.sin(sym_theta),
            ],
            [
                0,
                0,
                -sym.I * sym.exp(sym.I * (sym_alpha + sym_phi)) * sym.sin(sym_theta),
                sym.exp(sym.I * sym_alpha) * sym.cos(sym_theta),
            ],
        ]
    )

    # Note: parameter order as defined by original paper
    def __init__(
        self,
        phi: Variable,
        alpha: Variable,
        theta: Variable,
        q0: Qubit,
        q1: Qubit,
    ) -> None:
        super().__init__(phi, alpha, theta, q0, q1)

    @property
    def H(self) -> "Barenco":
        return Barenco(self.phi + sym.pi, -self.alpha, self.theta, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["───" + CTRL + "───", "Barenco({phi}, {alpha}, {theta})"]


class CH(StdCtrlGate):
    r"""A controlled-Hadamard gate

    Locally equivalent to ``Can(1/2, 0, 0)``.

    .. math::
           {autogenerated_latex}

    Args:
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CX
    """
    cv_target = H

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "CH":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CHPow":
        return CHPow(t, *self.qubits)

    # TODO HPow, CHPow


# end class CH


class CHPow(StdCtrlGate):
    r"""Powers of the controlled-Hadamard gate

    Locally equivalent to ``Can(1/2, 0, 0)``.

    .. math::
           {autogenerated_latex}

    Args:
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CXPow
    """
    cv_target = HPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def H(self) -> "CHPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "CHPow":
        return CHPow(t * self.t, *self.qubits)


# end class CHPow


class CrossResonance(StdGate):
    # DOCME
    # TESTME

    def __init__(self, s: Variable, b: Variable, c: Variable, q0: Qubit, q1: Qubit):
        super().__init__(s, b, c, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return self.s * (X(q0) - self.b * Z(q0) * X(q1) + self.c * X(q1)) * sym.pi / 2

    @property
    def operator(self) -> np.ndarray:
        if self._operator is None:
            U = Unitary.from_hamiltonian(self.hamiltonian, self.qubits)
            self._operator = U.operator
        return self._operator

    @property
    def sym_operator(self) -> sym.Matrix:
        if self._sym_operator is None:
            self._sym_operator = sym.Matrix(self.operator)
        return self._sym_operator

    @property
    def H(self) -> "CrossResonance":
        return self ** -1

    def __pow__(self, t: Variable) -> "CrossResonance":
        return CrossResonance(t * self.s, self.b, self.c, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["CR({s}, {b}, {c})_0", "CR({s}, {b}, {c})_1"]


# end class CrossResonance


class CS(StdCtrlGate):
    r"""A controlled-S gate

    .. math::
           {autogenerated_latex}
    """
    cv_target = S
    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    def __pow__(self, t: Variable) -> "CZPow":
        return CZPow(t / 2, *self.qubits)


# End class CS


class CT(StdCtrlGate):
    r"""A controlled-T gate

    .. math::
           {autogenerated_latex}
    """
    cv_target = T
    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    def __pow__(self, t: Variable) -> "CZPow":
        return CZPow(t / 4, *self.qubits)


# End class CT


class CV(StdCtrlGate):
    r"""A controlled V (sqrt of CNOT) gate."""

    cv_target = V

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "CV_H":
        return CV_H(*self.qubits)

    def __pow__(self, t: Variable) -> "CXPow":
        return CXPow(t / 2, *self.qubits)


# end class CV


class CV_H(StdCtrlGate):
    r"""A controlled V (sqrt of CNOT) gate."""

    cv_target = V_H

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "CV":
        return CV(*self.qubits)

    def __pow__(self, t: Variable) -> "CXPow":
        return CXPow(-t / 2, *self.qubits)


# end class CV_H


class CX(StdCtrlGate):
    r"""A controlled-X gate, also called controlled-not (cnot).

    Locally equivalent to ``Can(1/2, 0, 0)``.

    .. math::
           {autogenerated_latex}

    Args:
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CX
    """
    cv_target = X

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "CX":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CXPow":
        return CXPow(t, *self.qubits)


# end class CX


class CXPow(StdCtrlGate):
    r"""Powers of the controlled-X (CX, CNOT) gate.

    .. math::
        {autogenerated_latex}

    Args:
        t (Variable): Power to which the base gate is raised.
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CXPow
    """
    cv_target = XPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def H(self) -> "CXPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "CXPow":
        return CXPow(t * self.t, *self.qubits)


# end class CXPow


class CY(StdCtrlGate):
    r"""A controlled-Y gate.

    Locally equivalent to ``Can(1/2, 0, 0)``.

    .. math::
           {autogenerated_latex}

    Args:
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CY
    """
    cv_target = Y

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "CY":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CYPow":
        return CYPow(t, *self.qubits)


# end class CY


class CYPow(StdCtrlGate):
    r"""Powers of the controlled-Y (CY) gate.

    .. math::
        {autogenerated_latex}

    Attributes:
        t: Power to which the CY gate is raised.
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CYPow
    """
    cv_target = YPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def H(self) -> "CYPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "CYPow":
        return CYPow(t * self.t, *self.qubits)


# end class CYPow


class CZ(StdCtrlGate):
    r"""A controlled-Z gate.

    Locally equivalent to ``Can(1/2, 0, 0)``.

    .. math::
           {autogenerated_latex}

    Args:
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CZ
    """
    cv_target = Z

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "CZ":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CZPow":
        return CZPow(t, *self.qubits)


# end class CZ


class CZPow(StdCtrlGate):
    r"""Powers of the controlled-Z (CZ) gate.

    .. math::
        {autogenerated_latex}

    Args:
        t (Variable): Power to which the CZ gate is raised.
        q0 (Qubit): Control qubit
        q1 (Qubit): Target qubit

    [1]: https://threeplusone.com/gates#CZPow
    """
    cv_target = ZPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def H(self) -> "CZPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "CZPow":
        return CZPow(t * self.t, *self.qubits)


# end class CZPow


# DOCME
# TODO: Add citations x2
class ECP(StdGate):
    r"""The ECP gate. The peak of the pyramid of gates in the Weyl chamber
    that can be created with a square-root of iSWAP sandwich.

    Equivalent to ``Can(1/2, 1/4, 1/4)``.
    """
    cv_interchangeable = True
    cv_sym_operator = Can.cv_sym_operator.subs(
        [(sym_tx, 1 / 2), (sym_ty, 1 / 4), (sym_tz, 1 / 4)]
    )

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        return Can(1 / 2, 1 / 4, 1 / 4, *self.qubits).hamiltonian

    @property
    def H(self) -> "Can":
        return self ** -1

    def __pow__(self, t: Variable) -> "Can":
        return Can(t / 2, t / 4, t / 4, *self.qubits)


# end class ECP


# Deprecated in favor of SwapPow ???


class Exch(StdGate):
    r"""A 2-qubit parametric gate generated from an isotropic exchange interaction.

    Equivalent to Can(t,t,t), and to SwapPow(2t) up to a phase.

    """
    cv_interchangeable = True
    cv_sym_operator = Can.cv_sym_operator.subs(
        [(sym_tx, sym_t), (sym_ty, sym_t), (sym_tz, sym_t)]
    )

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        return SwapPow(self.t * 2, *self.qubits).hamiltonian

    @property
    def H(self) -> "Exch":
        return self ** -1

    def __pow__(self, t: Variable) -> "Exch":
        return Exch(t * self.t, *self.qubits)


# end class Exch


class Givens(StdGate):
    r"""
    In quantum computational chemistry refers to a 2-qubit gate that defined as

        Givens(θ) ≡ exp(-i θ (Y⊗X - X⊗Y) / 2)

    Locally equivalent to the XY gate.

    ::
        ───Givens(θ)_0───     ───T⁺───XY^θ/π───T────
              │                        │
        ───Givens(θ)_1───     ───T────XY^θ/π───T⁺───

    .. math::
        {autogenerated_latex}

    """
    # Kudos: Adapted from cirq

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(theta, q0, q1)

    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, sym.cos(sym_theta), -sym.sin(sym_theta), 0],
            [0, sym.sin(sym_theta), sym.cos(sym_theta), 0],
            [0, 0, 0, 1],
        ]
    )

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return self.theta * (Y(q0) * X(q1) - X(q0) * Y(q1)) / 2

    @property
    def H(self) -> "Givens":
        return self ** -1

    def __pow__(self, t: Variable) -> "Givens":
        return Givens(t * self.theta, *self.qubits)


# end class Givens


class ISwap(StdGate):
    r"""A 2-qubit iSwap gate

    Equivalent to ``Can(-1/2,-1/2,0)``.

    .. math::
        {autogenerated_latex}
    """
    cv_interchangeable = True
    cv_tensor_structure = "monomial"
    cv_sym_operator = sym.Matrix(
        [[1, 0, 0, 0], [0, 0, sym.I, 0], [0, sym.I, 0, 0], [0, 0, 0, 1]]
    )

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        return Can(-sym.S.Half, -sym.S.Half, 0, *self.qubits).hamiltonian

    @property
    def H(self) -> "XY":
        return self ** -1

    def __pow__(self, t: Variable) -> "XY":
        return XY(-t / 2, *self.qubits)


# end class ISWAP


class Swap(StdGate):
    r"""A 2-qubit swap gate

    Equivalent to ``Can(1/2, 1/2, 1/2)``.

    .. math::
        {autogenerated_latex}

    Args:
        q0 (Qubit): First qubit
        q1 (Qubit): Second qubit

    [1]: https://threeplusone.com/gates#Swap
    """
    cv_interchangeable = True
    cv_hermitian = True
    cv_operator_structure = OperatorStructure.swap
    cv_sym_operator = sym.Matrix(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "Swap":
        return self  # Hermitian

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return (X(q0) * X(q1) + Y(q0) * Y(q1) + Z(q0) * Z(q1) - I(q0)) * sym.pi / 4

    def __pow__(self, t: Variable) -> "SwapPow":
        return SwapPow(t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return [SWAP_TARGET] * 2


# end class Swap


class SwapPow(StdGate):
    r"""Powers of the 2-qubit swap gate

    Equivalent to ``Can(t/2, t/2, t/2)``.

    .. math::
        {autogenerated_latex}

    Args:
        t(Variable):
        q0 (Qubit): First qubit
        q1 (Qubit): Second qubit

    [1]: https://threeplusone.com/gates#SwapPow
    """
    cv_interchangeable = True

    cv_sym_operator = sym.exp(sym.I * sym.pi * sym_t / 2) * sym.Matrix(
        [
            [sym.exp(-sym.I * sym.pi * sym_t / 2), 0, 0, 0],
            [0, sym.cos(sym.pi * sym_t / 2), -sym.I * sym.sin(sym.pi * sym_t / 2), 0],
            [0, -sym.I * sym.sin(sym.pi * sym_t / 2), sym.cos(sym.pi * sym_t / 2), 0],
            [0, 0, 0, sym.exp(-sym.I * sym.pi * sym_t / 2)],
        ]
    )

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def H(self) -> "SwapPow":
        return self ** -1

    @property
    def hamiltonian(self) -> Pauli:
        return self.t * Swap(*self.qubits).hamiltonian

    def __pow__(self, t: Variable) -> "SwapPow":
        return SwapPow(self.t * t, *self.qubits)

    # TODO: _diagram_labels_


# end class SwapPow


class SqrtISwap(StdGate):
    r"""A square root of the iswap gate

    Equivalent to ``Can(-1/4,-1/4,0)``.
    """
    cv_interchangeable = True
    cv_sym_operator = (XX.cv_sym_operator @ YY.cv_sym_operator).subs(
        sym_t, -sym.S.One / 4
    )

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        return Can(-sym.S.One / 4, -sym.S.One / 4, 0, *self.qubits).hamiltonian

    @property
    def H(self) -> "SqrtISwap_H":
        return SqrtISwap_H(*self.qubits)

    def __pow__(self, t: Variable) -> "XY":
        return XY(-t / 4, *self.qubits)


# end class SqrtISwap


# TESTME
class SqrtISwap_H(StdGate):
    r"""The Hermitian conjugate of the square root iswap gate

    Equivalent to ``Can(1/4, 1/4, 0)``.
    """
    cv_interchangeable = True
    cv_sym_operator = (XX.cv_sym_operator @ YY.cv_sym_operator).subs(
        sym_t, sym.S.One / 4
    )

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        return Can(sym.S.One / 4, sym.S.One / 4, 0, *self.qubits).hamiltonian

    @property
    def H(self) -> "SqrtISwap":
        return SqrtISwap(*self.qubits)

    def __pow__(self, t: Variable) -> "XY":
        return XY(t / 4, *self.qubits)


# end class SqrtISwap_H


class SqrtSwap(StdGate):
    r"""Square root of the 2-qubit swap gate

    Equivalent to ``Can(1/4, 1/4, 1/4)``.
    """
    cv_interchangeable = True
    cv_sym_operator = SwapPow.cv_sym_operator.subs(sym_t, sym.Rational(1, 2))

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "SqrtSwap_H":
        return SqrtSwap_H(*self.qubits)

    def __pow__(self, t: Variable) -> "SwapPow":
        return SwapPow(t / 2, *self.qubits)


# end class SqrtSwap


class SqrtSwap_H(StdGate):
    r"""The conjugate of the Square root swap gate

    Equivalent to ``Can(-1/4, -1/4, -1/4)``, and locally equivalent to
    ``Can(3/4, 1/4, 1/4)``
    """
    cv_interchangeable = True
    cv_sym_operator = SwapPow.cv_sym_operator.subs(sym_t, -sym.Rational(1, 2))

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "SqrtSwap":
        return SqrtSwap(*self.qubits)

    def __pow__(self, t: Variable) -> "SwapPow":
        return SwapPow(-t / 2, *self.qubits)


# end class SqrtSwap_H


class W(StdGate):
    r"""A dual-rail Hadamard gate.

    Locally equivalent to ECP, `Can(1/2, 1/4, 1/4)`.

   .. math::
        \text{W} \equiv \begin{pmatrix} 1&0&0&0 \\
        0&\tfrac{1}{\sqrt{2}}&\tfrac{1}{\sqrt{2}}&0 \\
        0&\tfrac{1}{\sqrt{2}}&-\tfrac{1}{\sqrt{2}}&0 \\
        0&0&0&1 \end{pmatrix}

    ::
        ───X───●───●───●───X───
           │   │   │   │   │
        ───●───X───H───X───●───

    Notably, the W gate diagonalizes the swap gate.
    ::
        ──x──   ───W_q0───●───Z───W_q0───
          │   =    │      │       │
        ──x──   ───W_q1───●───────W_q1───

    Refs:
        https://arxiv.org/pdf/quant-ph/0209131.pdf
        https://arxiv.org/pdf/1206.0758v3.pdf
        https://arxiv.org/pdf/1505.06552.pdf

    """
    # Kudos: Adapted from Quipper
    cv_hermitian = True
    cv_sym_operator = sym.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1 / sym.sqrt(2), 1 / sym.sqrt(2), 0],
            [0, 1 / sym.sqrt(2), -1 / sym.sqrt(2), 0],
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(q0, q1)

    @property
    def H(self) -> "W":
        return self


# end class W


class XY(StdGate):
    r"""XY interaction gate.

    Powers of the iSWAP gate. Equivalent to ``Can(t, t, 0)``.
    """
    # https://arxiv.org/abs/1912.04424v1
    cv_interchangeable = True
    cv_sym_operator = XX.cv_sym_operator @ YY.cv_sym_operator

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(t, q0, q1)

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return self.t * (X(q0) * X(q1) + Y(q0) * Y(q1)) * sym.pi / 2

    @property
    def H(self) -> "XY":
        return self ** -1

    def __pow__(self, t: Variable) -> "XY":
        return XY(t * self.t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["XY^{t}"] * 2


# end class XY


CNot: TypeAlias = CX  # FIXME: MOVE

CNotPow: TypeAlias = CXPow  # FIXME: MOVE


# fin
