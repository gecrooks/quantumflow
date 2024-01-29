# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Standard two qubit gates
"""

from typing import List

import numpy as np

from .. import tensors, utils, var
from ..config import CTRL, SWAP_TARGET, TARGET
from ..future import cached_property
from ..ops import UnitaryGate
from ..paulialgebra import Pauli, sX, sY, sZ
from ..qubits import Qubit
from ..states import State
from ..tensors import QubitTensor
from ..var import PI, Variable
from .stdgates import StdCtrlGate, StdGate
from .stdgates_1q import V_H, H, I, S, T, V, X, XPow, Y, YPow, Z, ZPow

_H = H

# 2 qubit gates, alphabetic order

__all__ = (
    "A",
    "B",
    "Barenco",
    "Can",
    "CH",
    "CNot",
    "CNotPow",
    "CrossResonance",
    "CS",
    "CT",
    "CV",
    "CV_H",
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
    "W",
    "XX",
    "XY",
    "YY",
    "ZZ",
)


class A(StdGate):
    """The A gate. A 2-qubit, 2-parameter gate that is locally
    equivalent to Can(1/2, t, t)

    Refs:
        :cite:`Barkoutsos2018a`

        :cite:`Gard2020a`
    """

    cv_hermitian = True

    def __init__(self, theta: Variable, phi: Variable, q0: Qubit, q1: Qubit):
        super().__init__(params=[theta, phi], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        phi = self.float_param("phi")
        theta = self.float_param("theta")

        U = [
            [1, 0, 0, 0],
            [0, np.cos(theta), np.exp(1j * phi) * np.sin(theta), 0],
            [0, np.exp(-1j * phi) * np.sin(theta), -np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
        return tensors.asqutensor(U)

    @property
    def H(self) -> "A":
        return self


class B(StdGate):
    """The B (Berkeley) gate. Equivalent to Can(-1/2, -1/4, 0)"""

    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit):
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(-1 / 2, -1 / 4, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        U = (
            np.asarray(
                [
                    [1 + np.sqrt(2), 0, 0, 1j],
                    [0, 1, 1j * (1 + np.sqrt(2)), 0],
                    [0, 1j * (1 + np.sqrt(2)), 1, 0],
                    [1j, 0, 0, 1 + np.sqrt(2)],
                ]
            )
            * np.sqrt(2 - np.sqrt(2))
            / 2
        )
        return tensors.asqutensor(U)

    @property
    def H(self) -> "Can":
        return self**-1

    def __pow__(self, t: Variable) -> "Can":
        return Can(-t / 2, -t / 4, 0, *self.qubits)


class Barenco(StdGate):
    """A universal two-qubit gate:

    Ref:
        A Universal Two–Bit Gate for Quantum Computation, A. Barenco (1996)
        https://arxiv.org/pdf/quant-ph/9505016.pdf
    """

    # Note: parameter order as defined by original paper
    def __init__(
        self,
        phi: Variable,
        alpha: Variable,
        theta: Variable,
        q0: Qubit,
        q1: Qubit,
    ) -> None:
        super().__init__(params=[phi, alpha, theta], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        phi = self.float_param("phi")
        alpha = self.float_param("alpha")
        theta = self.float_param("theta")

        unitary = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [
                0,
                0,
                np.exp(1j * alpha) * np.cos(theta),
                -1j * np.exp(1j * (alpha - phi)) * np.sin(theta),
            ],
            [
                0,
                0,
                -1j * np.exp(1j * (alpha + phi)) * np.sin(theta),
                np.exp(1j * alpha) * np.cos(theta),
            ],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Barenco":
        phi, alpha, theta = self.params
        alpha = -alpha
        phi = phi + PI
        return Barenco(phi, alpha, theta, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["───" + CTRL + "───", "Barenco({phi}, {alpha}, {theta})"]


# TODO: Add references and explanation
# DOCME: Comment on sign conventions.
class Can(StdGate):
    r"""The canonical 2-qubit gate

    The canonical decomposition of 2-qubits gates removes local 1-qubit
    rotations, and leaves only the non-local interactions.

    .. math::
        \text{CAN}(t_x, t_y, t_z) \equiv
            \exp\Big\{-i\frac{\pi}{2}(t_x X\otimes X
            + t_y Y\otimes Y + t_z Z\otimes Z)\Big\}

    """

    cv_interchangeable = True

    def __init__(
        self, tx: Variable, ty: Variable, tz: Variable, q0: Qubit, q1: Qubit
    ) -> None:
        super().__init__(params=[tx, ty, tz], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        tx, ty, tz = self.params
        q0, q1 = self.qubits
        return (
            (tx * sX(q0) * sX(q1) + ty * sY(q0) * sY(q1) + tz * sZ(q0) * sZ(q1))
            * PI
            / 2
        )

    @cached_property
    def tensor(self) -> QubitTensor:
        tx, ty, tz = [var.asfloat(v) for v in self.params]
        xx = XX(tx, 0, 1)
        yy = YY(ty, 0, 1)
        zz = ZZ(tz, 0, 1)

        gate = yy @ xx
        gate = zz @ gate
        return gate.tensor

    @property
    def H(self) -> "Can":
        return self**-1

    def __pow__(self, t: Variable) -> "Can":
        tx, ty, tz = self.params
        return Can(tx * t, ty * t, tz * t, *self.qubits)

    # TODO: XY, ECP, ...
    def specialize(self) -> StdGate:
        qbs = self.qubits
        tx, ty, tz = [t % 2 for t in self.params]

        tx_zero = var.isclose(tx, 0.0) or var.isclose(tx, 2.0)
        ty_zero = var.isclose(ty, 0.0) or var.isclose(ty, 2.0)
        tz_zero = var.isclose(tz, 0.0) or var.isclose(tz, 2.0)

        if ty_zero and tz_zero:
            return XX(tx, *qbs).specialize()
        elif tx_zero and tz_zero:
            return YY(ty, *qbs).specialize()
        elif tx_zero and ty_zero:
            return ZZ(tz, *qbs).specialize()
        elif np.isclose(tx, ty) and np.isclose(tx, tz):
            return Exch(tx, *qbs).specialize()
        return self


# end class Can


class CH(StdCtrlGate):
    r"""A controlled-Hadamard gate

    Equivalent to ``controlled_gate(H())`` and locally equivalent to
    ``Can(1/2, 0, 0)``

    .. math::
        \text{CH}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \tfrac{1}{\sqrt{2}} &  \tfrac{1}{\sqrt{2}} \\
                0 & 0 & \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}
            \end{pmatrix}
    """

    cv_target = _H
    cv_hermitian = True

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def H(self) -> "CH":
        return self  # Hermitian

    # TODO: __pow__


# end class CH


class CNot(StdCtrlGate):
    r"""A controlled-not gate

    Equivalent to ``controlled_gate(X())``, and
    locally equivalent to ``Can(1/2, 0, 0)``

     .. math::
        \text{CNot}() \equiv \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                            0&0&0&1 \\ 0&0&1&0 \end{pmatrix}
    """

    cv_target = X
    cv_hermitian = True
    cv_tensor_structure = "permutation"

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def H(self) -> "CNot":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CNotPow":
        return CNotPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        axes = ket.qubit_indices(self.qubits)
        s10 = utils.multi_slice(axes, [1, 0])
        s11 = utils.multi_slice(axes, [1, 1])
        tensor = ket.tensor.copy()
        tensor[s10] = ket.tensor[s11]
        tensor[s11] = ket.tensor[s10]
        return State(tensor, ket.qubits, ket.memory)

    def _diagram_labels_(self) -> List[str]:
        return [CTRL, TARGET]


# end class CNot


# FIXME: matrix in docs looks wrong?
# FIXME: docs, controlled_gate
class CNotPow(StdCtrlGate):
    r"""Powers of the CNot gate.

    Equivalent to ``controlled_gate(TX(t))``, and locally equivalent to
    ``Can(t/2, 0 ,0)``.

    .. math::
        \text{CTX}(t) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos(\frac{\theta}{2}) e^{i\frac{\theta}{2}}
                      & -i \sin(\frac{\theta}{2}) e^{i\frac{\theta}{2}} \\
                0 & 0 & -i \sin(\frac{\theta}{2}) e^{i\frac{\theta}{2}}
                      & \cos(\frac{\theta}{2}) e^{i\frac{\theta}{2}}
            \end{pmatrix}

    args:
        t:  turns (powers of the CNot gate, or controlled-powers of X)
        q0: control qubit
        q1: target qubit
    """

    cv_target = XPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def H(self) -> "CNotPow":
        return self**-1

    def __pow__(self, e: Variable) -> "CNotPow":
        (t,) = self.params
        return CNotPow(e * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (t,) = self.params
        t %= 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return I(qbs[0])
        elif np.isclose(t, 1.0):
            return CNot(*qbs)
        return self


# end class CNotPow


class CrossResonance(StdGate):
    # DOCME
    # TESTME

    def __init__(self, s: Variable, b: Variable, c: Variable, q0: Qubit, q1: Qubit):
        super().__init__(qubits=(q0, q1), params=[s, b, c])

    @property
    def hamiltonian(self) -> Pauli:
        s, b, c = self.params
        q0, q1 = self.qubits
        return s * (sX(q0) - b * sZ(q0) * sX(q1) + c * sX(q1)) * PI / 2

    @cached_property
    def tensor(self) -> QubitTensor:
        U = UnitaryGate.from_hamiltonian(self.hamiltonian, self.qubits)
        return tensors.asqutensor(U.tensor)

    @property
    def H(self) -> "CrossResonance":
        return self**-1

    # TESTME
    def __pow__(self, t: Variable) -> "CrossResonance":
        s, b, c = self.params
        return CrossResonance(t * s, b, c, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["CR({s}, {b}, {c})_0", "CR({s}, {b}, {c})_1"]


# end class CrossResonance


class CS(StdCtrlGate):
    r"""A controlled-S gate

    .. math::
        \text{CS}() = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                    0&0&1&0 \\ 0&0&0&i \end{pmatrix}
    """

    cv_target = S
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    def __pow__(self, t: Variable) -> "CZPow":
        return CZPow(t / 2, *self.qubits)


# End class CS


class CT(StdCtrlGate):
    r"""A controlled-T gate

    .. math::
        \text{CT}() = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                    0&0&1&0 \\ 0&0&0&  \end{pmatrix}
    """

    cv_target = T
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    def __pow__(self, t: Variable) -> "CZPow":
        return CZPow(t / 4, *self.qubits)


# End class CS


class CY(StdCtrlGate):
    r"""A controlled-Y gate

    Equivalent to ``controlled_gate(Y())`` and locally equivalent to
    ``Can(1/2,0,0)``

    .. math::
        \text{CY}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & i & 0
            \end{pmatrix}
    """

    cv_target = Y
    cv_hermitian = True
    cv_tensor_structure = "monomial"

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def H(self) -> "CY":
        return self  # Hermitian

    def __pow__(self, e: Variable) -> "CYPow":
        return CYPow(e, *self.qubits)


# end class CY


class CYPow(StdCtrlGate):
    r"""Powers of the controlled-Y gate

    Locally equivalent to ``Can(t/2, 0, 0)``

    """

    cv_target = YPow

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def H(self) -> "CYPow":
        return self**-1

    def __pow__(self, e: Variable) -> "CYPow":
        (t,) = self.params
        return CYPow(e * t, *self.qubits)


# End class CYPow


class CV(StdCtrlGate):
    r"""A controlled V (sqrt of CNOT) gate."""

    cv_target = V

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def H(self) -> "CV_H":
        return CV_H(*self.qubits)

    def __pow__(self, t: Variable) -> "CNotPow":
        return CNotPow(t / 2, *self.qubits)


# end class CV


class CV_H(StdCtrlGate):
    r"""A controlled V (sqrt of CNOT) gate."""

    cv_target = V_H

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def H(self) -> "CV":
        return CV(*self.qubits)

    def __pow__(self, t: Variable) -> "CNotPow":
        return CNotPow(-t / 2, *self.qubits)


# end class CV_H


class CZ(StdCtrlGate):
    r"""A controlled-Z gate

    Equivalent to ``controlled_gate(Z())`` and locally equivalent to
    ``Can(1/2, 0, 0)``

    .. math::
        \text{CZ}() = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                    0&0&1&0 \\ 0&0&0&-1 \end{pmatrix}
    """

    cv_target = Z
    cv_hermitian = True
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def H(self) -> "CZ":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CZPow":
        return CZPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        axes = ket.qubit_indices(self.qubits)
        s11 = utils.multi_slice(axes, [1, 1])
        tensor = ket.tensor.copy()
        tensor[s11] *= -1
        return State(tensor, ket.qubits, ket.memory)

    def _diagram_labels_(self) -> List[str]:
        return [CTRL, CTRL]


# End class CZ


class CZPow(StdCtrlGate):
    r"""Powers of the controlled-Z gate

    Locally equivalent to ``Can(t/2, 0, 0)``

    .. math::
        \text{CZ}^t = \begin{pmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 \\
                        0 & 0 & 1 & 0 \\
                        0 & 0 & 0 & \exp(i \pi t)
                      \end{pmatrix}
    """

    cv_target = ZPow
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def H(self) -> "CZPow":
        return self**-1

    def __pow__(self, e: Variable) -> "CZPow":
        (t,) = self.params
        return CZPow(e * t, *self.qubits)


# End class CZPow


# TESTME
# DOCME
# TODO: Add citations x2
class ECP(StdGate):
    r"""The ECP gate. The peak of the pyramid of gates in the Weyl chamber
    that can be created with a square-root of iSWAP sandwich.

    Equivalent to ``Can(1/2, 1/4, 1/4)``.
    """

    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(1 / 2, 1 / 4, 1 / 4, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        return Can(1 / 2, 1 / 4, 1 / 4, *self.qubits).tensor

    @property
    def H(self) -> "Can":
        return self**-1

    def __pow__(self, t: Variable) -> "Can":
        return Can(t / 2, t / 4, t / 4, *self.qubits)


# end class ECP


class Exch(StdGate):
    r"""A 2-qubit parametric gate generated from an isotropic exchange interaction.

    Equivalent to Can(t,t,t)

    """

    cv_interchangeable = True

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        return Can(t, t, t, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        (t,) = self.params
        return Can(t, t, t, *self.qubits).tensor

    @property
    def H(self) -> "Exch":
        return self**-1

    def __pow__(self, e: Variable) -> "Exch":
        (t,) = self.params
        return Exch(e * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (t,) = self.params
        t %= 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return I(qbs[0])
        return self


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
        \text{Givens}(\theta) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\theta) & -\sin(\theta) & 0 \\
                0 & -sin(\theta)  & \cos(\theta) & 0 \\
                0 & 0 & 0 & 1)}
            \end{pmatrix}

    """

    # Kudos: Adapted from cirq

    def __init__(self, theta: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        (theta,) = self.params
        q0, q1 = self.qubits
        return theta * (sY(q0) * sX(q1) - sX(q0) * sY(q1)) / 2

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Givens":
        return self**-1

    def __pow__(self, t: Variable) -> "Givens":
        (theta,) = self.params
        return Givens(t * theta, *self.qubits)


# end class Givens


class ISwap(StdGate):
    r"""A 2-qubit iSwap gate

    Equivalent to ``Can(-1/2,-1/2,0)``.

    .. math::
        \text{ISWAP}() \equiv
        \begin{pmatrix} 1&0&0&0 \\ 0&0&i&0 \\ 0&i&0&0 \\ 0&0&0&1 \end{pmatrix}

    """

    cv_interchangeable = True
    cv_tensor_structure = "monomial"

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(-1 / 2, -1 / 2, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        unitary = np.asarray([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

        return tensors.asqutensor(unitary)

    def run(self, ket: State) -> State:
        tensor = ket.tensor.copy()
        axes = ket.qubit_indices(self.qubits)
        s10 = utils.multi_slice(axes, [1, 0])
        s01 = utils.multi_slice(axes, [0, 1])
        tensor[s01] = 1.0j * ket.tensor[s10]
        tensor[s10] = 1.0j * ket.tensor[s01]
        return State(tensor, ket.qubits, ket.memory)

    @property
    def H(self) -> "XY":
        return self**-1

    def __pow__(self, t: Variable) -> "XY":
        return XY(-t / 2, *self.qubits)


# end class ISWAP


# TESTME
class SqrtISwap(StdGate):
    r"""A square root of the iswap gate

    Equivalent to ``Can(-1/4,-1/4,0)``.
    """

    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(-1 / 4, -1 / 4, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        return Can(-1 / 4, -1 / 4, 0, *self.qubits).tensor

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

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(1 / 4, 1 / 4, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        return Can(1 / 4, 1 / 4, 0, *self.qubits).tensor

    @property
    def H(self) -> "SqrtISwap":
        return SqrtISwap(*self.qubits)

    def __pow__(self, t: Variable) -> "XY":
        return XY(t / 4, *self.qubits)


# end class SqrtISwap_H


# TESTME
class SqrtSwap(StdGate):
    r"""Square root of the 2-qubit swap gate

    Equivalent to ``Can(1/4, 1/4, 1/4)``.
    """

    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(1 / 4, 1 / 4, 1 / 4, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        return Can(1 / 4, 1 / 4, 1 / 4, *self.qubits).tensor

    @property
    def H(self) -> "SqrtSwap_H":
        return SqrtSwap_H(*self.qubits)

    def __pow__(self, t: Variable) -> "Exch":
        return Exch(t / 4, *self.qubits)


# end class SqrtSwap


# TESTME
class SqrtSwap_H(StdGate):
    r"""The conjugate of the Square root swap gate

    Equivalent to ``Can(-1/4, -1/4, -1/4)``, and locally equivalent to
    ``Can(3/4, 1/4, 1/4)``
    """

    cv_interchangeable = True

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(-1 / 4, -1 / 4, -1 / 4, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        return Can(-1 / 4, -1 / 4, -1 / 4, *self.qubits).tensor

    @property
    def H(self) -> "SqrtSwap":
        return SqrtSwap(*self.qubits)

    def __pow__(self, t: Variable) -> "Exch":
        return Exch(-t / 4, *self.qubits)


# end class SqrtSwap_H


class Swap(StdGate):
    r"""A 2-qubit swap gate

    Equivalent to ``Can(1/2, 1/2, 1/2)``.

    .. math::
        \text{SWAP}() \equiv
            \begin{pmatrix}
            1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1
            \end{pmatrix}

    """

    cv_interchangeable = True
    cv_hermitian = True
    cv_tensor_structure = "permutation"

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return (sX(q0) * sX(q1) + sY(q0) * sY(q1) + sZ(q0) * sZ(q1) - 1) * PI / 4

    @cached_property
    def tensor(self) -> QubitTensor:
        unitary = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Swap":
        return self  # Hermitian

    # TESTME
    def __pow__(self, t: Variable) -> "Exch":
        return Exch(t / 2, *self.qubits)

    def run(self, ket: State) -> State:
        idx0, idx1 = ket.qubit_indices(self.qubits)
        perm = list(range(ket.qubit_nb))
        perm[idx0] = idx1
        perm[idx1] = idx0
        tensor = np.transpose(ket.tensor, perm)
        return State(tensor, ket.qubits, ket.memory)

    def _diagram_labels_(self) -> List[str]:
        return [SWAP_TARGET] * 2


# end class SWAP


# TESTME DOCME
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

    def __init__(self, q0: Qubit, q1: Qubit) -> None:
        super().__init__(qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        rs2 = 1 / np.sqrt(2)
        unitary = [[1, 0, 0, 0], [0, rs2, rs2, 0], [0, rs2, -rs2, 0], [0, 0, 0, 1]]

        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "W":
        return self


# end class W


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

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        return Can(t, 0, 0, *self.qubits).hamiltonian

    # FIXME: Phase
    @cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * var.asfloat(self.param("t"))
        unitary = [
            [np.cos(theta / 2), 0, 0, -1.0j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1.0j * np.sin(theta / 2), 0],
            [0, -1.0j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [-1.0j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "XX":
        return self**-1

    def __pow__(self, e: Variable) -> "XX":
        (t,) = self.params
        return XX(e * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (t,) = self.params
        t %= 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return I(qbs[0])
        return self

    def _diagram_labels_(self) -> List[str]:
        return ["XX^{t}"] * 2


class XY(StdGate):
    r"""XY interaction gate.

    Powers of the iSWAP gate. Equivalent to ``Can(t, t, 0)``.
    """

    # https://arxiv.org/abs/1912.04424v1
    cv_interchangeable = True

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        q0, q1 = self.qubits
        return t * (sX(q0) * sX(q1) + sY(q0) * sY(q1)) * PI / 2

    @cached_property
    def tensor(self) -> QubitTensor:
        t = var.asfloat(self.param("t"))
        return Can(t, t, 0, *self.qubits).tensor

    @property
    def H(self) -> "XY":
        return self**-1

    def __pow__(self, e: Variable) -> "XY":
        (t,) = self.params
        return XY(e * t, *self.qubits)

    def _diagram_labels_(self) -> List[str]:
        return ["XY^{t}"] * 2


class YY(StdGate):
    r"""A parametric 2-qubit gate generated from a YY interaction.

    Equivalent to ``Can(0, t, 0)``, and locally equivalent to
    ``Can(t, 0, 0)``

    Args:
        t:
    """

    cv_interchangeable = True

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        q0, q1 = self.qubits
        return t * sY(q0) * sY(q1) * PI / 2

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * var.asfloat(self.param("t"))
        unitary = [
            [np.cos(theta / 2), 0, 0, 1.0j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1.0j * np.sin(theta / 2), 0],
            [0, -1.0j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [1.0j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "YY":
        return self**-1

    def __pow__(self, e: Variable) -> "YY":
        (t,) = self.params
        return YY(e * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (t,) = self.params
        t %= 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return I(qbs[0])
        return self

    def _diagram_labels_(self) -> List[str]:
        return ["YY^{t}"] * 2


class ZZ(StdGate):
    r"""A parametric 2-qubit gate generated from a ZZ interaction.

    Equivalent to ``Can(0,0,t)``, and locally equivalent to
    ``Can(t,0,0)``

    Args:
        t:
    """

    cv_interchangeable = True
    cv_tensor_structure = "diagonal"

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        q0, q1 = self.qubits
        return t * sZ(q0) * sZ(q1) * PI / 2

    # FIXME: Phase?
    @cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * var.asfloat(self.param("t"))
        unitary = [
            [
                [[np.exp(-1j * theta / 2), 0], [0, 0]],
                [[0, np.exp(1j * theta / 2)], [0, 0]],
            ],
            [
                [[0, 0], [np.exp(1j * theta / 2), 0]],
                [[0, 0], [0, np.exp(-1j * theta / 2)]],
            ],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "ZZ":
        return self**-1

    def __pow__(self, e: Variable) -> "ZZ":
        (t,) = self.params
        return ZZ(e * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (t,) = self.params
        # if variable_is_symbolic(t):
        #     return self
        t = t % 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return I(qbs[0])
        return self

    def _diagram_labels_(self) -> List[str]:
        return ["ZZ^{t}"] * 2


# end class ZZ

# fin
