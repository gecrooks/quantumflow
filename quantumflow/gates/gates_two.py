# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: One qubit gates
"""

# Replacing numpy pi with sympi pi makes tests x3 slower!?
# So we use symbolic PI sparingly

from numpy import pi
from sympy import pi as PI
# from numpy import pi as PI

import numpy as np
import scipy
from typing import Iterator, Union

from .. import backend as bk
from ..qubits import Qubit
# from ..variables import variable_is_symbolic
from ..ops import Gate
from ..states import State
from ..utils import multi_slice, cached_property
from ..variables import Variable
from ..config import CTRL, TARGET, SWAP_TARGET, SQRT, CONJ
from ..paulialgebra import Pauli, sX, sY, sZ

from .gates_one import IDEN, V, V_H, H, X, Z, Y, S, S_H
from .gates_utils import control_gate


# 2 qubit gates, alphabetic order

__all__ = (
    'B', 'Barenco', 'Can', 'CH', 'CNOT', 'CNotPow',
    'CrossResonance', 'CV', 'CV_H', 'CY', 'CYPow', 'CZ', 'CZPow', 'ECP',
    'EXCH', 'Givens',
    'ISWAP', 'SqrtISwap', 'SqrtISwap_H', 'SqrtSwap', 'SqrtSwap_H',
    'SWAP', 'W', 'XX', 'XY', 'YY', 'ZZ',
    # Deprecated
    'CAN',
    )


class B(Gate):
    """ The B (Berkeley) gate. Equivalent to CAN(-1/2, -1/4, 0)
    """
    interchangeable = True

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1):
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return CAN(-1/2, -1/4, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        U = np.asarray([[1+np.sqrt(2), 0, 0, 1j],
                       [0, 1, 1j*(1+np.sqrt(2)), 0],
                       [0, 1j*(1+np.sqrt(2)), 1, 0],
                       [1j, 0, 0, 1+np.sqrt(2)]]) * np.sqrt(2-np.sqrt(2))/2
        return bk.astensorproduct(U)

    @property
    def H(self) -> 'CAN':
        return self ** -1

    def __pow__(self, t: Variable) -> 'CAN':
        return CAN(-t/2, -t/4, 0, *self.qubits)


class Barenco(Gate):
    """A universal two-qubit gate:

    Ref:
        A Universal Two–Bit Gate for Quantum Computation, A. Barenco (1996)
        https://arxiv.org/pdf/quant-ph/9505016.pdf
    """
    _diagram_labels = ['───'+CTRL+'───', 'Barenco({phi}, {alpha}, {theta})']

    # Note: parameter order as defined by original paper
    def __init__(self,
                 phi: Variable,
                 alpha: Variable,
                 theta: Variable,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        params = dict(phi=phi, alpha=alpha, theta=theta)
        qubits = [q0, q1]
        super().__init__(params=params, qubits=qubits)

    @cached_property
    def tensor(self) -> bk.BKTensor:
        phi, alpha, theta = self.parameters()

        calpha = bk.ccast(alpha)
        cphi = bk.ccast(phi)
        ctheta = bk.ccast(theta)

        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, bk.exp(1j*calpha) * bk.cos(ctheta),
                    -1j * bk.exp(1j*(calpha - cphi)) * bk.sin(ctheta)],
                   [0, 0, -1j * bk.exp(1j*(calpha + cphi)) * bk.sin(ctheta),
                    bk.exp(1j*calpha) * bk.cos(ctheta)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'Barenco':
        phi, alpha, theta = self.parameters()
        alpha = - alpha
        phi = phi + pi
        return Barenco(phi, alpha, theta, *self.qubits)


# TODO: Add references and explanation
# DOCME: Comment on sign conventions.
class Can(Gate):
    r"""The canonical 2-qubit gate

    The canonical decomposition of 2-qubits gates removes local 1-qubit
    rotations, and leaves only the non-local interactions.

    .. math::
        \text{CAN}(t_x, t_y, t_z) \equiv
            \exp\Big\{-i\frac{\pi}{2}(t_x X\otimes X
            + t_y Y\otimes Y + t_z Z\otimes Z)\Big\}

    """
    interchangeable = True

    def __init__(self,
                 tx: Variable, ty: Variable, tz: Variable,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(tx=tx, ty=ty, tz=tz), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        tx, ty, tz = self.parameters()
        q0, q1 = self.qubits
        return (tx*sX(q0)*sX(q1) + ty*sY(q0)*sY(q1) + tz*sZ(q0)*sZ(q1)) * PI/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        tx, ty, tz = self.parameters()
        xx = XX(tx)
        yy = YY(ty)
        zz = ZZ(tz)

        gate = yy @ xx
        gate = zz @ gate
        return gate.tensor

    @property
    def H(self) -> 'Can':
        return self ** -1

    def __pow__(self, t: Variable) -> 'Can':
        tx, ty, tz = self.parameters()
        return Can(tx * t, ty * t, tz * t, *self.qubits)

    # FIXME: Won't work with Symbolic parameters
    # TODO: Rename
    # TODO: XY, ECP, ...
    def specialize(self) -> Gate:
        qbs = self.qubits
        tx, ty, tz = [t % 2 for t in self.parameters()]

        tx_zero = (np.isclose(tx, 0.0) or np.isclose(tx, 2.0))
        ty_zero = (np.isclose(ty, 0.0) or np.isclose(ty, 2.0))
        tz_zero = (np.isclose(tz, 0.0) or np.isclose(tz, 2.0))

        if ty_zero and tz_zero:
            return XX(tx, *qbs).specialize()
        elif tx_zero and tz_zero:
            return YY(ty, *qbs).specialize()
        elif tx_zero and ty_zero:
            return ZZ(tz, *qbs).specialize()
        elif np.isclose(tx, ty) and np.isclose(tx, tz):
            return EXCH(tx, *qbs).specialize()
        return self

# end class Can


CAN = Can
# FIXME: Remove (Backwards compatibility)
# def CAN(*args, **kwargs) -> Can:
#     return Can(*args, **kwargs)


class CH(Gate):
    r"""A controlled-Hadamard gate

    Equivalent to ``controlled_gate(H())`` and locally equivalent to
    ``CAN(1/2, 0, 0)``

    .. math::
        \text{CH}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \tfrac{1}{\sqrt{2}} &  \tfrac{1}{\sqrt{2}} \\
                0 & 0 & \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}
            \end{pmatrix}
    """
    _diagram_labels = [CTRL, 'H']

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return H(q1).hamiltonian * (1-sZ(q0))/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                              [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CH':
        return self  # Hermitian

    # TODO: __pow__
# end class CH


class CNOT(Gate):
    r"""A controlled-not gate

    Equivalent to ``controlled_gate(X())``, and
    locally equivalent to ``CAN(1/2, 0, 0)``

     .. math::
        \text{CNOT}() \equiv \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                            0&0&0&1 \\ 0&0&1&0 \end{pmatrix}
    """
    permutation = True
    _diagram_labels = [CTRL, TARGET]

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return X(q1).hamiltonian * (1-sZ(q0))/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CNOT':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'CNotPow':
        return CNotPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s10 = multi_slice(axes, [1, 0])
            s11 = multi_slice(axes, [1, 1])
            tensor = ket.tensor.copy()
            tensor[s10] = ket.tensor[s11]
            tensor[s11] = ket.tensor[s10]
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover

# end class CNOT


# FIXME: matrix in docs looks wrong?
class CNotPow(Gate):
    r"""Powers of the CNOT gate.

    Equivalent to ``controlled_gate(TX(t))``, and locally equivalent to
    ``CAN(t/2, 0 ,0)``.

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
        t:  turns (powers of the CNOT gate, or controlled-powers of X)
        q0: control qubit
        q1: target qubit
    """

    _diagram_labels = [CTRL, 'X^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        return CNOT(*self.qubits).hamiltonian * t

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        cht = bk.cos(ctheta / 2)
        sht = bk.sin(ctheta / 2)
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, phase * cht, phase * -1.0j * sht],
                   [0, 0, phase * -1.0j * sht, phase * cht]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CNotPow':
        return self ** -1

    def __pow__(self, e: Variable) -> 'CNotPow':
        t, = self.parameters()
        return CNotPow(e*t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        t, = self.parameters()
        t %= 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return IDEN(*qbs)
        elif np.isclose(t, 1.0):
            return CNOT(*qbs)
        return self

# end class CNotPow


class CrossResonance(Gate):
    # DOCME
    # TESTME

    _diagram_labels = ['CR({s}, {b}, {c})_q0', 'CR({s}, {b}, {c})_q1']

    def __init__(self,
                 s: float,
                 b: float,
                 c: float,
                 q0: Qubit = 0,
                 q1: Qubit = 1):
        super().__init__(qubits=(q0, q1), params=dict(s=s, b=b, c=c))

    @property
    def hamiltonian(self) -> Pauli:
        s, b, c = self.parameters()
        q0, q1 = self.qubits
        return s*(sX(q0) - b * sZ(q0) * sX(q1) + c * sX(q1)) * PI/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        gen = self.hamiltonian.asoperator(self.qubits)
        U = scipy.linalg.expm(-1j * gen)
        return bk.astensorproduct(U)

    @property
    def H(self) -> 'CrossResonance':
        return self ** -1

    # TESTME
    def __pow__(self, t: Variable) -> 'CrossResonance':
        s, b, c = self.parameters()
        return CrossResonance(t*s, b, c, *self.qubits)

# end class CrossResonance


class CY(Gate):
    r"""A controlled-Y gate

    Equivalent to ``controlled_gate(Y())`` and locally equivalent to
    ``CAN(1/2,0,0)``

    .. math::
        \text{CY}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & i & 0
            \end{pmatrix}
    """
    monomial = True
    _diagram_labels = [CTRL, 'Y']

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return Y(q1).hamiltonian * (1-sZ(q0))/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, -1j],
                              [0, 0, 1j, 0]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CY':
        return self  # Hermitian

    def __pow__(self, e: Variable) -> 'CYPow':
        return CYPow(e, *self.qubits)

# end class CY


class CYPow(Gate):
    r"""Powers of the controlled-Y gate

    Locally equivalent to ``CAN(t/2, 0, 0)``

    """
    _diagram_labels = [CTRL, 'Y^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        return CY(*self.qubits).hamiltonian * t

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, phase * bk.cos(ctheta / 2.0),
                    phase * -bk.sin(ctheta / 2.0)],
                   [0, 0, phase * bk.sin(ctheta / 2.0),
                    phase * bk.cos(ctheta / 2.0)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CYPow':
        return self ** -1

    def __pow__(self, e: Variable) -> 'CYPow':
        t, = self.parameters()
        return CYPow(e*t, *self.qubits)

    def decompose(self) -> Iterator[Union[S, S_H, CNotPow]]:
        """Convert powers of CZ gate to powers of CNOT gate"""
        t, = self.parameters()
        q0, q1 = self.qubits
        yield S_H(q1)
        yield CNOT(q0, q1) ** t
        yield S(q1)

# End class CYPow


class CV(Gate):
    r"""A controlled V (sqrt of CNOT) gate."""

    interchangeable = True
    _diagram_labels = [CTRL, 'V']

    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return CNOT(*self.qubits).hamiltonian / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        q0, q1 = self.qubits
        return control_gate(q0, V(q1)).tensor

    @property
    def H(self) -> 'CV_H':
        return CV_H(*self.qubits)

    def __pow__(self, t: Variable) -> 'CNotPow':
        return CNotPow(t/2, *self.qubits)

# end class CV


class CV_H(Gate):
    r"""A controlled V (sqrt of CNOT) gate."""

    interchangeable = True
    _diagram_labels = [CTRL, 'V' + CONJ]

    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return -CNOT(*self.qubits).hamiltonian / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        q0, q1 = self.qubits
        return control_gate(q0, V_H(q1)).tensor

    @property
    def H(self) -> 'CV':
        return CV(*self.qubits)

    def __pow__(self, t: Variable) -> 'CNotPow':
        return CNotPow(-t/2, *self.qubits)

# end class CV_H


class CZ(Gate):
    r"""A controlled-Z gate

    Equivalent to ``controlled_gate(Z())`` and locally equivalent to
    ``CAN(1/2, 0, 0)``

    .. math::
        \text{CZ}() = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                    0&0&1&0 \\ 0&0&0&-1 \end{pmatrix}
    """
    interchangeable = True
    diagonal = True
    _diagram_labels = [CTRL, CTRL]

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return Z(q1).hamiltonian * (1-sZ(q0)) / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CZ':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'CZPow':
        return CZPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s11 = multi_slice(axes, [1, 1])
            tensor = ket.tensor.copy()
            # tensor[s11] = -ket.tensor[s11]
            tensor[s11] *= -1
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover
# End class CZ


class CZPow(Gate):
    r"""Powers of the controlled-Z gate

    Locally equivalent to ``CAN(t/2, 0, 0)``

    .. math::
        \text{CZ}^t = \begin{pmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 \\
                        0 & 0 & 1 & 0 \\
                        0 & 0 & 0 & \exp(i \pi t)
                      \end{pmatrix}
    """
    interchangeable = True
    diagonal = True
    _diagram_labels = [CTRL, 'Z^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        return CZ(*self.qubits).hamiltonian * t

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, bk.exp(1j*t*pi)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CZPow':
        return self ** -1

    def __pow__(self, e: Variable) -> 'CZPow':
        t, = self.parameters()
        return CZPow(e*t, *self.qubits)

# End class CZPow


# TESTME
# DOCME
# TODO: Add citations x2
class ECP(Gate):
    r"""The ECP gate. The peak of the pyramid of gates in the Weyl chamber
    that can be created with a square-root of iSWAP sandwich.

    Equivalent to ``Can(1/2, 1/4, 1/4)``.
    """
    interchangeable = True

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return Can(1/2, 1/4, 1/4, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return Can(1/2, 1/4, 1/4).tensor

    @property
    def H(self) -> 'Can':
        return self ** -1

    def __pow__(self, t: Variable) -> 'Can':
        return Can(t/2, t/4, t/4, *self.qubits)

# end class ECP


class EXCH(Gate):
    r"""A 2-qubit parametric gate generated from an exchange interaction.

    Equivalent to CAN(t,t,t)

    """
    interchangeable = True

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        return CAN(t, t, t, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        return CAN(t, t, t, *self.qubits).tensor

    @property
    def H(self) -> 'EXCH':
        return self ** -1

    def __pow__(self, e: Variable) -> 'EXCH':
        t, = self.parameters()
        return EXCH(e*t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        t, = self.parameters()
        t %= 2
        if (np.isclose(t, 0.0) or np.isclose(t, 2.0)):
            return IDEN(*qbs)
        return self

# end class Exch


class Givens(Gate):
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

    def __init__(self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        theta, = self.parameters()
        q0, q1 = self.qubits
        return theta * (sY(q0)*sX(q1) - sX(q0)*sY(q1))/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, = self.parameters()
        theta = bk.ccast(theta)
        unitary = [[1, 0, 0, 0],
                   [0, bk.cos(theta), -bk.sin(theta), 0],
                   [0, bk.sin(theta), bk.cos(theta), 0],
                   [0, 0, 0, 1]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'Givens':
        return self ** -1

    def __pow__(self, t: Variable) -> 'Givens':
        theta, = self.parameters()
        return Givens(t*theta, *self.qubits)

# end class Givens


class ISWAP(Gate):
    r"""A 2-qubit iSwap gate

    Equivalent to ``CAN(-1/2,-1/2,0)``.

    .. math::
        \text{ISWAP}() \equiv
        \begin{pmatrix} 1&0&0&0 \\ 0&0&i&0 \\ 0&i&0&0 \\ 0&0&0&1 \end{pmatrix}

    """
    interchangeable = True
    monomial = True
    _diagram_labels = ['iSwap']

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return CAN(-1/2, -1/2, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.array([[1, 0, 0, 0],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [0, 0, 0, 1]])

        return bk.astensorproduct(unitary)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            tensor = ket.tensor.copy()
            axes = ket.qubit_indices(self.qubits)
            s10 = multi_slice(axes, [1, 0])
            s01 = multi_slice(axes, [0, 1])
            tensor[s01] = 1.0j * ket.tensor[s10]
            tensor[s10] = 1.0j * ket.tensor[s01]
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover

    @property
    def H(self) -> 'XY':
        return self ** -1

    def __pow__(self, t: Variable) -> 'XY':
        return XY(-t/2, *self.qubits)

# end class ISWAP


# TESTME
class SqrtISwap(Gate):
    r"""A square root of the iswap gate

    Equivalent to ``CAN(-1/4,-1/4,0)``.
    """
    interchangeable = True
    _diagram_labels = [SQRT+"iSwap"]

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return CAN(-1/4, -1/4, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return CAN(-1/4, -1/4, 0).tensor

    @property
    def H(self) -> 'SqrtISwap_H':
        return SqrtISwap_H(*self.qubits)

    def __pow__(self, t: Variable) -> 'XY':
        return XY(-t/4, *self.qubits)

# end class SqrtISwap


# TESTME
class SqrtISwap_H(Gate):
    r"""The Hermitian conjugate of the square root iswap gate

    Equivalent to ``CAN(1/4, 1/4, 0)``.
    """
    interchangeable = True
    _diagram_labels = [SQRT+"iSwap"+CONJ]

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return CAN(1/4, 1/4, 0, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return CAN(1/4, 1/4, 0).tensor

    @property
    def H(self) -> 'SqrtISwap':
        return SqrtISwap(*self.qubits)

    def __pow__(self, t: Variable) -> 'XY':
        return XY(t/4, *self.qubits)

# end class SqrtISwap_H


# TESTME
class SqrtSwap(Gate):
    r"""Square root of the 2-qubit swap gate

    Equivalent to ``CAN(1/4, 1/4, 1/4)``.
    """
    interchangeable = True
    _diagram_labels = [SQRT+"Swap"]

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return CAN(1/4, 1/4, 1/4, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return CAN(1/4, 1/4, 1/4).tensor

    @property
    def H(self) -> 'SqrtSwap_H':
        return SqrtSwap_H(*self.qubits)

    def __pow__(self, t: Variable) -> 'EXCH':
        return EXCH(t/4, *self.qubits)

# end class SqrtSwap


# TESTME
class SqrtSwap_H(Gate):
    r"""The conjugate of the Square root swap gate

    Equivalent to ``CAN(-1/4, -1/4, -1/4)``, and locally equivalent to
    ``CAN(3/4, 1/4, 1/4)``
    """
    interchangeable = True
    _diagram_labels = [SQRT+"Swap"+CONJ]

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        return CAN(-1/4, -1/4, -1/4, *self.qubits).hamiltonian

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return CAN(-1/4, -1/4, -1/4).tensor

    @property
    def H(self) -> 'SqrtSwap':
        return SqrtSwap(*self.qubits)

    def __pow__(self, t: Variable) -> 'EXCH':
        return EXCH(-t/4, *self.qubits)

# end class SR_SWAP_H


class SWAP(Gate):
    r"""A 2-qubit swap gate

    Equivalent to ``CAN(1/2, 1/2, 1/2)``.

    .. math::
        \text{SWAP}() \equiv
            \begin{pmatrix}
            1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1
            \end{pmatrix}

    """
    interchangeable = True
    permutation = True
    _diagram_labels = [SWAP_TARGET, SWAP_TARGET]

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return (sX(q0)*sX(q1) + sY(q0)*sY(q1) + sZ(q0)*sZ(q1) - 1) * PI/4

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'SWAP':
        return self  # Hermitian

    # TESTME
    def __pow__(self, t: Variable) -> 'EXCH':
        return EXCH(t/2, *self.qubits)

    # TESTME DOCME
    # TODO: evolve
    def run(self, ket: State) -> State:
        idx0, idx1 = ket.qubit_indices(self.qubits)
        perm = list(range(ket.qubit_nb))
        perm[idx0] = idx1
        perm[idx1] = idx0
        tensor = bk.transpose(ket.tensor, perm)
        return State(tensor, ket.qubits, ket.memory)

# end class SWAP


# TESTME DOCME
class W(Gate):
    r"""A dual-rail Hadamard gate.

    Locally equivalent to ECP, `CAN(1/2, 1/4, 1/4)`.

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

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        rs2 = 1/np.sqrt(2)
        unitary = [[1, 0, 0, 0], [0, rs2, rs2, 0],
                   [0, rs2, -rs2, 0], [0, 0, 0, 1]]

        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'W':
        return self

# end class W


class XX(Gate):
    r"""A parametric 2-qubit gate generated from an XX interaction,

    Equivalent to ``CAN(t, 0, 0)``.

    XX(1/2) is the Mølmer-Sørensen gate.

    Ref: Sørensen, A. & Mølmer, K. Quantum computation with ions in thermal
    motion. Phys. Rev. Lett. 82, 1971–1974 (1999)

    Args:
        t:
    """
    # TODO: Is XX(1/2) MS gate, or is it XX(-1/2)???
    interchangeable = True
    _diagram_labels = ['XX^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        return Can(t, 0, 0, *self.qubits).hamiltonian

    # FIXME: Phase
    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        theta = bk.ccast(pi * t)
        unitary = [[bk.cos(theta / 2), 0, 0, -1.0j * bk.sin(theta / 2)],
                   [0, bk.cos(theta / 2), -1.0j * bk.sin(theta / 2), 0],
                   [0, -1.0j * bk.sin(theta / 2), bk.cos(theta / 2), 0],
                   [-1.0j * bk.sin(theta / 2), 0, 0, bk.cos(theta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'XX':
        return self ** -1

    def __pow__(self, e: Variable) -> 'XX':
        t, = self.parameters()
        return XX(e*t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        t, = self.parameters()
        t %= 2
        if (np.isclose(t, 0.0) or np.isclose(t, 2.0)):
            return IDEN(*qbs)
        return self


class XY(Gate):
    r"""XY interaction gate.

    Powers of the iSWAP gate. Equivalent to ``CAN(t, t, 0)``.
    """
    interchangeable = True
    _diagram_labels = ['XY^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        q0, q1 = self.qubits
        return t*(sX(q0)*sX(q1) + sY(q0)*sY(q1)) * PI/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        return CAN(t, t, 0).tensor

    @property
    def H(self) -> 'XY':
        return self ** -1

    def __pow__(self, e: Variable) -> 'XY':
        t, = self.parameters()
        return XY(e*t, *self.qubits)


class YY(Gate):
    r"""A parametric 2-qubit gate generated from a YY interaction.

    Equivalent to ``CAN(0, t, 0)``, and locally equivalent to
    ``CAN(t, 0, 0)``

    Args:
        t:
    """
    interchangeable = True
    _diagram_labels = ['YY^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        q0, q1 = self.qubits
        return t*sY(q0)*sY(q1) * PI/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        theta = bk.ccast(pi * t)
        unitary = [[bk.cos(theta / 2), 0, 0, 1.0j * bk.sin(theta / 2)],
                   [0, bk.cos(theta / 2), -1.0j * bk.sin(theta / 2), 0],
                   [0, -1.0j * bk.sin(theta / 2), bk.cos(theta / 2), 0],
                   [1.0j * bk.sin(theta / 2), 0, 0, bk.cos(theta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'YY':
        return self ** -1

    def __pow__(self, e: Variable) -> 'YY':
        t, = self.parameters()
        return YY(e*t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        t, = self.parameters()
        t %= 2
        if (np.isclose(t, 0.0) or np.isclose(t, 2.0)):
            return IDEN(*qbs)
        return self


class ZZ(Gate):
    r"""A parametric 2-qubit gate generated from a ZZ interaction.

    Equivalent to ``CAN(0,0,t)``, and locally equivalent to
    ``CAN(t,0,0)``

    Args:
        t:
    """
    interchangeable = True
    diagonal = True
    _diagram_labels = ['ZZ^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        q0, q1 = self.qubits
        return t*sZ(q0)*sZ(q1) * PI/2

    # FIXME: Phase
    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        theta = bk.ccast(pi * t)
        unitary = [[[[bk.exp(-1j*theta / 2), 0], [0, 0]],
                    [[0, bk.exp(1j*theta / 2)], [0, 0]]],
                   [[[0, 0], [bk.exp(1j*theta / 2), 0]],
                    [[0, 0], [0, bk.exp(-1j*theta / 2)]]]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'ZZ':
        return self ** -1

    def __pow__(self, e: Variable) -> 'ZZ':
        t, = self.parameters()
        return ZZ(e*t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        t, = self.parameters()
        # if variable_is_symbolic(t):
        #     return self
        t = t % 2
        if (np.isclose(t, 0.0) or np.isclose(t, 2.0)):
            return IDEN(*qbs)
        return self


# fin
