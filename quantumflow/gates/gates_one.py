
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: One qubit gates
"""

from math import sqrt, pi
from typing import Dict, List, Type, Iterator
import numpy as np

from numpy import pi as PI
# from sympy import pi as PI

from ..config import CONJ, SQRT
from .. import backend as bk
# from ..variables import variable_is_symbolic
from ..qubits import Qubit
from ..states import State, Density
from ..ops import Gate
from ..utils import multi_slice, cached_property
from ..variables import Variable
from ..paulialgebra import Pauli, sX, sY, sZ, sI

__all__ = (
    'IDEN', 'I', 'Ph',
    'X', 'Y', 'Z', 'H', 'S', 'T', 'PhaseShift',
    'RX', 'RY', 'RZ', 'RN', 'TX', 'TY', 'TZ', 'TH', 'S_H', 'T_H',
    'V', 'V_H', 'SqrtY', 'SqrtY_H')


def _specialize_gate(gate: Gate,
                     periods: List[float],
                     opts: Dict[float, Type[Gate]]) -> Gate:
    """Return a specialized instance of a given general gate. Used
    by the specialize code of various gates.

    args:
        gate:       The gate instance to specialize
        periods:    The periods of the gate's parameters. Gate parameters
                    are wrapped to range[0,period]
        opts:       A map from particular gate parameters to a special case
                    of the original gate type
    """
    params = list(gate.params.values())

    # for p in params:
    #     if variable_is_symbolic(p):
    #         return gate

    params = [p % pd for p, pd in zip(params, periods)]

    for values, gatetype in opts.items():
        if np.isclose(params, values):
            return gatetype(*gate.qubits)       # type: ignore

    return type(gate)(*params, *gate.qubits)    # type: ignore


# Standard 1 qubit gates

class I(Gate):                                      # noqa: E742
    r"""
    The 1-qubit identity gate.

    .. math::
        I() \equiv \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        return Pauli.zero()

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return bk.astensorproduct(np.eye(2))

    @property
    def H(self) -> 'I':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'I':
        return self

    def run(self, ket: State) -> State:
        return ket

    def evolve(self, rho: Density) -> Density:
        return rho


# TODO: Move to gate_utils? Or gates_three?
class IDEN(Gate):
    r"""
    The multi-qubit identity gate.
    """
    interchangeable = True
    diagonal = True
    _diagram_labels = ['I']
    _diagram_noline = True

    def __init__(self, *qubits: Qubit) -> None:
        if not qubits:
            qubits = (0,)
        super().__init__(qubits=qubits)

    @property
    def hamiltonian(self) -> Pauli:
        return Pauli.zero()

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return bk.astensorproduct(np.eye(2 ** self.qubit_nb))

    @property
    def H(self) -> 'IDEN':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'IDEN':
        return self

    def run(self, ket: State) -> State:
        return ket

    def evolve(self, rho: Density) -> Density:
        return rho

    def specialize(self) -> Gate:
        if len(self.qubits) == 1:
            return I(*self.qubits)
        return self

# end class IDEN


class Ph(Gate):
    r"""
    Apply a global phase shift of exp(i phi).

    Since this gate applies a global phase it technically doesn't need to
    specify qubits at all. But we instead anchor the gate to 1 specific
    qubit so that we can keep track of the phase as when manipulate gates,
    circuits, and DAGCircuits.

    We generally don't actually care about the global phase, since it has no
    physical meaning. It does matter when constructing controlled gates.
    GlobalPhase and the identity differ only by the phase, but a
    controlled-identity is the identity, but a controlled-Global-Phase gate
    will be some instance of the CPHASE (controlled-phase) gate.

    .. math::
        \operatorname{Ph}(\phi) \equiv \begin{pmatrix} e^{i \phi}& 0 \\
                              0 & e^{i \phi} \end{pmatrix}
    """
    # Ref: Explorations in Quantum Computing, Williams, p77

    _diagram_labels = ['Ph({phi})']
    diagonal = True

    def __init__(self, phi: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(phi=phi), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        phi, = self.parameters()
        return - phi * sI(q0)

    @cached_property
    def tensor(self) -> bk.BKTensor:
        phi, = self.parameters()
        return bk.astensorproduct(np.eye(2) * np.exp(1j*phi))

    @property
    def H(self) -> 'Ph':
        return self ** -1

    def __pow__(self, e: Variable) -> 'Ph':
        phi, = self.parameters()
        return Ph(e * phi, *self.qubits)

    # FIXME, shortcut doesn't work as written
    # def run(self, ket: State) -> State:
    #     phi, = self.parameters()
    #     return ket * bk.exp(1j*phi)

    def evolve(self, rho: Density) -> Density:
        """Global phase shifts have no effect on density matrices. Returns argument
        unchanged."""
        return rho
# End class Ph


class X(Gate):
    r"""
    A 1-qubit Pauli-X gate.

    .. math::
        X() &\equiv \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
     """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return - (PI/2) * (1 - sX(q0))

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[0, 1], [1, 0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'X':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'TX':
        return TX(t, *self.qubits)

    def run(self, ket: State) -> State:
        # Fast implementation for X special case
        idx, = ket.qubit_indices(self.qubits)

        if bk.BACKEND == 'numpy':
            take = multi_slice(axes=[idx], items=[[1, 0]])
            tensor = ket.tensor[take]
            return State(tensor, ket.qubits, ket.memory)
        elif bk.BACKEND != 'ctf':                            # pragma: no cover
            tensor = bk.roll(ket.tensor, axis=idx, shift=1)
            return State(tensor, ket.qubits, ket.memory)
        return super().run(ket)                              # pragma: no cover

    # TODO: Optimized evolve method

# end class X


class Y(Gate):
    r"""
    A 1-qubit Pauli-Y gate.

    .. math::
        Y() &\equiv \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

    mnemonic: "Minus eye high".
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return - (PI/2) * (1 - sY(q0))

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[0, -1.0j], [1.0j, 0]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'Y':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'TY':
        return TY(t, *self.qubits)

    def run(self, ket: State) -> State:
        # This is fast Since X and Z have fast optimizations.
        ket = Z(*self.qubits).run(ket)
        ket = X(*self.qubits).run(ket)
        return ket

# end class Y


class Z(Gate):
    r"""
    A 1-qubit Pauli-Z gate.

    .. math::
        Z() &\equiv \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
    """

    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return - (PI/2) * (1 - sZ(q0))

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0], [0, -1.0]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'Z':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'TZ':
        return TZ(t, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(1, *self.qubits).run(ket)

# end class Z


class H(Gate):
    r"""
    A 1-qubit Hadamard gate.

    .. math::
        H() \equiv \frac{1}{\sqrt{2}}
        \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return (PI/2) * ((sX(q0) + sZ(q0)) / sqrt(2) - 1)

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 1], [1, -1]]) / sqrt(2)
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> '_H':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'TH':
        return TH(t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s0 = multi_slice(axes, [0])
            s1 = multi_slice(axes, [1])
            tensor = ket.tensor.copy()
            tensor[s1] -= tensor[s0]
            tensor[s1] *= -0.5
            tensor[s0] -= tensor[s1]
            tensor *= np.sqrt(2)
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)     # pragma: no cover


# Hack. H().H -> H, but the method shadows the class, so can't
# annotate directly.
_H = H

# End class H


class S(Gate):
    r"""
    A 1-qubit phase S gate, equivalent to ``Z ** (1/2)``. The square root
    of the Z gate. Also sometimes denoted as the P gate.

    .. math::
        S() \equiv \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return (PI/2) * (sZ(q0)-1) / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1.0, 0.0], [0.0, 1.0j]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'S_H':
        return S_H(*self.qubits)

    def __pow__(self, t: Variable) -> 'TZ':
        return TZ(t/2, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(1/2, *self.qubits).run(ket)

# end class S


class T(Gate):
    r"""
    A 1-qubit T (pi/8) gate, equivalent to ``X ** (1/4)``. The forth root
    of the Z gate (up to global phase).

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}
    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return (PI/2) * (sZ(q0)-1) / 4

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1.0, 0.0], [0.0, bk.ccast(bk.exp(1j*pi / 4.0))]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'T_H':
        return T_H(*self.qubits)

    def __pow__(self, t: Variable) -> 'TZ':
        return TZ(t/4, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(1/4, *self.qubits).run(ket)

# end class T


class PhaseShift(Gate):
    r"""
    A 1-qubit parametric phase shift gate.
    Equivalent to RZ up to a global phase.

    .. math::
        \text{PhaseShift}(\theta) \equiv \begin{pmatrix}
         1 & 0 \\ 0 & e^{i \theta} \end{pmatrix}
    """
    diagonal = True

    def __init__(self, theta: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        theta, = self.parameters()
        q0, = self.qubits
        return theta * (sZ(q0)-1)/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, = self.parameters()
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0.0], [0.0, bk.exp(1j*ctheta)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'PhaseShift':
        return self ** -1

    def __pow__(self, t: Variable) -> 'PhaseShift':
        theta, = self.parameters()
        return PhaseShift(theta * t, *self.qubits)

    # TODO: CHECKME
    def run(self, ket: State) -> State:
        theta, = self.parameters()
        return TZ(theta/pi, *self.qubits).run(ket)

    def specialize(self) -> Gate:
        qbs = self.qubits
        theta, = self.parameters()
        t = theta/pi
        gate0 = TZ(t, *qbs)
        gate1 = gate0.specialize()
        return gate1

# end class PhaseShift


class RX(Gate):
    r"""A 1-qubit Pauli-X parametric rotation gate.

    .. math::
        R_X(\theta) =   \begin{pmatrix}
                            \cos(\frac{\theta}{2}) & -i \sin(\theta/2) \\
                            -i \sin(\theta/2) & \cos(\theta/2)
                        \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    _diagram_labels = ['Rx({theta})']

    def __init__(self, theta: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        theta, = self.parameters()
        q0, = self.qubits
        return theta * sX(q0) / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, = self.parameters()
        ctheta = bk.ccast(theta)
        unitary = [[bk.cos(ctheta / 2), -1.0j * bk.sin(ctheta / 2)],
                   [-1.0j * bk.sin(ctheta / 2), bk.cos(ctheta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RX':
        return self ** -1

    def __pow__(self, t: Variable) -> 'RX':
        theta, = self.parameters()
        return RX(theta * t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        theta, = self.parameters()
        t = theta/pi
        gate0 = TX(t, *qbs)
        gate1 = gate0.specialize()
        return gate1

# end class RX


class RY(Gate):
    r"""A 1-qubit Pauli-Y parametric rotation gate

    .. math::
        R_Y(\theta) \equiv \begin{pmatrix}
        \cos(\theta / 2) & -\sin(\theta / 2)
        \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    _diagram_labels = ['Ry({theta})']

    def __init__(self, theta: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        theta, = self.parameters()
        q0, = self.qubits
        return theta * sY(q0) / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, = self.parameters()
        ctheta = bk.ccast(theta)
        unitary = [[bk.cos(ctheta / 2.0), -bk.sin(ctheta / 2.0)],
                   [bk.sin(ctheta / 2.0), bk.cos(ctheta / 2.0)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RY':
        return self ** -1

    def __pow__(self, t: Variable) -> 'RY':
        theta, = self.parameters()
        return RY(theta * t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        theta, = self.parameters()
        t = theta/pi
        gate0 = TY(t, *qbs)
        gate1 = gate0.specialize()
        return gate1

# end class RY


class RZ(Gate):
    r"""A 1-qubit Pauli-X parametric rotation gate

    .. math::
        R_Z(\theta)\equiv   \begin{pmatrix}
                                \cos(\theta/2) - i \sin(\theta/2) & 0 \\
                                0 & \cos(\theta/2) + i \sin(\theta/2)
                            \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    diagonal = True
    _diagram_labels = ['Rz({theta})']

    def __init__(self, theta: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        theta, = self.parameters()
        q0, = self.qubits
        return theta * sZ(q0) / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, = self.parameters()
        ctheta = bk.ccast(theta)
        unitary = [[bk.exp(-ctheta * 0.5j), 0],
                   [0, bk.exp(ctheta * 0.5j)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RZ':
        return self ** -1

    def __pow__(self, t: Variable) -> 'RZ':
        theta, = self.parameters()
        return RZ(theta * t, *self.qubits)

    def run(self, ket: State) -> State:
        theta, = self.parameters()
        return TZ(theta/pi, *self.qubits).run(ket)

    def specialize(self) -> Gate:
        qbs = self.qubits
        theta, = self.parameters()
        t = theta/pi
        gate0 = TZ(t, *qbs)
        gate1 = gate0.specialize()
        return gate1

# end class RZ


# Other 1-qubit gates

class S_H(Gate):
    r"""
    The inverse of the 1-qubit phase S gate, equivalent to
    ``Z ** -1/2``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}

    """
    diagonal = True
    _diagram_labels = ['S' + CONJ]

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return -PI*(sZ(q0)-1) / 4

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1.0, 0.0], [0.0, -1.0j]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'S':
        return S(*self.qubits)

    def __pow__(self, t: Variable) -> 'TZ':
        return TZ(-t/2, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(-1/2, *self.qubits).run(ket)

# end class S_H


class T_H(Gate):
    r"""
    The inverse (complex conjugate) of the 1-qubit T (pi/8) gate, equivalent
    to ``Z ** -1/4``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{-i \pi / 4} \end{pmatrix}
    """
    diagonal = True
    _diagram_labels = ['T' + CONJ]

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return - PI * (sZ(q0)-1) / 8

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1.0, 0.0], [0.0, bk.ccast(bk.exp(-1j*pi / 4.0))]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'T':
        return T(*self.qubits)

    def __pow__(self, t: Variable) -> 'TZ':
        return TZ(-t/4, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(-1/4, *self.qubits).run(ket)

# end class T_H


class RN(Gate):
    r"""A 1-qubit rotation of angle theta about axis (nx, ny, nz)

    .. math::
        R_n(\theta) = \cos \frac{\theta}{2} I - i \sin\frac{\theta}{2}
            (n_x X+ n_y Y + n_z Z)

    Args:
        theta: Angle of rotation on Block sphere
        (nx, ny, nz): A three-dimensional real unit vector
    """
    _diagram_labels = ['Rn({theta}, {nx}, {ny}, {nz})']

    def __init__(self,
                 theta: Variable,
                 nx: Variable,
                 ny: Variable,
                 nz: Variable,
                 q0: Qubit = 0) -> None:

        norm = np.sqrt(np.real(nx**2 + ny**2 + nz**2))
        nx /= norm
        ny /= norm
        nz /= norm
        theta *= norm

        params = dict(theta=theta, nx=nx, ny=ny, nz=nz)
        super().__init__(params=params, qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        theta, nx, ny, nz = self.parameters()
        q0, = self.qubits
        return theta*(nx*sX(q0) + ny*sY(q0) + nz*sZ(q0))/2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, nx, ny, nz = self.parameters()

        ctheta = bk.ccast(theta)
        cost = bk.cos(ctheta / 2)
        sint = bk.sin(ctheta / 2)
        unitary = [[cost - 1j * sint * nz, -1j * sint * nx - sint * ny],
                   [-1j * sint * nx + sint * ny, cost + 1j * sint * nz]]

        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RN':
        return self ** -1

    def __pow__(self, t: Variable) -> 'RN':
        theta, nx, ny, nz = self.parameters()
        return RN(t * theta, nx, ny, nz, *self.qubits)

    # TODO:     def specialize(self) -> Gate:


# end class RN


class TX(Gate):
    r"""Powers of the 1-qubit Pauli-X gate.

    .. math::
        TX(t) = X^t = e^{i \pi t/2} R_X(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """

    def __init__(self, t: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(t=t), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        q0, = self.qubits
        return t * (sX(q0) - 1) * PI / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.cos(ctheta / 2),
                    phase * -1.0j * bk.sin(ctheta / 2)],
                   [phase * -1.0j * bk.sin(ctheta / 2),
                    phase * bk.cos(ctheta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TX':
        return self ** -1

    def __pow__(self, e: Variable) -> 'TX':
        t, = self.parameters()
        return TX(e * t, *self.qubits)

    def specialize(self) -> Gate:
        opts = {0.0: I, 0.5: V, 1.0: X, 1.5: V_H, 2.0: I}
        return _specialize_gate(self, [2], opts)
# end class TX


class TY(Gate):
    r"""Powers of the 1-qubit Pauli-Y gate.

    The pseudo-Hadamard gate is TY(3/2), and its inverse is TY(1/2).

    .. math::
        TY(t) = Y^t = e^{i \pi t/2} R_Y(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere

    """
    _diagram_labels = ['Y^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(t=t), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        q0, = self.qubits
        return t * (sY(q0) - 1) * pi / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.cos(ctheta / 2.0),
                    phase * -bk.sin(ctheta / 2.0)],
                   [phase * bk.sin(ctheta / 2.0),
                    phase * bk.cos(ctheta / 2.0)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TY':
        return self ** -1

    def __pow__(self, e: Variable) -> 'TY':
        t, = self.parameters()
        return TY(e*t, *self.qubits)

    def specialize(self) -> Gate:
        opts = {0.0: I, 1.0: Y, 2.0: I}
        return _specialize_gate(self, [2], opts)

# end class TY


class TZ(Gate):
    r"""Powers of the 1-qubit Pauli-Z gate.

    .. math::
        TZ(t) = Z^t = e^{i \pi t/2} R_Z(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """
    diagonal = True
    _diagram_labels = ['Z^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(t=t), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        q0, = self.qubits
        return t * (sZ(q0) - 1) * pi / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.exp(-ctheta * 0.5j), 0],
                   [0, phase * bk.exp(ctheta * 0.5j)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TZ':
        t = - self.params['t']
        return TZ(t, *self.qubits)

    def __pow__(self, e: Variable) -> 'TZ':
        t, = self.parameters()
        return TZ(e*t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            t, = self.parameters()
            axes = ket.qubit_indices(self.qubits)
            s1 = multi_slice(axes, [1])
            tensor = ket.tensor.copy()
            tensor[s1] *= bk.exp(+1.0j*pi*t)
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover

    def specialize(self) -> Gate:
        opts = {0.0: I, 0.25: T, 0.5: S, 1.0: Z, 1.5: S_H, 1.75: T_H, 2.0: I}
        return _specialize_gate(self, [2], opts)


# end class TZ


class TH(Gate):
    r"""
    Powers of the 1-qubit Hadamard gate.

    .. math::
        TH(t) = H^t = e^{i \pi t/2}
        \begin{pmatrix}
            \cos(\tfrac{t}{2}) + \tfrac{i}{\sqrt{2}}\sin(\tfrac{t}{2})) &
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) \\
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) &
            \cos(\tfrac{t}{2}) -\tfrac{i}{\sqrt{2}} \sin(\frac{t}{2})
        \end{pmatrix}
    """
    _diagram_labels = ['H^{t}']

    def __init__(self, t: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(t=t), qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        t, = self.parameters()
        return t * ((sX(q0) + sZ(q0)) / sqrt(2) - 1) * pi / 2

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t, = self.parameters()
        theta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * theta)
        unitary = [[phase * bk.cos(theta / 2)
                    - (phase * 1.0j * bk.sin(theta / 2)) / sqrt(2),
                    -(phase * 1.0j * bk.sin(theta / 2)) / sqrt(2)],
                   [-(phase * 1.0j * bk.sin(theta / 2)) / sqrt(2),
                    phase * bk.cos(theta / 2)
                    + (phase * 1.0j * bk.sin(theta / 2)) / sqrt(2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TH':
        return self ** -1

    def __pow__(self, e: Variable) -> 'TH':
        t, = self.parameters()
        return TH(e*t, *self.qubits)

    def specialize(self) -> Gate:
        opts = {0.0: I, 1.0: H, 2.0: I}
        return _specialize_gate(self, [2], opts)

# end class TH


class V(Gate):
    r"""
    Principal square root of the X gate, X-PLUS-90 gate.
    """

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return (sX(q0) - 1) * pi / 4

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return TX(0.5).tensor

    @property
    def H(self) -> 'V_H':
        return V_H(*self.qubits)

    def __pow__(self, t: Variable) -> 'TX':
        return TX(0.5*t, *self.qubits)

# end class V


class V_H(Gate):
    r"""
    Complex conjugate of the V gate, X-MINUS-90 gate.
    """

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return -(sX(q0) - 1) * pi / 4

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return TX(-0.5).tensor

    @property
    def H(self) -> 'V':
        return V(*self.qubits)

    def __pow__(self, t: Variable) -> 'TX':
        return TX(-0.5*t, *self.qubits)

# end class V_H


class SqrtY(Gate):
    r"""
    Principal square root of the Y gate.
    """
    _diagram_labels = [SQRT+'Y']

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return (sY(q0) - 1) * pi / 4

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return TY(0.5).tensor

    @property
    def H(self) -> 'SqrtY_H':
        return SqrtY_H(*self.qubits)

    def __pow__(self, t: Variable) -> 'TY':
        return TY(0.5*t, *self.qubits)

    # TODO: Experiment. Maybe makes more sense to have main decompositions in
    # gate classes, rather than separate in translate?
    def decompose(self) -> Iterator[TY]:
        """Translate  gate to TY"""
        q0, = self.qubits
        yield TY(0.5, q0)

# end class SqrtY


class SqrtY_H(Gate):
    r"""
    Complex conjugate of the SqrtY gate.
    """
    _diagram_labels = [SQRT+'Y'+CONJ]

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        q0, = self.qubits
        return -(sY(q0) - 1) * pi / 4

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return TY(-0.5).tensor

    @property
    def H(self) -> 'SqrtY':
        return SqrtY(*self.qubits)

    def __pow__(self, t: Variable) -> 'TY':
        return TY(-0.5*t, *self.qubits)

    def decompose(self) -> Iterator[TY]:
        """Translate  gate to TY"""
        q0, = self.qubits
        yield TY(-0.5, q0)
# end class SqrtY_H
