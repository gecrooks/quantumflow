# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Standard one qubit gates
"""

from typing import Dict, List, Type

import numpy as np

from .. import tensors, utils, var
from ..config import CONJ, SQRT
from ..ops import StdGate
from ..paulialgebra import Pauli, sI, sX, sY, sZ
from ..qubits import Qubit
from ..states import Density, State
from ..tensors import QubitTensor
from ..var import PI, Variable

__all__ = (
    "I",
    "Ph",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "T",
    "PhaseShift",
    "Rx",
    "Ry",
    "Rz",
    "Rn",
    "XPow",
    "YPow",
    "ZPow",
    "HPow",
    "S_H",
    "T_H",
    "V",
    "V_H",
    "SqrtY",
    "SqrtY_H",
)


def _specialize_gate(
    gate: StdGate, periods: List[float], opts: Dict[float, Type[StdGate]]
) -> StdGate:
    """Return a specialized instance of a given general gate. Used
    by the specialize code of various gates.

    Args:
        gate:       The gate instance to specialize
        periods:    The periods of the gate's parameters. Gate parameters
                    are wrapped to range[0,period]
        opts:       A map from particular gate parameters to a special case
                    of the original gate type
    """
    # params = list(gate.params)

    params = list(gate.params)

    # for p in params:
    #     if variable_is_symbolic(p):
    #         return gate

    params = [p % pd for p, pd in zip(params, periods)]

    for values, gatetype in opts.items():
        if np.isclose(params, values):
            return gatetype(*gate.qubits)  # type: ignore

    return type(gate)(*params, *gate.qubits)  # type: ignore


# Standard 1 qubit gates


class I(StdGate):  # noqa: E742
    r"""
    The 1-qubit identity gate.

    .. math::
        I() \equiv \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
    """
    cv_hermitian = True
    cv_tensor_structure = "identity"

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        return Pauli.zero()

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return tensors.asqutensor(np.eye(2))

    @property
    def H(self) -> "I":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "I":
        return self

    def run(self, ket: State) -> State:
        return ket

    def evolve(self, rho: Density) -> Density:
        return rho


class Ph(StdGate):
    r"""
    Apply a global phase shift of exp(i phi).

    Since this gate applies a global phase it technically doesn't need to
    specify qubits at all. But we instead anchor the gate to 1 specific
    qubit so that we can keep track of the phase as when manipulate gates,
    circuits, and DAGCircuits.

    We generally don't actually care about the global phase, since it has no
    physical meaning, although it does matter when constructing controlled gates.

    .. math::
        \operatorname{Ph}(\phi) \equiv \begin{pmatrix} e^{i \phi}& 0 \\
                              0 & e^{i \phi} \end{pmatrix}
    """
    # TODO
    # Ref: Explorations in Quantum Computing, Williams, p77
    # Ref: Barenco

    cv_tensor_structure = "diagonal"
    _diagram_labels = ["Ph({phi})"]

    def __init__(self, phi: float, q0: Qubit) -> None:
        super().__init__(params=[phi], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        (phi,) = self.params
        return -phi * sI(q0)

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        phi = var.asfloat(self.param("phi"))
        unitary = [[np.exp(1j * phi), 0.0], [0.0, np.exp(1j * phi)]]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Ph":
        return self ** -1

    def __pow__(self, t: Variable) -> "Ph":
        return Ph(t * self.param("phi"), *self.qubits)

    def run(self, ket: State) -> State:
        (phi,) = self.params
        tensor = ket.tensor * np.exp(1j * phi)
        return State(tensor, ket.qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        """Global phase shifts have no effect on density matrices. Returns argument
        unchanged."""
        return rho


# End class Ph


class X(StdGate):
    r"""
    A 1-qubit Pauli-X gate.

    .. math::
        X() &\equiv \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
    """
    cv_hermitian = True
    cv_tensor_structure = "permutation"

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return -(PI / 2) * (1 - sX(q0))

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = [[0, 1], [1, 0]]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "X":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "XPow":
        return XPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        (idx,) = ket.qubit_indices(self.qubits)
        take = utils.multi_slice(axes=[idx], items=[[1, 0]])
        tensor = ket.tensor[take]
        return State(tensor, ket.qubits, ket.memory)


# end class X


class Y(StdGate):
    r"""
    A 1-qubit Pauli-Y gate.

    .. math::
        Y() &\equiv \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

    mnemonic: "Minus eye high".
    """
    cv_hermitian = True
    cv_tensor_structure = "monomial"

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return -(PI / 2) * (1 - sY(q0))

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = np.asarray([[0, -1.0j], [1.0j, 0]])
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Y":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "YPow":
        return YPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        # This is fast Since X and Z have fast optimizations.
        ket = Z(*self.qubits).run(ket)
        ket = X(*self.qubits).run(ket)
        return ket


# end class Y


class Z(StdGate):
    r"""
    A 1-qubit Pauli-Z gate.

    .. math::
        Z() &\equiv \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
    """
    cv_hermitian = True
    cv_tensor_structure = "diagonal"

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return -(PI / 2) * (1 - sZ(q0))

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = np.asarray([[1, 0], [0, -1.0]])
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Z":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "ZPow":
        return ZPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        return ZPow(1, *self.qubits).run(ket)


# end class Z


class H(StdGate):
    r"""
    A 1-qubit Hadamard gate.

    .. math::
        H() \equiv \frac{1}{\sqrt{2}}
        \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
    """
    cv_hermitian = True

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return (PI / 2) * ((sX(q0) + sZ(q0)) / np.sqrt(2) - 1)

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = np.asarray([[1, 1], [1, -1]]) / np.sqrt(2)
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "_H":  # See NB implementation note below
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "HPow":
        return HPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        axes = ket.qubit_indices(self.qubits)
        s0 = utils.multi_slice(axes, [0])
        s1 = utils.multi_slice(axes, [1])
        tensor = ket.tensor.copy()
        tensor[s1] -= tensor[s0]
        tensor[s1] *= -0.5
        tensor[s0] -= tensor[s1]
        tensor *= np.sqrt(2)
        return State(tensor, ket.qubits, ket.memory)


# Note: H().H -> H, but the method shadows the class, so we can't
# annotate directly.
_H = H

# End class H


class S(StdGate):
    r"""
    A 1-qubit phase S gate, equivalent to ``Z ** (1/2)``. The square root
    of the Z gate. Also sometimes denoted as the P gate.

    .. math::
        S() \equiv \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

    """
    cv_tensor_structure = "diagonal"

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return (PI / 2) * (sZ(q0) - 1) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = np.asarray([[1.0, 0.0], [0.0, 1.0j]])
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "S_H":
        return S_H(*self.qubits)

    def __pow__(self, t: Variable) -> "ZPow":
        return ZPow(t / 2, *self.qubits)

    def run(self, ket: State) -> State:
        return ZPow(1 / 2, *self.qubits).run(ket)


# end class S


class T(StdGate):
    r"""
    A 1-qubit T (pi/8) gate, equivalent to ``X ** (1/4)``. The forth root
    of the Z gate (up to global phase).

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}
    """
    cv_tensor_structure = "diagonal"

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return (PI / 2) * (sZ(q0) - 1) / 4

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = [[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4.0)]]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "T_H":
        return T_H(*self.qubits)

    def __pow__(self, t: Variable) -> "ZPow":
        return ZPow(t / 4, *self.qubits)

    def run(self, ket: State) -> State:
        return ZPow(1 / 4, *self.qubits).run(ket)


# end class T


class PhaseShift(StdGate):
    r"""
    A 1-qubit parametric phase shift gate.
    Equivalent to Rz up to a global phase.

    .. math::
        \text{PhaseShift}(\theta) \equiv \begin{pmatrix}
         1 & 0 \\ 0 & e^{i \theta} \end{pmatrix}
    """
    cv_tensor_structure = "diagonal"

    def __init__(self, theta: Variable, q0: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (theta,) = self.params
        (q0,) = self.qubits
        return theta * (sZ(q0) - 1) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [[1.0, 0.0], [0.0, np.exp(1j * theta)]]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "PhaseShift":
        return self ** -1

    def __pow__(self, t: Variable) -> "PhaseShift":
        return PhaseShift(self.param("theta") * t, *self.qubits)

    def run(self, ket: State) -> State:
        (theta,) = self.params
        return ZPow(theta / np.pi, *self.qubits).run(ket)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (theta,) = self.params
        t = theta / np.pi
        gate0 = ZPow(t, *qbs)
        gate1 = gate0.specialize()
        return gate1


# end class PhaseShift


class Rx(StdGate):
    r"""A 1-qubit Pauli-X parametric rotation gate.

    .. math::
        R_x(\theta) =   \begin{bmatrix*}[r]
                            \cos(\half\theta) & -i \sin(\half\theta) \\
                            -i \sin(\half\theta) & \cos(\half\theta)
                        \end{bmatrix*}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    _diagram_labels = ["Rx({theta})"]

    def __init__(self, theta: Variable, q0: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (theta,) = self.params
        (q0,) = self.qubits
        return theta * sX(q0) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [
            [np.cos(theta / 2), -1.0j * np.sin(theta / 2)],
            [-1.0j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Rx":
        return self ** -1

    def __pow__(self, t: Variable) -> "Rx":
        return Rx(self.param("theta") * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (theta,) = self.params
        t = theta / np.pi
        gate0 = XPow(t, *qbs)
        gate1 = gate0.specialize()
        return gate1


# end class Rx


class Ry(StdGate):
    r"""A 1-qubit Pauli-Y parametric rotation gate

    .. math::
        R_y(\theta) =   \begin{bmatrix*}[r]
            \cos(\half\theta) & -\sin(\half\theta)
            \\ \sin(\half\theta) & \cos(\half\theta)
                        \end{bmatrix*}

    Args:
        theta: Angle of rotation in Bloch sphere
    """

    _diagram_labels = ["Ry({theta})"]

    def __init__(self, theta: Variable, q0: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (theta,) = self.params
        (q0,) = self.qubits
        return theta * sY(q0) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Ry":
        return self ** -1

    def __pow__(self, t: Variable) -> "Ry":
        return Ry(self.param("theta") * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (theta,) = self.params
        t = theta / np.pi
        gate0 = YPow(t, *qbs)
        gate1 = gate0.specialize()
        return gate1


# end class Ry


class Rz(StdGate):
    r"""A 1-qubit Pauli-X parametric rotation gate

    .. math::
        R_z(\theta) =   \begin{bmatrix*}
        e^{-i\half\theta} & 0 \\
        0 & e^{+i\half\theta}
                        \end{bmatrix*}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    cv_tensor_structure = "diagonal"
    _diagram_labels = ["Rz({theta})"]

    def __init__(self, theta: Variable, q0: Qubit) -> None:
        super().__init__(params=[theta], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return self.param("theta") * sZ(q0) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        unitary = [[np.exp(-theta * 0.5j), 0], [0, np.exp(theta * 0.5j)]]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Rz":
        return self ** -1

    def __pow__(self, t: Variable) -> "Rz":
        return Rz(self.param("theta") * t, *self.qubits)

    def run(self, ket: State) -> State:
        (theta,) = self.params
        return ZPow(theta / np.pi, *self.qubits).run(ket)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        (theta,) = self.params
        t = theta / np.pi
        gate0 = ZPow(t, *qbs)
        gate1 = gate0.specialize()
        return gate1


# end class Rz


# Other 1-qubit gates


class S_H(StdGate):
    r"""
    The inverse of the 1-qubit phase S gate, equivalent to
    ``Z ** -1/2``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}

    """
    cv_tensor_structure = "diagonal"
    _diagram_labels = ["S" + CONJ]

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return -PI * (sZ(q0) - 1) / 4

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = np.asarray([[1.0, 0.0], [0.0, -1.0j]])
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "S":
        return S(*self.qubits)

    def __pow__(self, t: Variable) -> "ZPow":
        return ZPow(-t / 2, *self.qubits)

    def run(self, ket: State) -> State:
        return ZPow(-1 / 2, *self.qubits).run(ket)


# end class S_H


class T_H(StdGate):
    r"""
    The inverse (complex conjugate) of the 1-qubit T (pi/8) gate, equivalent
    to ``Z ** -1/4``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{-i \pi / 4} \end{pmatrix}
    """
    cv_tensor_structure = "diagonal"
    _diagram_labels = ["T" + CONJ]

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return -PI * (sZ(q0) - 1) / 8

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = [[1.0, 0.0], [0.0, np.exp(-1j * np.pi / 4.0)]]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "T":
        return T(*self.qubits)

    def __pow__(self, t: Variable) -> "ZPow":
        return ZPow(-t / 4, *self.qubits)

    def run(self, ket: State) -> State:
        return ZPow(-1 / 4, *self.qubits).run(ket)


# end class T_H


class Rn(StdGate):
    r"""A 1-qubit rotation of angle theta about axis (nx, ny, nz)

    .. math::
        R_n(\theta) = \cos \frac{\theta}{2} I - i \sin\frac{\theta}{2}
            (n_x X+ n_y Y + n_z Z)

    Args:
        theta: Angle of rotation on Block sphere
        (nx, ny, nz): A three-dimensional real unit vector
    """
    _diagram_labels = ["Rn({theta}, {nx}, {ny}, {nz})"]

    def __init__(
        self, theta: Variable, nx: Variable, ny: Variable, nz: Variable, q0: Qubit
    ) -> None:

        norm = var.sqrt(nx ** 2 + ny ** 2 + nz ** 2)

        nx /= norm
        ny /= norm
        nz /= norm
        theta *= norm

        super().__init__(params=[theta, nx, ny, nz], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        theta, nx, ny, nz = self.params
        (q0,) = self.qubits
        return theta * (nx * sX(q0) + ny * sY(q0) + nz * sZ(q0)) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        nx = var.asfloat(self.param("nx"))
        ny = var.asfloat(self.param("ny"))
        nz = var.asfloat(self.param("nz"))

        cost = np.cos(theta / 2)
        sint = np.sin(theta / 2)
        unitary = [
            [cost - 1j * sint * nz, -1j * sint * nx - sint * ny],
            [-1j * sint * nx + sint * ny, cost + 1j * sint * nz],
        ]

        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "Rn":
        return self ** -1

    def __pow__(self, t: Variable) -> "Rn":
        theta, nx, ny, nz = self.params
        return Rn(t * theta, nx, ny, nz, *self.qubits)

    # TODO:     def specialize(self) -> Gate:


# end class RN


class XPow(StdGate):
    r"""Powers of the 1-qubit Pauli-X gate.

    .. math::
        XPow(t) = X^t = e^{i \pi t/2} R_X(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """

    _diagram_labels = ["X^{t}"]

    def __init__(self, t: Variable, q0: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        (q0,) = self.qubits
        return t * (sX(q0) - 1) * PI / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * var.asfloat(self.param("t"))
        phase = np.exp(0.5j * theta)
        unitary = [
            [phase * np.cos(theta / 2), phase * -1.0j * np.sin(theta / 2)],
            [phase * -1.0j * np.sin(theta / 2), phase * np.cos(theta / 2)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "XPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "XPow":
        return XPow(t * self.param("t"), *self.qubits)

    def specialize(self) -> StdGate:
        opts = {0.0: I, 0.5: V, 1.0: X, 1.5: V_H, 2.0: I}
        return _specialize_gate(self, [2], opts)


# end class XPow


class YPow(StdGate):
    r"""Powers of the 1-qubit Pauli-Y gate.

    The pseudo-Hadamard gate is YPow(3/2), and its inverse is YPow(1/2).

    .. math::
        YPow(t) = Y^t = e^{i \pi t/2} R_Y(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere

    """
    _diagram_labels = ["Y^{t}"]

    def __init__(self, t: Variable, q0: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        (q0,) = self.qubits
        return t * (sY(q0) - 1) * PI / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * var.asfloat(self.param("t"))
        phase = np.exp(0.5j * theta)
        unitary = [
            [phase * np.cos(theta / 2.0), phase * -np.sin(theta / 2.0)],
            [phase * np.sin(theta / 2.0), phase * np.cos(theta / 2.0)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "YPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "YPow":
        return YPow(t * self.param("t"), *self.qubits)

    def specialize(self) -> StdGate:
        opts = {0.0: I, 1.0: Y, 2.0: I}
        return _specialize_gate(self, [2], opts)


# end class YPow


class ZPow(StdGate):
    r"""Powers of the 1-qubit Pauli-Z gate.

    .. math::
        ZPow(t) = Z^t = e^{i \pi t/2} R_Z(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """
    cv_tensor_structure = "diagonal"
    _diagram_labels = ["Z^{t}"]

    def __init__(self, t: Variable, q0: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (t,) = self.params
        (q0,) = self.qubits
        return t * (sZ(q0) - 1) * PI / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * var.asfloat(self.param("t"))
        phase = np.exp(0.5j * theta)
        unitary = [
            [phase * np.exp(-theta * 0.5j), 0],
            [0, phase * np.exp(theta * 0.5j)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "ZPow":
        return ZPow(-self.param("t"), *self.qubits)

    def __pow__(self, t: Variable) -> "ZPow":
        return ZPow(t * self.param("t"), *self.qubits)

    def run(self, ket: State) -> State:
        (t,) = self.params
        axes = ket.qubit_indices(self.qubits)
        s1 = utils.multi_slice(axes, [1])
        tensor = ket.tensor.copy()
        tensor[s1] *= np.exp(+1.0j * np.pi * t)
        return State(tensor, ket.qubits, ket.memory)

    def specialize(self) -> StdGate:
        opts = {0.0: I, 0.25: T, 0.5: S, 1.0: Z, 1.5: S_H, 1.75: T_H, 2.0: I}
        return _specialize_gate(self, [2], opts)


# end class ZPow


class HPow(StdGate):
    r"""
    Powers of the 1-qubit Hadamard gate.

    .. math::
        HPow(t) = H^t = e^{i \pi t/2}
        \begin{pmatrix}
            \cos(\tfrac{t}{2}) + \tfrac{i}{\sqrt{2}}\sin(\tfrac{t}{2})) &
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) \\
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) &
            \cos(\tfrac{t}{2}) -\tfrac{i}{\sqrt{2}} \sin(\frac{t}{2})
        \end{pmatrix}
    """
    _diagram_labels = ("H^{t}",)

    def __init__(self, t: Variable, q0: Qubit) -> None:
        super().__init__(params=[t], qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        return H(*self.qubits).hamiltonian * self.param("t")

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * var.asfloat(self.param("t"))
        phase = np.exp(0.5j * theta)
        unitary = [
            [
                phase * np.cos(theta / 2)
                - (phase * 1.0j * np.sin(theta / 2)) / np.sqrt(2),
                -(phase * 1.0j * np.sin(theta / 2)) / np.sqrt(2),
            ],
            [
                -(phase * 1.0j * np.sin(theta / 2)) / np.sqrt(2),
                phase * np.cos(theta / 2)
                + (phase * 1.0j * np.sin(theta / 2)) / np.sqrt(2),
            ],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "HPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "HPow":
        return HPow(t * self.param("t"), *self.qubits)

    def specialize(self) -> StdGate:
        opts = {0.0: I, 1.0: H, 2.0: I}
        return _specialize_gate(self, [2], opts)


# end class HPow


class V(StdGate):
    r"""
    Principal square root of the X gate, X-PLUS-90 gate.
    """

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return (sX(q0) - 1) * PI / 4

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return XPow(0.5, *self.qubits).tensor

    @property
    def H(self) -> "V_H":
        return V_H(*self.qubits)

    def __pow__(self, t: Variable) -> "XPow":
        return XPow(0.5 * t, *self.qubits)


# end class V


class V_H(StdGate):
    r"""
    Complex conjugate of the V gate, X-MINUS-90 gate.
    """

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return -(sX(q0) - 1) * PI / 4

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return XPow(-0.5, *self.qubits).tensor

    @property
    def H(self) -> "V":
        return V(*self.qubits)

    def __pow__(self, t: Variable) -> "XPow":
        return XPow(-0.5 * t, *self.qubits)


# end class V_H


class SqrtY(StdGate):
    r"""
    Principal square root of the Y gate.
    """
    _diagram_labels = [SQRT + "Y"]

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return (sY(q0) - 1) * PI / 4

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return YPow(0.5, *self.qubits).tensor

    @property
    def H(self) -> "SqrtY_H":
        return SqrtY_H(*self.qubits)

    def __pow__(self, t: Variable) -> "YPow":
        return YPow(0.5 * t, *self.qubits)


# end class SqrtY


class SqrtY_H(StdGate):
    r"""
    Complex conjugate of the np.sqrtY gate.
    """
    _diagram_labels = [SQRT + "Y" + CONJ]

    def __init__(self, q0: Qubit) -> None:
        super().__init__(qubits=[q0])

    @property
    def hamiltonian(self) -> Pauli:
        (q0,) = self.qubits
        return -(sY(q0) - 1) * PI / 4

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        return YPow(-0.5, *self.qubits).tensor

    @property
    def H(self) -> "SqrtY":
        return SqrtY(*self.qubits)

    def __pow__(self, t: Variable) -> "YPow":
        return YPow(-0.5 * t, *self.qubits)


# end class SqrtY_H


# fin
