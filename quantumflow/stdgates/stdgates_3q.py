# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Three qubit gates
"""

import numpy as np

from .. import tensors, utils, var
from ..config import CTRL, SWAP_TARGET, TARGET
from ..ops import StdGate
from ..paulialgebra import Pauli, sX, sZ
from ..qubits import Qubit
from ..states import State
from ..tensors import QubitTensor
from ..var import PI, Variable
from .stdgates_2q import CZ, CNot, Swap

# 3-qubit gates, in alphabetic order

__all__ = ("CCiX", "CCNot", "CCXPow", "CCZ", "CSwap", "Deutsch")


class CCiX(StdGate):
    r"""
    A doubly controlled iX gate.

    Equivalent to ``controlled_gate(cnot())``

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
          http://arxiv.org/abs/1210.0974
    """
    # Kudos: Adapted from 1210.0974, via Quipper

    cv_tensor_structure = "monomial"
    _diagram_labels = ("─" + CTRL + "─", "─" + CTRL + "─", "iX─")

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1, q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return -sX(q2) * (1 - sZ(q1)) * (1 - sZ(q0)) * PI / 8

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0j],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0j, 0.0],
            ]
        )
        return tensors.asqutensor(unitary)


# end class CCiX


class CCNot(StdGate):
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
    cv_tensor_structure = "permutation"
    _diagram_labels = (CTRL, CTRL, TARGET)

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1, q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return CNot(q1, q2).hamiltonian * (1 - sZ(q0)) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CCNot":
        return self  # Hermitian

    def __pow__(self, t: Variable) -> "CCXPow":
        return CCXPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        axes = ket.qubit_indices(self.qubits)
        s110 = utils.multi_slice(axes, [1, 1, 0])
        s111 = utils.multi_slice(axes, [1, 1, 1])
        tensor = ket.tensor.copy()
        tensor[s110] = ket.tensor[s111]
        tensor[s111] = ket.tensor[s110]
        return State(tensor, ket.qubits, ket.memory)


# end class CCNot


class CCXPow(StdGate):
    r"""
    Powers of the Toffoli gate.

    args:
        t:  turns (powers of the CNOT gate, or controlled-powers of X)
        q0: control qubit
        q1: control qubit
        q2: target qubits
    """
    _diagram_labels = (CTRL, CTRL, "X^{t}")

    def __init__(
        self, t: Variable, q0: Qubit = 0, q1: Qubit = 1, q2: Qubit = 2
    ) -> None:
        super().__init__(params=[t], qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        return CCNot(*self.qubits).hamiltonian * self.param("t")

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = np.pi * self.float_param("t")
        phase = np.exp(0.5j * theta)
        cht = np.cos(theta / 2)
        sht = np.sin(theta / 2)
        unitary = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, phase * cht, phase * -1.0j * sht],
            [0, 0, 0, 0, 0, 0, phase * -1.0j * sht, phase * cht],
        ]

        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CCXPow":
        return self ** -1

    def __pow__(self, e: Variable) -> "CCXPow":
        (t,) = self.params
        return CCXPow(e * t, *self.qubits)


# end class CCXPow


class CSwap(StdGate):
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
    cv_hermitian = True
    cv_tensor_structure = "permutation"
    _diagram_labels = (CTRL, SWAP_TARGET, SWAP_TARGET)

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1, q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return Swap(q1, q2).hamiltonian * (1 - sZ(q0)) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CSwap":
        return self  # Hermitian

    # TODO: __pow__

    def run(self, ket: State) -> State:
        axes = ket.qubit_indices(self.qubits)
        s101 = utils.multi_slice(axes, [1, 0, 1])
        s110 = utils.multi_slice(axes, [1, 1, 0])
        tensor = ket.tensor.copy()
        tensor[s101] = ket.tensor[s110]
        tensor[s110] = ket.tensor[s101]
        return State(tensor, ket.qubits, ket.memory)


# end class CSWAP


class CCZ(StdGate):
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
    cv_interchangeable = True
    cv_tensor_structure = "diagonal"
    _diagram_labels = (CTRL, CTRL, CTRL)

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1, q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return CZ(q1, q2).hamiltonian * (1 - sZ(q0)) / 2

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        unitary = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "CCZ":
        return self  # Hermitian

    # TODO: __pow__

    def run(self, ket: State) -> State:
        axes = ket.qubit_indices(self.qubits)
        s11 = utils.multi_slice(axes, [1, 1, 1])
        tensor = ket.tensor.copy()
        tensor[s11] *= -1
        return State(tensor, ket.qubits, ket.memory)


# end class CCZ


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
    _diagram_labels = (CTRL, CTRL, "iRx({theta})^2")

    def __init__(
        self, theta: Variable, q0: Qubit = 0, q1: Qubit = 1, q2: Qubit = 2
    ) -> None:
        super().__init__(params=[theta], qubits=[q0, q1, q2])

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))

        unitary = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1.0j * np.cos(theta), np.sin(theta)],
            [0, 0, 0, 0, 0, 0, np.sin(theta), 1.0j * np.cos(theta)],
        ]
        return tensors.asqutensor(unitary)

    # TODO: Specializes to Toffoli


# end class Deutsch
