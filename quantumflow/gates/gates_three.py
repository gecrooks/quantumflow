
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Three qubit gates
"""
import numpy as np
# from .. import backend as bk
from ..qubits import Qubit
from ..states import State
from ..ops import Gate, StdGate
from ..utils import multi_slice, cached_property
from ..variables import Variable
from ..config import CTRL, TARGET, SWAP_TARGET
from ..paulialgebra import Pauli, sZ, sX
from .gates_two import SWAP, CNOT, CZ


from ..backends import get_backend, BKTensor
bk = get_backend()
pi = bk.pi
PI = bk.PI

# 3-qubit gates, in alphabetic order

__all__ = ('CCiX', 'CCNOT', 'CCXPow', 'CCZ', 'CSWAP', 'Deutsch')


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

    monomial = True
    _diagram_labels = ('─'+CTRL+'─', '─'+CTRL+'─', 'iX─')

    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return -sX(q2) * (1 - sZ(q1)) * (1-sZ(q0)) * bk.PI/8

    @cached_property
    def tensor(self) -> BKTensor:
        unitary = np.asarray(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0j],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0j, 0.0]])
        return bk.astensorproduct(unitary)

# end class CCiX


class CCNOT(StdGate):
    r"""
    A 3-qubit Toffoli gate. A controlled, controlled-not.

    Equivalent to ``controlled_gate(cnot())``

    .. math::
        \text{CCNOT}() \equiv \begin{pmatrix}
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
    permutation = True
    _diagram_labels = (CTRL, CTRL, TARGET)

    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return CNOT(q1, q2).hamiltonian * (1-sZ(q0))/2

    @cached_property
    def tensor(self) -> BKTensor:
        unitary = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1, 0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CCNOT':
        return self  # Hermitian

    def __pow__(self, t: Variable) -> 'CCXPow':
        return CCXPow(t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s110 = multi_slice(axes, [1, 1, 0])
            s111 = multi_slice(axes, [1, 1, 1])
            tensor = ket.tensor.copy()
            tensor[s110] = ket.tensor[s111]
            tensor[s111] = ket.tensor[s110]
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover

# end class CCNOT


class CCXPow(StdGate):
    r"""
    Powers of the Toffoli gate.

    args:
        t:  turns (powers of the CNOT gate, or controlled-powers of X)
        q0: control qubit
        q1: control qubit
        q2: target qubits
    """
    _diagram_labels = (CTRL, CTRL, 'X^{t}')

    def __init__(self,
                 t: Variable,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        t, = self.parameters()
        return CCNOT(*self.qubits).hamiltonian * t

    @cached_property
    def tensor(self) -> BKTensor:
        t, = self.parameters()
        ctheta = bk.ccast(bk.pi * t)
        phase = bk.exp(0.5j * ctheta)
        cht = bk.cos(ctheta / 2)
        sht = bk.sin(ctheta / 2)
        unitary = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, phase * cht, phase * -1.0j * sht],
                   [0, 0, 0, 0, 0, 0, phase * -1.0j * sht, phase * cht]]

        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CCXPow':
        return self ** -1

    def __pow__(self, e: Variable) -> 'CCXPow':
        t, = self.parameters()
        return CCXPow(e*t, *self.qubits)

# end class CCXPow


class CSWAP(StdGate):
    r"""
    A 3-qubit Fredkin gate. A controlled swap.

    Equivalent to ``controlled_gate(swap())``

    .. math::
        \text{CSWAP}() \equiv \begin{pmatrix}
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
    permutation = True
    _diagram_labels = (CTRL, SWAP_TARGET, SWAP_TARGET)

    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return SWAP(q1, q2).hamiltonian * (1-sZ(q0))/2

    @cached_property
    def tensor(self) -> BKTensor:
        unitary = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CSWAP':
        return self  # Hermitian

    # TODO: __pow__

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s101 = multi_slice(axes, [1, 0, 1])
            s110 = multi_slice(axes, [1, 1, 0])
            tensor = ket.tensor.copy()
            tensor[s101] = ket.tensor[s110]
            tensor[s110] = ket.tensor[s101]
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover

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
    interchangeable = True
    diagonal = True
    _diagram_labels = (CTRL, CTRL, CTRL)

    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1, q2 = self.qubits
        return CZ(q1, q2).hamiltonian * (1-sZ(q0))/2

    @cached_property
    def tensor(self) -> BKTensor:
        unitary = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, -1]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> Gate:
        return self  # Hermitian

    # TODO: __pow__

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s11 = multi_slice(axes, [1, 1, 1])
            tensor = ket.tensor.copy()
            tensor[s11] *= -1
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover

# end class CCZ


# TODO: hamiltonian
class Deutsch(StdGate):
    r"""
    The Deutsch gate, a 3-qubit universal quantum gate.

    A controlled-controlled-i*R_x(2*theta) gate.
    Note that Deutsch(pi/2) is the CCNOT gate.

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
    _diagram_labels = (CTRL, CTRL, 'iRx({theta})^2')

    def __init__(self,
                 theta: Variable,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1, q2])

    @cached_property
    def tensor(self) -> BKTensor:
        theta, = self.parameters()
        ctheta = bk.ccast(theta)

        unitary = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1.0j * bk.cos(ctheta), bk.sin(ctheta)],
                   [0, 0, 0, 0, 0, 0, bk.sin(ctheta), 1.0j * bk.cos(ctheta)]]
        return bk.astensorproduct(unitary)

    # TODO: Specializes to Toffoli

# end class Deutsch
