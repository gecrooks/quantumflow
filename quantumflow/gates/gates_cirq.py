
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates peculiar to Cirq
"""

import numpy as np
from typing import Iterator, Union

from ..config import SWAP_TARGET
from ..ops import Gate
from ..qubits import Qubit
from ..utils import cached_property
from ..variables import Variable
from .gates_one import TX, TZ, X, I
from .gates_two import SWAP, CZ, EXCH, CZPow
from ..paulialgebra import Pauli

from ..backends import backend as bk
from ..backends import BKTensor
pi = bk.pi
PI = bk.PI

__all__ = ('PhasedX', 'PhasedXPow', 'FSim', 'FSwap', 'FSwapPow', 'Sycamore')


# Kudos: Phased X gate and powers of phased X gates adapted from Cirq

class PhasedX(Gate):
    r""" A phased X gate, equivalent to the circuit
    ───Z^-p───X───Z^p───
    """
    # Kudos: Adapted from Cirq

    _diagram_labels = ['PhX({p})']

    def __init__(self, p: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(p=p), qubits=[q0])

    @cached_property
    def tensor(self) -> BKTensor:
        p = self.params['p']
        gate = TZ(p) @ X() @ TZ(-p)
        return gate.tensor

    @property
    def H(self) -> 'PhasedX':
        return self

    def __pow__(self, t: Variable) -> 'PhasedXPow':
        p = self.params['p']
        return PhasedXPow(p, t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        p = self.params['p'] % 2
        if np.isclose(p, 0.0) or np.isclose(p, 2.0):
            return X(*qbs)
        return self

# end class PhasedX


class PhasedXPow(Gate):
    """A phased X gate raised to a power.

    Equivalent to the circuit ───Z^-p───X^t───Z^p───

    """
    # Kudos: Adapted from Cirq

    _diagram_labels = ['PhX({p})^{t}']

    def __init__(self, p: Variable, t: Variable, q0: Qubit = 0) -> None:
        super().__init__(params=dict(p=p, t=t), qubits=[q0])

    @cached_property
    def tensor(self) -> BKTensor:
        p, t = self.params.values()
        gate = TZ(p) @ TX(t) @ TZ(-p)
        return gate.tensor

    @property
    def H(self) -> 'PhasedXPow':
        return self ** -1

    def __pow__(self, t: Variable) -> 'PhasedXPow':
        p, s = self.params.values()
        return PhasedXPow(p, s * t, *self.qubits)

    def specialize(self) -> Gate:
        qbs = self.qubits
        p = self.params['p'] % 2
        t = self.params['t'] % 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return I(*qbs)
        if np.isclose(p, 0.0):
            return TX(t, *qbs).specialize()
        return self

# end class PhasedXPow


class FSim(Gate):
    r"""Fermionic simulation gate family.

    Contains all two qubit interactions that preserve excitations, up to
    single-qubit rotations and global phase.

    Locally equivalent to ``Can(theta/pi, theta/pi, phi/(2*pi))``

    .. math::
        \text{FSim}(\theta, \phi) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\theta) & -i \sin(\theta) & 0 \\
                0 & -i sin(\theta)  & \cos(\theta) & 0 \\
                0 & 0 & 0 & e^{-i\phi)}
            \end{pmatrix}
    """
    # Kudos: Adapted from Cirq

    interchangeable = True

    def __init__(self, theta: Variable, phi: Variable,
                 q0: Qubit = 0, q1: Qubit = 1):
        """
        Args:
            theta: Swap angle on the span(|01⟩, |10⟩) subspace, in radians.
            phi: Phase angle, in radians, applied to |11⟩ state.
        """
        super().__init__(params=dict(theta=theta, phi=phi), qubits=[q0, q1])

    @cached_property
    def tensor(self) -> BKTensor:
        theta, phi = list(self.params.values())
        theta = bk.ccast(theta)
        phi = bk.ccast(phi)
        unitary = [[1, 0, 0, 0],
                   [0, bk.cos(theta), -1.0j * bk.sin(theta), 0],
                   [0, -1.0j * bk.sin(theta), bk.cos(theta), 0],
                   [0, 0, 0, bk.exp(-1j*phi)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'FSim':
        return self ** -1

    def __pow__(self, t: Variable) -> 'FSim':
        theta, phi = list(self.params.values())
        return FSim(theta * t, phi * t, *self.qubits)

# end class FSim


# Kudos: Adapted from OpenFermion-Cirq
# https://github.com/quantumlib/OpenFermion-Cirq/blob/master/openfermioncirq/gates/common_gates.py
class FSwap(Gate):
    r"""Fermionic swap gate. It swaps adjacent fermionic modes in
    the Jordan-Wigner representation.
    Locally equivalent to iSwap and ``Can(1/2, 1/2, 0)``

    .. math::
        \text{FSwap} = \begin{pmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 0 & i & 0 \\
                        0 & i & 0 & 0 \\
                        0 & 0 & 0 & -1
                       \end{pmatrix}

    """

    interchangable = True
    monomial = True
    _diagram_labels = [SWAP_TARGET+'ᶠ', SWAP_TARGET+'ᶠ']

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1):
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return SWAP(q0, q1).hamiltonian + CZ(q0, q1).hamiltonian

    @cached_property
    def tensor(self) -> BKTensor:
        return FSwapPow(1).tensor

    @property
    def H(self) -> 'FSwapPow':
        return FSwapPow(-1, *self.qubits)

    def __pow__(self, t: Variable) -> 'FSwapPow':
        return FSwapPow(t, *self.qubits)

    def decompose(self) -> 'Iterator[Union[SWAP, CZ]]':
        q0, q1 = self.qubits
        yield SWAP(q0, q1)
        yield CZ(q0, q1)

# End class FSwap


# Kudos: Adapted from OpenFermion-Cirq
# https://github.com/quantumlib/OpenFermion-Cirq/blob/master/openfermioncirq/gates/common_gates.py
class FSwapPow(Gate):
    """Powers of the fermionic swap (FSwap) gate.
    Locally equivalent to ``Can(t/2, t/2, 0)``
    """
    interchangable = True
    _diagram_labels = [SWAP_TARGET+'ᶠ^{t}', SWAP_TARGET+'ᶠ^{t}']

    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1):
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        t, = self.parameters()
        return t * FSwap(q0, q1).hamiltonian

    @cached_property
    def tensor(self) -> BKTensor:
        t, = self.parameters()
        t = bk.ccast(t)
        c = bk.cos(np.pi*t/2)
        s = bk.sin(np.pi*t/2)
        g = bk.exp(0.5j * np.pi * t)
        p = bk.exp(1.0j*np.pi*t)
        unitary = [[1, 0, 0, 0],
                   [0, g*c, -1.0j*g*s, 0],
                   [0, -1.0j * g*s, g*c, 0],
                   [0, 0, 0, p]]
        return bk.astensorproduct(unitary)

    def decompose(self) -> 'Iterator[Union[EXCH, CZPow]]':
        t, = self.parameters()
        q0, q1 = self.qubits
        yield SWAP(q0, q1) ** t
        yield CZ(q0, q1) ** t

    @property
    def H(self) -> 'FSwapPow':
        return self ** -1

    def __pow__(self, e: Variable) -> 'FSwapPow':
        t, = self.parameters()
        return FSwapPow(e * t, *self.qubits)

# End class FSwapPow


class Sycamore(Gate):
    r"""The Sycamore gate is a two-qubit gate equivalent to
    ``FSim(π/2, π/6)``, and locally equivalent to ``Can(1/2, 1/2, 1/6)``.

    This gate was used to demonstrate quantum on Google's Sycamore chip.

    .. math::
        \text{Sycamore}() =
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

    interchangeable = True

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1):
        super().__init__(qubits=[q0, q1])

    @cached_property
    def tensor(self) -> BKTensor:
        return FSim(np.pi/2, np.pi/6).tensor

    @property
    def H(self) -> 'FSim':
        return self ** -1

    def __pow__(self, t: Variable) -> 'FSim':
        return FSim(t*np.pi/2, t*np.pi/6, *self.qubits)

# end class Sycamore
