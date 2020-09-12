# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates peculiar to Cirq
"""

import numpy as np

from .. import tensors, var
from ..config import SWAP_TARGET
from ..ops import StdGate
from ..paulialgebra import Pauli
from ..qubits import Qubit
from ..tensors import QubitTensor
from ..utils import cached_property
from ..var import Variable
from .stdgates_1q import I, X, XPow, ZPow
from .stdgates_2q import CZ, Swap

__all__ = ("PhasedX", "PhasedXPow", "FSim", "FSwap", "FSwapPow", "Sycamore")


# Kudos: Phased X gate and powers of phased X gates adapted from Cirq


class PhasedX(StdGate):
    r"""A phased X gate, equivalent to the circuit
    ───Z^-p───X───Z^p───
    """
    # Kudos: Adapted from Cirq

    _diagram_labels = ["PhX({p})"]
    cv_tensor_structure = "monomial"

    def __init__(self, p: Variable, q0: Qubit) -> None:
        super().__init__(params=[p], qubits=[q0])

    @cached_property
    def tensor(self) -> QubitTensor:
        p = var.asfloat(self.param("p"))
        gate = ZPow(p, 0) @ X(0) @ ZPow(-p, 0)
        return gate.tensor

    @property
    def H(self) -> "PhasedX":
        return self

    def __pow__(self, t: Variable) -> "PhasedXPow":
        p = self.param("p")
        return PhasedXPow(p, t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        p = self.param("p") % 2
        if np.isclose(p, 0.0) or np.isclose(p, 2.0):
            return X(*qbs)
        return self


# end class PhasedX


class PhasedXPow(StdGate):
    """A phased X gate raised to a power.

    Equivalent to the circuit ───Z^-p───X^t───Z^p───

    """

    # Kudos: Adapted from Cirq

    _diagram_labels = ["PhX({p})^{t}"]

    def __init__(self, p: Variable, t: Variable, q0: Qubit) -> None:
        super().__init__(params=[p, t], qubits=[q0])

    @cached_property
    def tensor(self) -> QubitTensor:
        p, t = self.params
        gate = ZPow(p, 0) @ XPow(t, 0) @ ZPow(-p, 0)
        return gate.tensor

    @property
    def H(self) -> "PhasedXPow":
        return self ** -1

    def __pow__(self, t: Variable) -> "PhasedXPow":
        p, s = self.params
        return PhasedXPow(p, s * t, *self.qubits)

    def specialize(self) -> StdGate:
        qbs = self.qubits
        p = self.param("p") % 2
        t = self.param("t") % 2
        if np.isclose(t, 0.0) or np.isclose(t, 2.0):
            return I(*qbs)
        if np.isclose(p, 0.0):
            return XPow(t, *qbs).specialize()
        return self


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
                0 & -i sin(\theta)  & \cos(\theta) & 0 \\
                0 & 0 & 0 & e^{-i\phi)}
            \end{pmatrix}
    """
    # Kudos: Adapted from Cirq

    cv_interchangeable = True

    def __init__(self, theta: Variable, phi: Variable, q0: Qubit, q1: Qubit):
        """
        Args:
            theta: Swap angle on the span(|01⟩, |10⟩) subspace, in radians.
            phi: Phase angle, in radians, applied to |11⟩ state.
        """
        super().__init__(params=[theta, phi], qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        theta = var.asfloat(self.param("theta"))
        phi = var.asfloat(self.param("phi"))

        unitary = [
            [1, 0, 0, 0],
            [0, np.cos(theta), -1.0j * np.sin(theta), 0],
            [0, -1.0j * np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, np.exp(-1j * phi)],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "FSim":
        return self ** -1

    def __pow__(self, t: Variable) -> "FSim":
        theta, phi = self.params
        return FSim(theta * t, phi * t, *self.qubits)


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
                        0 & 0 & i & 0 \\
                        0 & i & 0 & 0 \\
                        0 & 0 & 0 & -1
                       \end{pmatrix}

    """

    cv_interchangable = True
    cv_tensor_structure = "monomial"
    _diagram_labels = [SWAP_TARGET + "ᶠ", SWAP_TARGET + "ᶠ"]

    def __init__(self, q0: Qubit, q1: Qubit):
        super().__init__(qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return Swap(q0, q1).hamiltonian + CZ(q0, q1).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        return FSwapPow(1, *self.qubits).tensor

    @property
    def H(self) -> "FSwapPow":
        return FSwapPow(-1, *self.qubits)

    def __pow__(self, t: Variable) -> "FSwapPow":
        return FSwapPow(t, *self.qubits)


# End class FSwap


# Kudos: Adapted from OpenFermion-Cirq
# https://github.com/quantumlib/OpenFermion-Cirq/blob/master/openfermioncirq/gates/common_gates.py
class FSwapPow(StdGate):
    """Powers of the fermionic swap (FSwap) gate.
    Locally equivalent to ``Can(t/2, t/2, 0)``
    """

    cv_interchangable = True
    _diagram_labels = [SWAP_TARGET + "ᶠ^{t}", SWAP_TARGET + "ᶠ^{t}"]

    def __init__(self, t: Variable, q0: Qubit, q1: Qubit):
        super().__init__(params=[t], qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        return self.param("t") * FSwap(q0, q1).hamiltonian

    @cached_property
    def tensor(self) -> QubitTensor:
        t = self.float_param("t")
        c = np.cos(np.pi * t / 2)
        s = np.sin(np.pi * t / 2)
        g = np.exp(0.5j * np.pi * t)
        p = np.exp(1.0j * np.pi * t)
        unitary = [
            [1, 0, 0, 0],
            [0, g * c, -1.0j * g * s, 0],
            [0, -1.0j * g * s, g * c, 0],
            [0, 0, 0, p],
        ]
        return tensors.asqutensor(unitary)

    @property
    def H(self) -> "FSwapPow":
        return self ** -1

    def __pow__(self, e: Variable) -> "FSwapPow":
        (t,) = self.params
        return FSwapPow(e * t, *self.qubits)


# End class FSwapPow


class Sycamore(StdGate):
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

    cv_interchangeable = True
    cv_tensor_structure = "monomial"

    def __init__(self, q0: Qubit, q1: Qubit):
        super().__init__(qubits=[q0, q1])

    @cached_property
    def tensor(self) -> QubitTensor:
        return FSim(np.pi / 2, np.pi / 6, 0, 1).tensor

    @property
    def H(self) -> "FSim":
        return self ** -1

    def __pow__(self, t: Variable) -> "FSim":
        return FSim(t * np.pi / 2, t * np.pi / 6, *self.qubits)


# end class Sycamore


# fin
