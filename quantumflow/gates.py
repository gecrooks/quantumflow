# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, Tuple, cast

import numpy as np
import scipy.linalg
import sympy as sym
from scipy.linalg import fractional_matrix_power as matpow
from scipy.stats import unitary_group

from .config import quantum_dtype
from .operations import OperatorStructure, CompositeOperation, Gate
from .pauli import Pauli
from .states import Addrs, Qubits, Variable

# standard workaround to avoid circular imports from type hints
if TYPE_CHECKING:
    # Numpy typing introduced in v1.20, which may not be installed by default
    from numpy.typing import ArrayLike  # pragma: no cover

__all__ = ("Identity", "Unitary", "CompositeGate", "RandomGate")


class CompositeGate(CompositeOperation, Gate):

    _elements: Tuple[Gate, ...]

    def __init__(
        self,
        *elements: Gate,
        qubits: Qubits = None,
        addrs: Addrs = None,
    ) -> None:
        CompositeOperation.__init__(self, *elements, qubits=qubits, addrs=addrs)

        for elem in self:
            if not isinstance(elem, Gate):
                raise ValueError("Elements of a composite gate must be gates")

    # TODO: Move up to CompositeOperation?
    @property
    def H(self) -> "CompositeGate":
        elements = [elem.H for elem in self._elements[::-1]]
        return CompositeGate(*elements, qubits=self.qubits, addrs=self.addrs)

    @property
    def operator(self) -> np.ndarray:
        from .circuits import Circuit

        if self._operator is None:
            self._operator = Circuit(*self, qubits=self.qubits).asgate().operator
        return self._operator

    def __contains__(self, key: Any) -> bool:
        return key in self._elements

    def __pow__(self, t: Variable) -> "Unitary":
        return Unitary.from_gate(self) ** t


class Identity(Gate):
    """
    A multi-qubit identity gate.
    """

    cv_hermitian = True
    cv_operator_structure = OperatorStructure.identity

    def __init__(self, qubits: Qubits) -> None:
        super().__init__(qubits=qubits)

    @property
    def H(self) -> "Identity":
        return self  # Hermitian

    @property
    def operator(self) -> np.ndarray:
        if self._operator is None:
            self._operator = np.eye(2 ** self.qubit_nb)
        return self._operator

    @property
    def sym_operator(self) -> sym.Matrix:
        if self._sym_operator is None:
            self._sym_operator = sym.eye(2 ** self.qubit_nb)
        return self._sym_operator

    def __pow__(self, t: Variable) -> "Identity":
        return self


# End class Identity


class Unitary(Gate):
    """
    A quantum logic gate specified by an explicit unitary operator.
    """

    def __init__(self, operator: "ArrayLike", qubits: Qubits) -> None:
        super().__init__(qubits=qubits)
        self._operator = np.asanyarray(operator, dtype=quantum_dtype)

    @classmethod
    def from_gate(cls, gate: Gate) -> "Unitary":
        """Construct an instance of Unitary from another quantum gate, with the
        same operator and qubits."""
        return cls(gate.operator, gate.qubits)

    @classmethod
    def from_hamiltonian(cls, hamiltonian: Pauli, qubits: Qubits) -> "Unitary":
        U = scipy.linalg.expm(-1j * hamiltonian.operator)
        gate = cast(Unitary, Unitary(U, hamiltonian.qubits) @ Identity(qubits))
        return gate

    @property
    def H(self) -> "Unitary":
        return Unitary(self.operator.conj().T, self.qubits)

    @property
    def operator(self) -> np.ndarray:
        return self._operator

    def __pow__(self, t: Variable) -> "Unitary":
        matrix = matpow(self.operator, t)
        return Unitary(matrix, self.qubits)


# End class Unitary


class RandomGate(Unitary):
    r"""Returns a random unitary gate acting on the given qubits, drawn
    from Harr measure.

    Ref:
        "How to generate random matrices from the classical compact groups",
        Francesco Mezzadri, Notices Am. Math. Soc. 54, 592 (2007).
        arXiv:math-ph/0609050
    """

    def __init__(self, qubits: Qubits) -> None:
        qubits = tuple(qubits)
        matrix = unitary_group.rvs(2 ** len(qubits))
        super().__init__(matrix, qubits)


# end class RandomGate
