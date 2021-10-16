# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import fractional_matrix_power as matpow

from .base import BaseGate, Variable, _asarray
from .bits import Qubits

# standard workaround to avoid circular imports from type hints
if TYPE_CHECKING:
    # Numpy typing introduced in v1.20, which may not be installed by default
    from numpy.typing import ArrayLike  # pragma: no cover

__all__ = ("Unitary",)


class Unitary(BaseGate):
    """
    A quantum logic gate specified by an explicit unitary operator.
    """

    def __init__(self, operator: "ArrayLike", qubits: Qubits) -> None:
        super().__init__(qubits=qubits)
        self._operator = _asarray(operator, ndim=2)

    @classmethod
    def from_gate(cls, gate: BaseGate) -> "Unitary":
        """Construct an instance of Unitary from another quantum gate, with the
        same operator and qubits."""
        return cls(gate.operator, gate.qubits)

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
