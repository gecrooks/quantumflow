
"""
QuantumFlow: Three qubit gates
"""

import numpy as np

from .. import backend as bk
from ..qubits import Qubit
from ..states import State, Density
from ..ops import Gate


__all__ = ['CCNOT', 'CSWAP', 'CCZ', 'IDEN']


class CCNOT(Gate):
    r"""
    A 3-qubit Toffoli gate. A controlled, controlled-not.

    Equivalent to ``controlled_gate(cnot())``

    .. math::
        \text{CCNOT}() \equiv \begin{pmatrix}
                1& 0& 0& 0& 0& 0& 0& 0 \\
                0& 1& 0& 0& 0& 0& 0& 0 \\
                0& 0& 1& 0& 0& 0& 0& 0 \\
                0& 0& 0& 1& 0& 0& 0& 0 \\
                0& 0& 0& 0& 1& 0& 0& 0 \\
                0& 0& 0& 0& 0& 1& 0& 0 \\
                0& 0& 0& 0& 0& 0& 0& 1 \\
                0& 0& 0& 0& 0& 0& 1& 0
            \end{pmatrix}

    """
    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def tensor(self) -> bk.BKTensor:
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

    # TODO: __pow__


class CSWAP(Gate):
    r"""
    A 3-qubit Fredkin gate. A controlled swap.

    Equivalent to ``controlled_gate(swap())``

    .. math::
        \text{CSWAP}() \equiv \begin{pmatrix}
                1& 0& 0& 0& 0& 0& 0& 0 \\
                0& 1& 0& 0& 0& 0& 0& 0 \\
                0& 0& 1& 0& 0& 0& 0& 0 \\
                0& 0& 0& 1& 0& 0& 0& 0 \\
                0& 0& 0& 0& 1& 0& 0& 0 \\
                0& 0& 0& 0& 0& 0& 1& 0 \\
                0& 0& 0& 0& 0& 1& 0& 0 \\
                0& 0& 0& 0& 0& 0& 0& 1
            \end{pmatrix}
    """
    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def tensor(self) -> bk.BKTensor:
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


class CCZ(Gate):
    r"""
    A controlled, controlled-Z.

    Equivalent to ``controlled_gate(CZ())``

    .. math::
        \text{CSWAP}() \equiv \begin{pmatrix}
                1& 0& 0& 0& 0& 0& 0& 0 \\
                0& 1& 0& 0& 0& 0& 0& 0 \\
                0& 0& 1& 0& 0& 0& 0& 0 \\
                0& 0& 0& 1& 0& 0& 0& 0 \\
                0& 0& 0& 0& 1& 0& 0& 0 \\
                0& 0& 0& 0& 0& 1& 0& 0 \\
                0& 0& 0& 0& 0& 0& 1& 0 \\
                0& 0& 0& 0& 0& 0& 0& -1
            \end{pmatrix}
    """
    interchangeable = True

    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        super().__init__(qubits=[q0, q1, q2])

    @property
    def tensor(self) -> bk.BKTensor:
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


class IDEN(Gate):                                      # noqa: E742
    r"""
    The multi-qubit identity gate.
    """
    def __init__(self, *qubits: Qubit) -> None:
        if not qubits:
            qubits = (0,)
        super().__init__(qubits=qubits)

    @property
    def tensor(self) -> bk.BKTensor:
        return bk.astensorproduct(np.eye(2 ** self.qubit_nb))

    def run(self, ket: State) -> State:
        return ket

    def evolve(self, rho: Density) -> Density:
        return rho

    @property
    def H(self) -> 'IDEN':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'IDEN':
        return self
