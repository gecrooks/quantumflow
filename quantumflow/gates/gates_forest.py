
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Gates peculiar to Rigetti's Forest
"""

# TODO: Rename CPHASE to CPhase, PSWAP to PSwap

from ..ops import Gate, StdGate
from ..qubits import Qubit
from ..utils import cached_property
from ..variables import Variable
from ..config import CTRL, NCTRL
from ..paulialgebra import Pauli, sZ

from ..backends import backend as bk
from ..backends import BKTensor

pi = bk.pi
PI = bk.PI


__all__ = ('CPHASE', 'CPHASE00', 'CPHASE01', 'CPHASE10', 'PSWAP')


class CPHASE(StdGate):
    r"""A 2-qubit 11 phase-shift gate

    .. math::
        \text{CPHASE}(\theta) \equiv \text{diag}(1, 1, 1, e^{i \theta})
    """
    interchangeable = True
    diagonal = True
    _diagram_labels = [CTRL+'({theta})', CTRL+'({theta})']

    def __init__(self, theta: Variable,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        theta = self.params['theta']
        return -theta*(1 + sZ(q0)*sZ(q1) - sZ(q0) - sZ(q1))/4

    @cached_property
    def tensor(self) -> BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, bk.exp(1j * ctheta)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CPHASE':
        return self ** -1

    def __pow__(self, t: Variable) -> 'CPHASE':
        theta = self.params['theta'] * t
        return CPHASE(theta, *self.qubits)
# end class CPHASE


class CPHASE00(StdGate):
    r"""A 2-qubit 00 phase-shift gate

    .. math::
        \text{CPHASE00}(\theta) \equiv \text{diag}(e^{i \theta}, 1, 1, 1)
    """
    interchangeable = True
    diagonal = True
    _diagram_labels = [NCTRL+'({theta})', NCTRL+'({theta})']

    def __init__(self, theta: Variable,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        theta = self.params['theta']
        return -theta*(1 + sZ(q0)*sZ(q1) + sZ(q0) + sZ(q1))/(4)

    @cached_property
    def tensor(self) -> BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[bk.exp(1j * ctheta), 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 1.0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CPHASE00':
        return self ** -1

    def __pow__(self, t: Variable) -> 'CPHASE00':
        theta = self.params['theta']
        return CPHASE00(theta*t, *self.qubits)

# end class CPHASE00


class CPHASE01(StdGate):
    r"""A 2-qubit 01 phase-shift gate

    .. math::
        \text{CPHASE01}(\theta) \equiv \text{diag}(1, e^{i \theta}, 1, 1)
    """
    diagonal = True
    _diagram_labels = [NCTRL+'({theta})', CTRL+'({theta})']

    def __init__(self, theta: Variable,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        theta = self.params['theta']
        return -theta*(1 - sZ(q0)*sZ(q1) + sZ(q0) - sZ(q1))/(4)

    @cached_property
    def tensor(self) -> BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0, 0, 0],
                   [0, bk.exp(1j * ctheta), 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 1.0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CPHASE01':
        return self ** -1

    def __pow__(self, t: Variable) -> 'CPHASE01':
        theta = self.params['theta']
        return CPHASE01(theta*t, *self.qubits)

# end class CPHASE01


class CPHASE10(StdGate):
    r"""A 2-qubit 10 phase-shift gate

    .. math::
        \text{CPHASE10}(\theta) \equiv \text{diag}(1, 1, e^{i \theta}, 1)
    """
    diagonal = True
    _diagram_labels = [CTRL+'({theta})', NCTRL+'({theta})']

    def __init__(self, theta: Variable,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @property
    def hamiltonian(self) -> Pauli:
        q0, q1 = self.qubits
        theta = self.params['theta']
        return -theta*(1 - sZ(q0)*sZ(q1) - sZ(q0) + sZ(q1))/(4)

    @cached_property
    def tensor(self) -> BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, bk.exp(1j * ctheta), 0],
                   [0, 0, 0, 1.0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CPHASE10':
        return self ** -1

    def __pow__(self, t: Variable) -> 'CPHASE10':
        theta = self.params['theta']
        return CPHASE10(theta*t, *self.qubits)

# end class CPHASE10


class PSWAP(StdGate):
    r"""A 2-qubit parametric-swap gate, as defined by Quil.
    Interpolates between SWAP (theta=0) and iSWAP (theta=pi/2).

    Locally equivalent to ``CAN(1/2, 1/2, 1/2 - theta/pi)``

    .. math::
        \text{PSWAP}(\theta) \equiv \begin{pmatrix} 1&0&0&0 \\
        0&0&e^{i\theta}&0 \\ 0&e^{i\theta}&0&0 \\ 0&0&0&1 \end{pmatrix}
    """
    interchangeable = True
    monomial = True

    def __init__(self, theta: Variable,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @cached_property
    def tensor(self) -> BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[[[1, 0], [0, 0]], [[0, 0], [bk.exp(ctheta * 1.0j), 0]]],
                   [[[0, bk.exp(ctheta * 1.0j)], [0, 0]], [[0, 0], [0, 1]]]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> Gate:
        theta = self.params['theta']
        theta = 2. * pi - theta % (2. * pi)
        return PSWAP(theta, *self.qubits)

# end class PSWAP
