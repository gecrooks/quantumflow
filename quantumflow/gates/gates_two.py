"""
QuantumFlow: One qubit gates
"""


from math import pi
import numpy as np

from .. import backend as bk
from ..qubits import Qubit
from ..ops import Gate
from ..states import State
from ..utils import multi_slice, immutable_property

from .gates_one import V, V_H
from .gates_utils import control_gate

__all__ = ['CZ', 'CNOT', 'SWAP', 'ISWAP', 'CPHASE00', 'CPHASE01', 'CPHASE10',
           'CPHASE', 'PSWAP',
           'CAN', 'XX', 'YY', 'ZZ', 'PISWAP', 'EXCH', 'CTX',
           'BARENCO', 'CV', 'CV_H', 'CY', 'CH', 'FSIM']


# Standard 2 qubit gates

class CZ(Gate):
    r"""A controlled-Z gate

    Equivalent to ``controlled_gate(Z())`` and locally equivalent to
    ``CAN(1/2,0,0)``

    .. math::
        \text{CZ}() = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                    0&0&1&0 \\ 0&0&0&-1 \end{pmatrix}
    """
    interchangeable = True

    diagonal = True

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CZ':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'CPHASE':
        # FIXME
        return CPHASE(pi * t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s11 = multi_slice(axes, [1, 1])
            tensor = ket.tensor.copy()
            # tensor[s11] = -ket.tensor[s11]
            tensor[s11] *= -1
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover
# End class CZ


class CNOT(Gate):
    r"""A controlled-not gate

    Equivalent to ``controlled_gate(X())``, and
    locally equivalent to ``CAN(1/2, 0, 0)``

     .. math::
        \text{CNOT}() \equiv \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                            0&0&0&1 \\ 0&0&1&0 \end{pmatrix}
    """

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CNOT':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'CTX':
        return CTX(t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s10 = multi_slice(axes, [1, 0])
            s11 = multi_slice(axes, [1, 1])
            tensor = ket.tensor.copy()
            tensor[s10] = ket.tensor[s11]
            tensor[s11] = ket.tensor[s10]
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover
# end class CNOT


class CTX(Gate):
    r"""Powers of the CNOT gate.

    Equivalent to ``controlled_gate(TX(t))``, and locally equivelant to
    ``CAN(t/2, 0 ,0)``. Cirq calls this a ``CNotPowGate``.

    .. math::
        \text{CTX}(t) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos(\frac{\theta}{2}) e^{i\frac{\theta}{2}}
                      & -i \sin(\frac{\theta}{2}) e^{i\frac{\theta}{2}} \\
                0 & 0 & -i \sin(\frac{\theta}{2}) e^{i\frac{\theta}{2}}
                      & \cos(\frac{\theta}{2}) e^{i\frac{\theta}{2}}
            \end{pmatrix}

    args:
        t:  turns (powers of the CNOT gate, or controlled-powers of X)
        q0: control qubit
        q1: target qubit
    """
    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        # t = t % 2
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        t = self.params['t']
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        cht = bk.cos(ctheta / 2)
        sht = bk.sin(ctheta / 2)
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, phase * cht, phase * -1.0j * sht],
                   [0, 0, phase * -1.0j * sht, phase * cht]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CTX':
        return self ** -1

    def __pow__(self, t: float) -> 'CTX':
        t = self.params['t'] * t
        return CTX(t, *self.qubits)


class SWAP(Gate):
    r"""A 2-qubit swap gate

    Locally equivalent to ``CAN(1/2, 1/2, 1/2)``.

    .. math::
        \text{SWAP}() \equiv
            \begin{pmatrix}
            1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1
            \end{pmatrix}

    """
    interchangeable = True

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'SWAP':
        return self  # Hermitian

    # TODO: Powers give EXCH gate

    # TESTME DOCME
    # TODO: evolve
    def run(self, ket: State) -> State:
        idx0, idx1 = ket.qubit_indices(self.qubits)
        perm = list(range(ket.qubit_nb))
        perm[idx0] = idx1
        perm[idx1] = idx0
        tensor = bk.transpose(ket.tensor, perm)
        return State(tensor, ket.qubits, ket.memory)


class ISWAP(Gate):
    r"""A 2-qubit iswap gate

    Locally equivalent to ``CAN(1/2,1/2,0)``.

    .. math::
        \text{ISWAP}() \equiv
        \begin{pmatrix} 1&0&0&0 \\ 0&0&i&0 \\ 0&i&0&0 \\ 0&0&0&1 \end{pmatrix}

    """
    interchangeable = True

    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.array([[1, 0, 0, 0],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [0, 0, 0, 1]])

        return bk.astensorproduct(unitary)

    # TODO: H, __pow__

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            tensor = ket.tensor.copy()
            axes = ket.qubit_indices(self.qubits)
            s10 = multi_slice(axes, [1, 0])
            s01 = multi_slice(axes, [0, 1])
            tensor[s01] = 1.0j * ket.tensor[s10]
            tensor[s10] = 1.0j * ket.tensor[s01]
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover


class CPHASE00(Gate):
    r"""A 2-qubit 00 phase-shift gate

    .. math::
        \text{CPHASE00}(\theta) \equiv \text{diag}(e^{i \theta}, 1, 1, 1)
    """
    interchangeable = True
    diagonal = True

    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
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

    def __pow__(self, t: float) -> 'CPHASE00':
        theta = self.params['theta']
        return CPHASE00(theta*t, *self.qubits)


class CPHASE01(Gate):
    r"""A 2-qubit 01 phase-shift gate

    .. math::
        \text{CPHASE01}(\theta) \equiv \text{diag}(1, e^{i \theta}, 1, 1)
    """
    diagonal = True

    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
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

    def __pow__(self, t: float) -> 'CPHASE01':
        theta = self.params['theta']
        return CPHASE01(theta*t, *self.qubits)


class CPHASE10(Gate):
    r"""A 2-qubit 10 phase-shift gate

    .. math::
        \text{CPHASE10}(\theta) \equiv \text{diag}(1, 1, e^{i \theta}, 1)
    """
    diagonal = True

    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
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

    def __pow__(self, t: float) -> 'CPHASE10':
        theta = self.params['theta']
        return CPHASE10(theta*t, *self.qubits)


class CPHASE(Gate):
    r"""A 2-qubit 11 phase-shift gate

    .. math::
        \text{CPHASE}(\theta) \equiv \text{diag}(1, 1, 1, e^{i \theta})
    """
    interchangeable = True
    diagonal = True

    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
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

    def __pow__(self, t: float) -> 'CPHASE':
        theta = self.params['theta'] * t
        return CPHASE(theta, *self.qubits)


class PSWAP(Gate):
    r"""A 2-qubit parametric-swap gate, as defined by Quil.
    Interpolates between SWAP (theta=0) and iSWAP (theta=pi/2).

    Locally equivalent to ``CAN(1/2, 1/2, 1/2 - theta/pi)``

    .. math::
        \text{PSWAP}(\theta) \equiv \begin{pmatrix} 1&0&0&0 \\
        0&0&e^{i\theta}&0 \\ 0&e^{i\theta}&0&0 \\ 0&0&0&1 \end{pmatrix}
    """
    interchangeable = True

    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
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

    # TODO: __pow__


class PISWAP(Gate):
    r"""A parametric iswap gate, generated from XY interaction.

    Locally equivalent to CAN(t,t,0), where t = theta / (2 * pi)

    .. math::
        \text{PISWAP}(\theta) \equiv
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(2\theta) & i \sin(2\theta) & 0 \\
                0 & i \sin(2\theta) & \cos(2\theta) & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    """
    interchangeable = True

    def __init__(self, theta: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[[[1, 0], [0, 0]],
                    [[0, bk.cos(2*ctheta)], [bk.sin(2*ctheta) * 1j, 0]]],
                   [[[0, bk.sin(2*ctheta) * 1j], [bk.cos(2*ctheta), 0]],
                    [[0, 0], [0, 1]]]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'PISWAP':
        return self ** -1

    def __pow__(self, t: float) -> 'PISWAP':
        theta = self.params['theta'] * t
        return PISWAP(theta, *self.qubits)


# Other 2-qubit gates

# TODO: Add references and explanation
# DOCME: Comment on sign conventions.
class CAN(Gate):
    r"""The canonical 2-qubit gate

    The canonical decomposition of 2-qubits gates removes local 1-qubit
    rotations, and leaves only the non-local interactions.

    .. math::
        \text{CAN}(t_x, t_y, t_z) \equiv
            \exp\Big\{-i\frac{\pi}{2}(t_x X\otimes X
            + t_y Y\otimes Y + t_z Z\otimes Z)\Big\}

    """
    interchangeable = True

    def __init__(self,
                 tx: float, ty: float, tz: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(tx=tx, ty=ty, tz=tz), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        tx, ty, tz = self.params.values()
        xx = XX(tx)
        yy = YY(ty)
        zz = ZZ(tz)

        gate = yy @ xx
        gate = zz @ gate
        return gate.tensor

    @property
    def H(self) -> 'CAN':
        return self ** -1

    def __pow__(self, t: float) -> 'CAN':
        tx, ty, tz = self.params.values()
        return CAN(tx * t, ty * t, tz * t, *self.qubits)


class XX(Gate):
    r"""A parametric 2-qubit gate generated from an XX interaction,

    Equivalent to ``CAN(t,0,0)``.

    XX(1/2) is the Mølmer-Sørensen gate.

    Ref: Sørensen, A. & Mølmer, K. Quantum computation with ions in thermal
    motion. Phys. Rev. Lett. 82, 1971–1974 (1999)

    Args:
        t:
    """
    interchangeable = True

    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        t, = self.params.values()
        theta = bk.ccast(pi * t)
        unitary = [[bk.cos(theta / 2), 0, 0, -1.0j * bk.sin(theta / 2)],
                   [0, bk.cos(theta / 2), -1.0j * bk.sin(theta / 2), 0],
                   [0, -1.0j * bk.sin(theta / 2), bk.cos(theta / 2), 0],
                   [-1.0j * bk.sin(theta / 2), 0, 0, bk.cos(theta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'XX':
        return self ** -1

    def __pow__(self, t: float) -> 'XX':
        t = self.params['t'] * t
        return XX(t, *self.qubits)


class YY(Gate):
    r"""A parametric 2-qubit gate generated from a YY interaction.

    Equivalent to ``CAN(0,t,0)``, and locally equivalent to
    ``CAN(t,0,0)``

    Args:
        t:
    """
    interchangeable = True

    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        t, = self.params.values()
        theta = bk.ccast(pi * t)
        unitary = [[bk.cos(theta / 2), 0, 0, 1.0j * bk.sin(theta / 2)],
                   [0, bk.cos(theta / 2), -1.0j * bk.sin(theta / 2), 0],
                   [0, -1.0j * bk.sin(theta / 2), bk.cos(theta / 2), 0],
                   [1.0j * bk.sin(theta / 2), 0, 0, bk.cos(theta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'YY':
        return self ** -1

    def __pow__(self, t: float) -> 'YY':
        t = self.params['t'] * t
        return YY(t, *self.qubits)


class ZZ(Gate):
    r"""A parametric 2-qubit gate generated from a ZZ interaction.

    Equivalent to ``CAN(0,0,t)``, and locally equivalent to
    ``CAN(t,0,0)``

    Args:
        t:
    """
    interchangeable = True
    diagonal = True

    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        t, = self.params.values()
        theta = bk.ccast(pi * t)
        unitary = [[[[bk.cis(-theta / 2), 0], [0, 0]],
                    [[0, bk.cis(theta / 2)], [0, 0]]],
                   [[[0, 0], [bk.cis(theta / 2), 0]],
                    [[0, 0], [0, bk.cis(-theta / 2)]]]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'ZZ':
        return self ** -1

    def __pow__(self, t: float) -> 'ZZ':
        t = self.params['t'] * t
        return ZZ(t, *self.qubits)


class EXCH(Gate):
    r"""A 2-qubit parametric gate generated from an exchange interaction.

    Equivalent to CAN(t,t,t)

    """
    interchangeable = True

    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(params=dict(t=t), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        t, = self.params.values()
        unitary = CAN(t, t, t).tensor
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'EXCH':
        return self ** -1

    def __pow__(self, t: float) -> 'EXCH':
        t = self.params['t'] * t
        return EXCH(t, *self.qubits)


# More 2-qubit gates

class CV(Gate):
    r"""A controlled V (sqrt of CNOT) gate."""
    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        q0, q1 = self.qubits
        return control_gate(q0, V(q1)).tensor

    @property
    def H(self) -> 'CV_H':
        return CV_H(*self.qubits)

    # TODO: __pow__


class CV_H(Gate):
    r"""A controlled V (sqrt of CNOT) gate."""
    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        q0, q1 = self.qubits
        return control_gate(q0, V_H(q1)).tensor

    @property
    def H(self) -> 'CV':
        return CV(*self.qubits)

    # TODO: __pow__


class BARENCO(Gate):
    """A universal two-qubit gate:

    Ref:
        A Universal Two–Bit Gate for Quantum Computation, A. Barenco (1996)
        https://arxiv.org/pdf/quant-ph/9505016.pdf
    """

    def __init__(self,
                 alpha: float,
                 phi: float,
                 theta: float,
                 q0: Qubit = 0,
                 q1: Qubit = 1) -> None:
        params = dict(alpha=alpha, phi=phi, theta=theta)
        qubits = [q0, q1]
        super().__init__(params=params, qubits=qubits)

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        alpha, phi, theta = self.params.values()

        calpha = bk.ccast(alpha)
        cphi = bk.ccast(phi)
        ctheta = bk.ccast(theta)

        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, bk.cis(calpha) * bk.cos(ctheta),
                    -1j * bk.cis(calpha - cphi) * bk.sin(ctheta)],
                   [0, 0, -1j * bk.cis(calpha + cphi) * bk.sin(ctheta),
                    bk.cis(calpha) * bk.cos(ctheta)]]
        return bk.astensorproduct(unitary)

    # TODO: H, __pow__


class CY(Gate):
    r"""A controlled-Y gate

    Equivalent to ``controlled_gate(Y())`` and locally equivalent to
    ``CAN(1/2,0,0)``

    .. math::
        \text{CY}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & i & 0
            \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, -1j],
                              [0, 0, 1j, 0]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CY':
        return self  # Hermitian

    # TODO: __pow__


class CH(Gate):
    r"""A controlled-Hadamard gate

    Equivalent to ``controlled_gate(H())`` and locally equivalent to
    ``CAN(1/2, 0, 0)``

    .. math::
        \text{CH}() =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \tfrac{1}{\sqrt{2}} &  \tfrac{1}{\sqrt{2}} \\
                0 & 0 & \tfrac{1}{\sqrt{2}} & -\tfrac{1}{\sqrt{2}}
            \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        super().__init__(qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                              [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'CH':
        return self  # Hermitian

    # TODO: __pow__


class FSIM(Gate):
    r"""Fermionic simulation gate family.

    Contains all two qubit interactions that preserve excitations, up to
    single-qubit rotations and global phase.

    Locally equivelent to ``CAN(theta/pi, theta/pi, phi/(2*pi))``

    .. math::
        \text{FSIM}(\theta, \phi) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\theta) & -i \sin(\theta) & 0 \\
                0 & -i sin(\theta)  & \cos(\theta) & 0 \\
                0 & 0 & 0 & e^{-i\phi)}
            \end{pmatrix}
    """
    # Kudos: Adaped from Cirq

    interchangeable = True

    def __init__(self, theta: float, phi: float, q0: Qubit = 0, q1: Qubit = 1):
        """
        Args:
            theta: Swap angle on the span(|01⟩, |10⟩) subspace, in radians.
            phi: Phase angle, in radians, applied to |11⟩ state.
        """
        super().__init__(params=dict(theta=theta, phi=phi), qubits=[q0, q1])

    @immutable_property
    def tensor(self) -> bk.BKTensor:
        theta, phi = list(self.params.values())
        theta = bk.ccast(theta)
        phi = bk.ccast(phi)
        unitary = [[1, 0, 0, 0],
                   [0, bk.cos(theta), -1.0j * bk.sin(theta), 0],
                   [0, -1.0j * bk.sin(theta), bk.cos(theta), 0],
                   [0, 0, 0, bk.cis(-phi)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'FSIM':
        return self ** -1

    def __pow__(self, t: float) -> 'FSIM':
        theta, phi = list(self.params.values())
        return FSIM(theta * t, phi * t, *self.qubits)

# fin
