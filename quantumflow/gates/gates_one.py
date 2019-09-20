
"""
QuantumFlow: One qubit gates
"""

from math import sqrt, pi
import numpy as np

from .. import backend as bk
from ..qubits import Qubit
from ..states import State, Density
from ..ops import Gate
from ..utils import multi_slice, cached_property

# Standard 1 qubit gates


class I(Gate):                                      # noqa: E742
    r"""
    The 1-qubit identity gate.

    .. math::
        I() \equiv \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return bk.astensorproduct(np.eye(2))

    @property
    def H(self) -> 'I':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'I':
        return self

    def run(self, ket: State) -> State:
        return ket

    def evolve(self, rho: Density) -> Density:
        return rho


class X(Gate):
    r"""
    A 1-qubit Pauli-X gate.

    .. math::
        X() &\equiv \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
     """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[0, 1], [1, 0]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'X':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'TX':
        return TX(t, *self.qubits)

    def run(self, ket: State) -> State:
        # Fast implementation for X special case
        idx, = ket.qubit_indices(self.qubits)

        if bk.BACKEND == 'numpy':
            take = multi_slice(axes=[idx], items=[[1, 0]])
            tensor = ket.tensor[take]
            return State(tensor, ket.qubits, ket.memory)
        elif bk.BACKEND != 'ctf':                            # pragma: no cover
            tensor = bk.roll(ket.tensor, axis=idx, shift=1)
            return State(tensor, ket.qubits, ket.memory)
        return super().run(ket)                              # pragma: no cover

    # TODO: Optimized evolve method

# end class X


class Y(Gate):
    r"""
    A 1-qubit Pauli-Y gate.

    .. math::
        Y() &\equiv \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

    mnemonic: "Minus eye high".
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[0, -1.0j], [1.0j, 0]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'Y':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'TY':
        return TY(t, *self.qubits)

    def run(self, ket: State) -> State:
        # Since X and Z have fast optimizations, this is actually faster
        ket = Z(*self.qubits).run(ket)
        ket = X(*self.qubits).run(ket)
        return ket

# end class Y


class Z(Gate):
    r"""
    A 1-qubit Pauli-Z gate.

    .. math::
        Z() &\equiv \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
    """

    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 0], [0, -1.0]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'Z':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'TZ':
        return TZ(t, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(1, *self.qubits).run(ket)

# end class Z


class H(Gate):
    r"""
    A 1-qubit Hadamard gate.

    .. math::
        H() \equiv \frac{1}{\sqrt{2}}
        \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1, 1], [1, -1]]) / sqrt(2)
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> '_H':
        return self  # Hermitian

    def __pow__(self, t: float) -> 'TH':
        return TH(t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            axes = ket.qubit_indices(self.qubits)
            s0 = multi_slice(axes, [0])
            s1 = multi_slice(axes, [1])
            tensor = ket.tensor.copy()
            tensor[s1] -= tensor[s0]
            tensor[s1] *= -0.5
            tensor[s0] -= tensor[s1]
            tensor *= np.sqrt(2)
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)     # pragma: no cover


# Hack. H().H -> H, but the method shadows the class, so can't
# annotate directly.
_H = H

# End class H


class S(Gate):
    r"""
    A 1-qubit phase S gate, equivalent to ``PHASE(pi/2)``. The square root
    of the Z gate (up to global phase). Also commonly denoted as the P gate.

    .. math::
        S() \equiv \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1.0, 0.0], [0.0, 1.0j]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'S_H':
        return S_H(*self.qubits)

    def __pow__(self, t: float) -> 'TZ':
        return TZ(t/2, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(1/2, *self.qubits).run(ket)

# end class S


class T(Gate):
    r"""
    A 1-qubit T (pi/8) gate, equivalent to ``PHASE(pi/4)``. The forth root
    of the Z gate (up to global phase).

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}
    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1.0, 0.0], [0.0, bk.ccast(bk.cis(pi / 4.0))]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'T_H':
        return T_H(*self.qubits)

    def __pow__(self, t: float) -> 'TZ':
        return TZ(t/4, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(1/4, *self.qubits).run(ket)

# end class T


class PHASE(Gate):
    r"""
    A 1-qubit parametric phase shift gate.
    Equivalent to RZ upto a global phase.

    .. math::
        \text{PHASE}(\theta) \equiv \begin{pmatrix}
         1 & 0 \\ 0 & e^{i \theta} \end{pmatrix}
    """
    diagonal = True

    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0.0], [0.0, bk.cis(ctheta)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'PHASE':
        return self ** -1

    def __pow__(self, t: float) -> 'PHASE':
        theta = self.params['theta']
        return PHASE(theta * t, *self.qubits)

    # TODO: CHECKME
    def run(self, ket: State) -> State:
        theta, = self.params.values()
        return TZ(theta/pi, *self.qubits).run(ket)

# end class PHASE


class RX(Gate):
    r"""A 1-qubit Pauli-X parametric rotation gate.

    .. math::
        R_X(\theta) =   \begin{pmatrix}
                            \cos(\frac{\theta}{2}) & -i \sin(\theta/2) \\
                            -i \sin(\theta/2) & \cos(\theta/2)
                        \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[bk.cos(ctheta / 2), -1.0j * bk.sin(ctheta / 2)],
                   [-1.0j * bk.sin(ctheta / 2), bk.cos(ctheta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RX':
        return self ** -1

    def __pow__(self, t: float) -> 'RX':
        theta = self.params['theta']
        return RX(theta * t, *self.qubits)

# end class RX


class RY(Gate):
    r"""A 1-qubit Pauli-Y parametric rotation gate

    .. math::
        R_Y(\theta) \equiv \begin{pmatrix}
        \cos(\theta / 2) & -\sin(\theta / 2)
        \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[bk.cos(ctheta / 2.0), -bk.sin(ctheta / 2.0)],
                   [bk.sin(ctheta / 2.0), bk.cos(ctheta / 2.0)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RY':
        return self ** -1

    def __pow__(self, t: float) -> 'RY':
        theta = self.params['theta']
        return RY(theta * t, *self.qubits)

# end class RY


class RZ(Gate):
    r"""A 1-qubit Pauli-X parametric rotation gate

    .. math::
        R_Z(\theta)\equiv   \begin{pmatrix}
                                \cos(\theta/2) - i \sin(\theta/2) & 0 \\
                                0 & \cos(\theta/2) + i \sin(\theta/2)
                            \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    diagonal = True

    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(theta=theta), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta = self.params['theta']
        ctheta = bk.ccast(theta)
        unitary = [[bk.exp(-ctheta * 0.5j), 0],
                   [0, bk.exp(ctheta * 0.5j)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RZ':
        return self ** -1

    def __pow__(self, t: float) -> 'RZ':
        theta = self.params['theta']
        return RZ(theta * t, *self.qubits)

    # TODO: CHECKME
    def run(self, ket: State) -> State:
        theta, = self.params.values()
        return TZ(theta/pi, *self.qubits).run(ket)

# end class RZ


# Other 1-qubit gates

class S_H(Gate):
    r"""
    The inverse of the 1-qubit phase S gate, equivalent to ``PHASE(-pi/2)``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}

    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = np.asarray([[1.0, 0.0], [0.0, -1.0j]])
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'S':
        return S(*self.qubits)

    def __pow__(self, t: float) -> 'TZ':
        return TZ(-t/2, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(-1/2, *self.qubits).run(ket)

# end class S_H


class T_H(Gate):
    r"""
    The inverse (complex conjugate) of the 1-qubit T (pi/8) gate, equivalent
    to ``PHASE(-pi/4)``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{-i \pi / 4} \end{pmatrix}
    """
    diagonal = True

    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        unitary = [[1.0, 0.0], [0.0, bk.ccast(bk.cis(-pi / 4.0))]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'T':
        return T(*self.qubits)

    def __pow__(self, t: float) -> 'TZ':
        return TZ(-t/4, *self.qubits)

    def run(self, ket: State) -> State:
        return TZ(-1/4, *self.qubits).run(ket)

# end class T_H


class RN(Gate):
    r"""A 1-qubit rotation of angle theta about axis (nx, ny, nz)

    .. math::
        R_n(\theta) = \cos \frac{theta}{2} I - i \sin\frac{theta}{2}
            (n_x X+ n_y Y + n_z Z)

    Args:
        theta: Angle of rotation on Block sphere
        (nx, ny, nz): A three-dimensional real unit vector
    """

    def __init__(self,
                 theta: float,
                 nx: float,
                 ny: float,
                 nz: float,
                 q0: Qubit = 0) -> None:
        params = dict(theta=theta, nx=nx, ny=ny, nz=nz)
        super().__init__(params=params, qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        theta, nx, ny, nz = self.params.values()
        ctheta = bk.ccast(theta)
        cost = bk.cos(ctheta / 2)
        sint = bk.sin(ctheta / 2)
        unitary = [[cost - 1j * sint * nz, -1j * sint * nx - sint * ny],
                   [-1j * sint * nx + sint * ny, cost + 1j * sint * nz]]

        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'RN':
        return self ** -1

    def __pow__(self, t: float) -> 'RN':
        theta, nx, ny, nz = self.params.values()
        return RN(t * theta, nx, ny, nz, *self.qubits)


class TX(Gate):
    r"""Powers of the 1-qubit Pauli-X gate.

    .. math::
        TX(t) = X^t = e^{i \pi t/2} R_X(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """
    def __init__(self, t: float, q0: Qubit = 0) -> None:
        t = t % 2
        super().__init__(params=dict(t=t), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t = self.params['t']
        ctheta = bk.ccast(pi * t)
        phase = bk.cis(0.5 * ctheta)
        unitary = [[phase * bk.cos(ctheta / 2),
                    phase * -1.0j * bk.sin(ctheta / 2)],
                   [phase * -1.0j * bk.sin(ctheta / 2),
                    phase * bk.cos(ctheta / 2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TX':
        return self ** -1

    def __pow__(self, t: float) -> 'TX':
        t = self.params['t'] * t
        return TX(t, *self.qubits)


class TY(Gate):
    r"""Powers of the 1-qubit Pauli-Y gate.

    The pseudo-Hadamard gate is TY(3/2), and its inverse is TY(1/2).

    .. math::
        TY(t) = Y^t = e^{i \pi t/2} R_Y(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere

    """
    def __init__(self, t: float, q0: Qubit = 0) -> None:
        # t = t % 2
        super().__init__(params=dict(t=t), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t = self.params['t']
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.cos(ctheta / 2.0),
                    phase * -bk.sin(ctheta / 2.0)],
                   [phase * bk.sin(ctheta / 2.0),
                    phase * bk.cos(ctheta / 2.0)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TY':
        return self ** -1

    def __pow__(self, t: float) -> 'TY':
        t = self.params['t'] * t
        return TY(t, *self.qubits)


class TZ(Gate):
    r"""Powers of the 1-qubit Pauli-Z gate.

    .. math::
        TZ(t) = Z^t = e^{i \pi t/2} R_Z(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """
    diagonal = True

    def __init__(self, t: float, q0: Qubit = 0) -> None:
        # t = t % 2
        super().__init__(params=dict(t=t), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t = self.params['t']
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.exp(-ctheta * 0.5j), 0],
                   [0, phase * bk.exp(ctheta * 0.5j)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TZ':
        t = - self.params['t']
        return TZ(t, *self.qubits)

    def __pow__(self, t: float) -> 'TZ':
        t = self.params['t'] * t
        return TZ(t, *self.qubits)

    def run(self, ket: State) -> State:
        if bk.BACKEND == 'numpy':
            t = self.params['t']
            axes = ket.qubit_indices(self.qubits)
            s1 = multi_slice(axes, [1])
            tensor = ket.tensor.copy()
            tensor[s1] *= bk.exp(+1.0j*pi*t)
            return State(tensor, ket.qubits, ket.memory)

        return super().run(ket)  # pragma: no cover


class TH(Gate):
    r"""
    Powers of the 1-qubit Hadamard gate.

    .. math::
        TH(t) = H^t = e^{i \pi t/2}
        \begin{pmatrix}
            \cos(\tfrac{t}{2}) + \tfrac{i}{\sqrt{2}}\sin(\tfrac{t}{2})) &
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) \\
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) &
            \cos(\tfrac{t}{2}) -\tfrac{i}{\sqrt{2}} \sin(\frac{t}{2})
        \end{pmatrix}
    """
    def __init__(self, t: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(t=t), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t = self.params['t']
        theta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * theta)
        unitary = [[phase * bk.cos(theta / 2)
                    - (phase * 1.0j * bk.sin(theta / 2)) / sqrt(2),
                    -(phase * 1.0j * bk.sin(theta / 2)) / sqrt(2)],
                   [-(phase * 1.0j * bk.sin(theta / 2)) / sqrt(2),
                    phase * bk.cos(theta / 2)
                    + (phase * 1.0j * bk.sin(theta / 2)) / sqrt(2)]]
        return bk.astensorproduct(unitary)

    @property
    def H(self) -> 'TH':
        return self ** -1

    def __pow__(self, t: float) -> 'TH':
        t = self.params['t'] * t
        return TH(t, *self.qubits)


# FIXME: Replace with euler_circuit?
class ZYZ(Gate):
    r"""A Z-Y-Z decomposition of one-qubit rotations in the Bloch sphere

    The ZYZ decomposition of one-qubit rotations is

    .. math::
        \text{ZYZ}(t_0, t_1, t_2)
            = Z^{t_2} Y^{t_1} Z^{t_0}

    This is the unitary group on a 2-dimensional complex vector space, SU(2).

    Ref: See Barenco et al (1995) section 4 (Warning: gates are defined as
    conjugate of what we now use?), or Eq 4.11 of Nielsen and Chuang.

    Args:
        t0: Parameter of first parametric Z gate.
            Number of half turns on Block sphere.
        t1: Parameter of parametric Y gate.
        t2: Parameter of second parametric Z gate.
    """
    def __init__(self, t0: float, t1: float,
                 t2: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(t0=t0, t1=t1, t2=t2), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        t0, t1, t2 = self.params.values()
        ct0 = bk.ccast(pi * t0)
        ct1 = bk.ccast(pi * t1)
        ct2 = bk.ccast(pi * t2)
        ct3 = 0

        unitary = [[bk.cis(ct3 - 0.5 * ct2 - 0.5 * ct0) * bk.cos(0.5 * ct1),
                    -bk.cis(ct3 - 0.5 * ct2 + 0.5 * ct0) * bk.sin(0.5 * ct1)],
                   [bk.cis(ct3 + 0.5 * ct2 - 0.5 * ct0) * bk.sin(0.5 * ct1),
                    bk.cis(ct3 + 0.5 * ct2 + 0.5 * ct0) * bk.cos(0.5 * ct1)]]

        return bk.astensorproduct(unitary)

    @property
    def H(self) -> Gate:
        t0, t1, t2 = self.params.values()
        return ZYZ(-t2, -t1, -t0, *self.qubits)


# TODO: Rename to SX (sqrt-X)? Add SY gate, (SZ is S)?
class V(Gate):
    r"""
    Principal square root of the X gate, X-PLUS-90 gate.
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return TX(0.5).tensor

    @property
    def H(self) -> 'V_H':
        return V_H(*self.qubits)

    def __pow__(self, t: float) -> 'TX':
        return TX(0.5*t, *self.qubits)


class V_H(Gate):
    r"""
    Complex conjugate of the V gate, X-MINUS-90 gate.
    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        return TX(-0.5).tensor

    @property
    def H(self) -> 'V':
        return V(*self.qubits)

    def __pow__(self, t: float) -> 'TX':
        return TX(-0.5*t, *self.qubits)


# Kudos: W (Phased X), and TW (PhasedXPow) gates adapted from Cirq

class W(Gate):
    r""" A phased X gate, equivalent to the circuit
    ───Z^-p───X───Z^p───
    """
    def __init__(self, p: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(p=p), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        p = self.params['p']
        gate = TZ(p) @ X() @ TZ(-p)
        return gate.tensor

    # TESTME
    @property
    def H(self) -> 'W':
        return self

    def __pow__(self, t: float) -> 'TW':
        p = self.params['p']
        return TW(p, t, *self.qubits)


class TW(Gate):
    """A phased X gate raise to a power.

    Equivalent to the circuit ───Z^-p───X^t───Z^p───

    """
    def __init__(self, p: float, t: float, q0: Qubit = 0) -> None:
        super().__init__(params=dict(p=p, t=t), qubits=[q0])

    @cached_property
    def tensor(self) -> bk.BKTensor:
        p, t = self.params.values()
        gate = TZ(p) @ TX(t) @ TZ(-p)
        return gate.tensor

    @property
    def H(self) -> 'TW':
        return self ** -1

    def __pow__(self, t: float) -> 'TW':
        p, s = self.params.values()
        return TW(p, s * t, *self.qubits)

# end class TW


cliffords = (
    I(),

    RN(0.5 * pi, 1, 0, 0),
    RN(0.5 * pi, 0, 1, 0),
    RN(0.5 * pi, 0, 0, 1),
    RN(pi, 1, 0, 0),
    RN(pi, 0, 1, 0),
    RN(pi, 0, 0, 1),
    RN(-0.5 * pi, 1, 0, 0),
    RN(-0.5 * pi, 0, 1, 0),
    RN(-0.5 * pi, 0, 0, 1),

    RN(pi, 1/sqrt(2), 1/sqrt(2), 0),
    RN(pi, 1/sqrt(2), 0, 1/sqrt(2)),
    RN(pi, 0, 1/sqrt(2), 1/sqrt(2)),
    RN(pi, -1/sqrt(2), 1/sqrt(2), 0),
    RN(pi, 1/sqrt(2), 0, -1/sqrt(2)),
    RN(pi, 0, -1/sqrt(2), 1/sqrt(2)),

    RN(+2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(-2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(+2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(-2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    RN(+2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    RN(-2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    RN(+2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3)),
    RN(-2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3)),
    )
"""
List of all 24 1-qubit Clifford gates. The first gate is the identity.
The rest are given as instances of the generic rotation gate RN
"""
