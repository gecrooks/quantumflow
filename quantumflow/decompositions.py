# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Gate Decompositions

.. contents:: :local:
.. currentmodule:: quantumflow


One-qubit gate decompositions
#############################

.. autofunction:: bloch_decomposition
.. autofunction:: zyz_decomposition
.. autofunction:: euler_decomposition

Two-qubit gate decompositions
#############################

.. autofunction:: kronecker_decomposition
.. autofunction:: canonical_decomposition
.. autofunction:: canonical_coords
.. autofunction:: cnot_decomposition
.. autofunction:: b_decomposition
"""


import itertools
from typing import List, Sequence, Tuple, cast

import numpy as np

from .circuits import Circuit, euler_circuit
from .config import ATOL
from .info import gates_close
from .ops import Gate, Unitary
from .stdgates import S_H, B, Can, I, Rn, S, V, X, Y, YPow, Z, ZPow
from .translate import translate_can_to_cnot

__all__ = [
    "bloch_decomposition",
    "zyz_decomposition",
    "euler_decomposition",
    "kronecker_decomposition",
    "canonical_decomposition",
    "canonical_coords",
    "cnot_decomposition",
    "b_decomposition",
    "convert_can_to_weyl",
]


# TODO: Optionally include phase
def bloch_decomposition(gate: Gate) -> Circuit:
    """
    Converts a 1-qubit gate into a RN gate, a 1-qubit rotation of angle theta
    about axis (nx, ny, nz) in the Bloch sphere.

    Returns:
        A Circuit containing a single RN gate
    """
    if gate.qubit_nb != 1:
        raise ValueError("Expected 1-qubit gate")

    U = gate.asoperator()
    U /= np.linalg.det(U) ** (1 / 2)

    nx = -U[0, 1].imag
    ny = -U[0, 1].real
    nz = -U[0, 0].imag
    N = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    if N == 0:  # Identity
        nx, ny, nz = 1, 1, 1
    else:
        nx /= N
        ny /= N
        nz /= N
    sin_halftheta = N
    cos_halftheta = U[0, 0].real
    theta = 2 * np.arctan2(sin_halftheta, cos_halftheta)

    # We return a Circuit (rather than just a gate) to keep the
    # interface of decomposition routines uniform.
    return Circuit([Rn(theta, nx, ny, nz, *gate.qubits)])


# TODO: Optionally include phase?
def zyz_decomposition(gate: Gate) -> Circuit:
    """
    Returns the Euler Z-Y-Z decomposition of a local 1-qubit gate.
    """
    if gate.qubit_nb != 1:
        raise ValueError("Expected 1-qubit gate")

    (q,) = gate.qubits

    U = gate.su().asoperator()  # SU(2)

    if abs(U[0, 0]) > abs(U[1, 0]):
        theta1 = 2 * np.arccos(min(abs(U[0, 0]), 1))
    else:
        theta1 = 2 * np.arcsin(min(abs(U[1, 0]), 1))

    cos_halftheta1 = np.cos(theta1 / 2)
    if not np.isclose(cos_halftheta1, 0.0):
        phase = U[1, 1] / cos_halftheta1
        theta0_plus_theta2 = 2 * np.arctan2(np.imag(phase), np.real(phase))
    else:
        theta0_plus_theta2 = 0.0

    sin_halftheta1 = np.sin(theta1 / 2)
    if not np.isclose(sin_halftheta1, 0.0):
        phase = U[1, 0] / sin_halftheta1
        theta0_sub_theta2 = 2 * np.arctan2(np.imag(phase), np.real(phase))
    else:
        theta0_sub_theta2 = 0.0

    theta0 = (theta0_plus_theta2 + theta0_sub_theta2) / 2
    theta2 = (theta0_plus_theta2 - theta0_sub_theta2) / 2

    t0 = theta0 / np.pi
    t1 = theta1 / np.pi
    t2 = theta2 / np.pi

    if np.isclose(t1, 0.0):
        t2 = t0 + t2
        t1 = 0.0
        t0 = 0.0

    circ1 = Circuit()
    circ1 += ZPow(t2, q)
    circ1 += YPow(t1, q)
    circ1 += ZPow(t0, q)

    return circ1


def euler_decomposition(gate: Gate, euler: str = "ZYZ") -> Circuit:
    """
    Returns an Euler angle decomposition of a local 1-qubit gate .

    The 'euler' argument can be used to specify any of the 6 Euler
    decompositions: 'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ' (Default)
    """
    if euler == "ZYZ":
        return zyz_decomposition(gate)

    euler_trans = {
        "XYX": Y(0) ** 0.5,
        "XZX": Rn(+2 * np.pi / 3, np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3), 0),
        "YXY": Rn(-2 * np.pi / 3, np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3), 0),
        "YZY": Rn(np.pi, 0, np.sqrt(1 / 2), np.sqrt(1 / 2), 0),
        "ZXZ": S_H(0),
        "ZYZ": I(0),
    }

    q0 = gate.qubits[0]
    trans = euler_trans[euler].on(q0)
    gate = Circuit([trans, gate, trans.H]).asgate()
    zyz = zyz_decomposition(gate)
    params = [elem.param("t") for elem in zyz]  # type: ignore
    return euler_circuit(params[0], params[1], params[2], q0, euler)


# FIXME: Return 2 sub circuits?
def kronecker_decomposition(gate: Gate, euler: str = "ZYZ") -> Circuit:
    """
    Decompose a 2-qubit unitary composed of two 1-qubit local gates.

    Uses the "Nearest Kronecker Product" algorithm. Will give erratic
    results if the gate is not the direct product of two 1-qubit gates.
    """
    # An alternative approach would be to take partial traces, but
    # this approach appears to be more robust.

    if gate.qubit_nb != 2:
        raise ValueError("Expected 2-qubit gate")

    U = gate.asoperator()
    rank = 2 ** gate.qubit_nb
    U /= np.linalg.det(U) ** (1 / rank)

    R = U.reshape(2, 2, 2, 2)
    R = R.transpose(0, 2, 1, 3)
    R = R.reshape(4, 4)

    u, s, vh = np.linalg.svd(R)
    A = np.sqrt(s[0]) * u[:, 0].reshape(2, 2)
    B = np.sqrt(s[0]) * vh[0, :].reshape(2, 2)
    # print(s)
    # A = (u[:, 0]).reshape(2, 2)
    # B = (vh[0, :]).reshape(2, 2)

    q0, q1 = gate.qubits
    g0 = Unitary(A, [q0])
    g1 = Unitary(B, [q1])

    if not gates_close(gate, Circuit([g0, g1]).asgate()):
        raise ValueError("Gate cannot be decomposed into two 1-qubit gates")

    circ = Circuit()
    circ += euler_decomposition(g0, euler)
    circ += euler_decomposition(g1, euler)

    assert gates_close(gate, circ.asgate())  # Sanity check

    return circ


def canonical_coords(gate: Gate) -> Sequence[float]:
    """Returns the canonical coordinates of a 2-qubit gate"""
    circ = canonical_decomposition(gate)
    gate = circ[1]  # type: ignore
    params = [gate.param(key) for key in ("tx", "ty", "tz")]
    return params  # type: ignore


# FIXME: Does not return a circuit of 5 gates.
# Instead return locals_before, canonical, locals_after
def canonical_decomposition(gate: Gate, euler: str = "ZYZ") -> Circuit:
    """Decompose a 2-qubit gate by removing local 1-qubit gates to leave
    the non-local canonical two-qubit gate. [1]_ [2]_ [3]_ [4]_

    Returns: A Circuit of 3 operations: a circuit of initial 1-qubit gates;
    a canonical gate, with coordinates in the Weyl chamber; and a final
    circuit contains the final 1-qubits gates.

    The canonical coordinates can be found in circ.elements[2].params

    More or less follows the algorithm outlined in [2]_.

    .. [1] A geometric theory of non-local two-qubit operations, J. Zhang,
        J. Vala, K. B. Whaley, S. Sastry quant-ph/0291120
    .. [2] An analytical decomposition protocol for optimal implementation of
        two-qubit entangling gates. M. Blaauboer, R.L. de Visser,
        cond-mat/0609750
    .. [3] Metric structure of two-qubit gates, perfect entangles and quantum
        control, P. Watts, M. O'Conner, J. Vala, Entropy (2013)
    .. [4] Constructive Quantum Shannon Decomposition from Cartan Involutions
        B. Drury, P. Love, arXiv:0806.4015
    """

    # Implementation note: The canonical decomposition is easy. Constraining
    # canonical coordinates to the Weyl chamber is easy. But doing the
    # canonical decomposition with the canonical gate in the Weyl chamber
    # proved to be surprisingly tricky.

    # Unitary transform to Magic Basis of Bell states
    Q = np.asarray(
        [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]
    ) / np.sqrt(2)
    Q_H = Q.conj().T

    if gate.qubit_nb != 2:
        raise ValueError("Expected 2-qubit gate")

    q0, q1 = gate.qubits

    U = gate.asoperator()
    rank = 2 ** gate.qubit_nb
    U /= np.linalg.det(U) ** (1 / rank)  # U is in SU(4) so det U = 1

    U_mb = Q_H @ U @ Q  # Transform gate to Magic Basis [1, (eq. 17, 18)]
    M = U_mb.transpose() @ U_mb  # Construct M matrix [1, (eq. 22)]

    # Diagonalize symmetric complex matrix
    eigvals, eigvecs = _eig_complex_symmetric(M)

    lambdas = np.sqrt(eigvals)  # Eigenvalues of F
    # Lambdas only fixed up to a sign. So make sure det F = 1 as it should
    det_F = np.prod(lambdas)
    if det_F.real < 0:
        lambdas[0] *= -1

    coords, signs, perm = _constrain_to_weyl(lambdas)

    # Construct local and canonical gates in magic basis
    lambdas = (lambdas * signs)[perm]
    O2 = (np.diag(signs) @ eigvecs.transpose())[perm]
    F = np.diag(lambdas)
    O1 = U_mb @ O2.transpose() @ F.conj()

    # Sanity check: Make sure O1 and O2 are orthogonal
    assert np.allclose(np.eye(4), O2.transpose() @ O2)  # Sanity check
    assert np.allclose(np.eye(4), O1.transpose() @ O1)  # Sanity check

    # Sometimes O1 & O2 end up with det = -1, instead of +1 as they should.
    # We can commute a diagonal matrix through F to fix this up.
    neg = np.diag([-1, 1, 1, 1])
    if np.linalg.det(O2).real < 0:
        O2 = neg @ O2
        O1 = O1 @ neg

    # Transform gates back from magic basis
    K1 = Q @ O1 @ Q_H
    A = Q @ F @ Q_H
    K2 = Q @ O2 @ Q_H

    assert gates_close(Unitary(U, [0, 1]), Unitary(K1 @ A @ K2, [0, 1]))  # Sanity check
    canon = Can(coords[0], coords[1], coords[2], q0, q1)

    # Sanity check
    assert gates_close(Unitary(A, gate.qubits), canon, atol=1e-4)

    # Decompose local gates into the two component 1-qubit gates
    gateK1 = Unitary(K1, gate.qubits)
    circK1 = kronecker_decomposition(gateK1, euler)
    assert gates_close(gateK1, circK1.asgate())  # Sanity check

    gateK2 = Unitary(K2, gate.qubits)
    circK2 = kronecker_decomposition(gateK2, euler)
    assert gates_close(gateK2, circK2.asgate())  # Sanity check

    # Build and return circuit
    # circ = Circuit()
    # circ += circK2
    # circ += canon
    # circ += circK1

    circ = Circuit([circK2, canon, circK1])

    return circ


def _eig_complex_symmetric(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize a complex symmetric  matrix. The eigenvalues are
    complex, and the eigenvectors form an orthogonal matrix.

    Returns:
        eigenvalues, eigenvectors
    """
    if not np.allclose(M, M.transpose()):
        raise np.linalg.LinAlgError("Not a symmetric matrix")

    # The matrix of eigenvectors should be orthogonal.
    # But the standard 'eig' method will fail to return an orthogonal
    # eigenvector matrix when the eigenvalues are degenerate. However,
    # both the real and
    # imaginary part of M must be symmetric with the same orthogonal
    # matrix of eigenvectors. But either the real or imaginary part could
    # vanish. So we use a randomized algorithm where we diagonalize a
    # random linear combination of real and imaginary parts to find the
    # eigenvectors, taking advantage of the 'eigh' subroutine for
    # diagonalizing symmetric matrices.
    # This can fail if we're very unlucky with our random coefficient, so we
    # give the algorithm a few chances to succeed.

    # Empirically, never seems to fail on randomly sampled complex
    # symmetric 4x4 matrices.
    # If failure rate is less than 1 in a million, then 16 rounds
    # will have overall failure rate less than 1 in a googol.
    # However, cannot (yet) guarantee that there aren't special cases
    # which have much higher failure rates.

    # GEC 2018

    max_attempts = 16
    for _ in range(max_attempts):
        c = np.random.uniform(0, 1)
        matrix = c * M.real + (1 - c) * M.imag
        _, eigvecs = np.linalg.eigh(matrix)
        eigvecs = np.array(eigvecs, dtype=complex)
        eigvals = np.diag(eigvecs.transpose() @ M @ eigvecs)

        # Finish if we got a correct answer.
        reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.transpose()
        if np.allclose(M, reconstructed):
            return eigvals, eigvecs

    # Should never happen. Hopefully.
    raise np.linalg.LinAlgError(
        "Cannot diagonalize complex symmetric matrix."
    )  # pragma: no cover


def _lambdas_to_coords(lambdas: Sequence[float]) -> np.ndarray:
    # [2, eq.11], but using [1]s coordinates.
    l1, l2, _, l4 = lambdas
    c1 = np.real(1j * np.log(l1 * l2))
    c2 = np.real(1j * np.log(l2 * l4))
    c3 = np.real(1j * np.log(l1 * l4))
    coords = np.asarray((c1, c2, c3)) / np.pi

    coords[np.abs(coords - 1) < ATOL] = -1
    if all(coords < 0):
        coords += 1

    # If we're close to the boundary, floating point errors can conspire
    # to make it seem that we're never on the inside
    # Fix: If near boundary, reset to boundary

    # Left
    if np.abs(coords[0] - coords[1]) < ATOL:
        coords[1] = coords[0]

    # Front
    if np.abs(coords[1] - coords[2]) < ATOL:
        coords[2] = coords[1]

    # Right
    if np.abs(coords[0] - coords[1] - 1 / 2) < ATOL:
        coords[1] = coords[0] - 1 / 2

    # Base
    coords[np.abs(coords) < ATOL] = 0

    return coords


def _constrain_to_weyl(
    lambdas: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    for permutation in itertools.permutations(range(4)):
        for signs in ([1, 1, 1, 1], [1, 1, -1, -1], [-1, 1, -1, 1], [1, -1, -1, 1]):
            signed_lambdas = lambdas * np.asarray(signs)
            perm = list(permutation)
            lambdas_perm = signed_lambdas[perm]

            coords = _lambdas_to_coords(lambdas_perm)

            if _in_weyl(*coords):
                return coords, np.asarray(signs), np.asarray(perm)

    # Should never get here
    assert False  # pragma: no cover


def _in_weyl(tx: float, ty: float, tz: float) -> bool:
    # Note 'tz>0' in second term. This takes care of symmetry across base
    # when tz==0
    return (1 / 2 >= tx >= ty >= tz >= 0) or (1 / 2 >= (1 - tx) >= ty >= tz > 0)


def cnot_decomposition(gate: Gate) -> Circuit:
    """Decompose any 2-qubit gate into a circuit of three CNot gates.

    Ref:
        Optimal Quantum Circuits for General Two-Qubit Gates, Vatan & Williams
        (2004) (quant-ph/0308006) Fig. 6
    """

    gate_dek = canonical_decomposition(gate)
    gate_before = cast(Circuit, gate_dek[0])
    gate_can = cast(Can, gate_dek[1])
    gate_after = cast(Circuit, gate_dek[2])

    cnot_circ = Circuit(translate_can_to_cnot(gate_can))
    assert gates_close(gate_can, cnot_circ.asgate())
    circ = gate_before + cnot_circ + gate_after

    return circ


def b_decomposition(gate: Gate) -> Circuit:
    """Decompose any 2-qubit gate into a sandwich of two B gates.

    Refs:
        Minimum construction of two-qubit quantum operations
        https://arxiv.org/abs/quant-ph/0312193
    """
    # Implementation Note
    # The analytic construction of a given canonical gate, up to local gates, is
    # given in quant-ph/0312193. But we don't know how to get the local
    # gates analytically. (Although, it looks like it could be done...)
    # Instead we follow the tactic used in cirq. We take a canonical
    # decomposition of the original gate, construct a B gate sandwich with the
    # same canonical coordinates, then canonically decompose the B gate
    # sandwich to figure out the appropriate local gates.
    # Kudos: Cirq, @Strilanc
    # Ref: https://github.com/quantumlib/Cirq/pull/2574/files

    q0, q1 = gate.qubits

    gate_dek = canonical_decomposition(gate)
    gate_before = cast(Circuit, gate_dek[0])
    gate_can = gate_dek[1]
    gate_after = cast(Circuit, gate_dek[2])

    tx, ty, tz = gate_can.params
    c0 = np.sin(0.5 * ty * np.pi) ** 2 * np.cos(0.5 * tz * np.pi) ** 2
    c0 = max(0.0, min(0.5, c0))
    c1 = (np.cos(ty * np.pi) * np.cos(tz * np.pi)) / (1 - 2 * c0)
    c1 = max(0.0, min(1.0, c1))

    sy = (1 / np.pi) * np.arccos(1 - 4 * c0)
    sz = -(1 / np.pi) * np.arcsin(np.sqrt(c1))

    bcirc = Circuit(
        [
            B(q0, q1),
            YPow(-tx, q0),
            ZPow(sz, q1),
            YPow(sy, q1),
            ZPow(sz, q1),
            B(q0, q1),
        ]
    )

    bcirc_dek = canonical_decomposition(bcirc.asgate())
    bcirc_before = cast(Circuit, bcirc_dek[0])
    bcirc_after = cast(Circuit, bcirc_dek[2])

    circ = gate_before + bcirc_before.H + bcirc + bcirc_after.H + gate_after

    return circ


# TODO: Should be able to replace _lambdas_to_coords with this method,
# which should be faster
# TODO: Better name?
# DOCME
def convert_can_to_weyl(gate: Can, euler: str = "ZYZ") -> Circuit:
    """Move a Canonical gate with arbitrary coordinates to a canonical gate
    in the Weyl chamber, plus local 1-qubit gates.

    Args:
        gate:  A canonical (Can) gate
        euler: The Euler decomposition to use when simplifying the local gates
                If None, do not simplify.
    Returns:
        A circuit of 5 elements --
            circuit of gates acting on the first qubit
            circuit of gates acting on the second qubit
            canonical gate in the Weyl chamber
            circuit of gates acting on the first qubit
            circuit of gates acting on the second qubit
    """

    tx, ty, tz = gate.params
    q0, q1 = gate.qubits

    # Local gates to put before the canonical Weyl gate
    before_q0: List[Gate] = [I(q0)]
    before_q1: List[Gate] = [I(q1)]

    # Local gates that come after the canonical Weyl gate, in reverse
    # (complex conjugate) order.
    after_q0_H: List[Gate] = [I(q0)]
    after_q1_H: List[Gate] = [I(q1)]

    def tx_minus_1() -> None:
        before_q0.append(Y(q0))
        before_q1.append(Y(q1))
        after_q0_H.append(Z(q0))
        after_q1_H.append(Z(q1))

    def ty_minus_1() -> None:
        before_q0.append(X(q0))
        before_q1.append(X(q1))
        after_q0_H.append(Z(q0))
        after_q1_H.append(Z(q1))

    def tz_minus_1() -> None:
        before_q0.append(X(q0))
        before_q1.append(X(q1))
        after_q0_H.append(Y(q0))
        after_q1_H.append(Y(q1))

    def flip_tx_ty() -> None:
        before_q0.append(Z(q0))
        after_q0_H.append(Z(q0))

    def flip_tx_tz() -> None:
        before_q0.append(Y(q0))
        after_q0_H.append(Y(q0))

    def flip_ty_tz() -> None:
        before_q0.append(X(q0))
        after_q0_H.append(X(q0))

    def swap_tx_ty() -> None:
        before_q0.append(S(q0))
        before_q1.append(S(q1))
        after_q0_H.append(S(q0))
        after_q1_H.append(S(q1))

    def swap_tx_tz() -> None:
        before_q0.append(Y(q0) ** 0.5)
        before_q1.append(Y(q1) ** 0.5)
        after_q0_H.append(Y(q0) ** 0.5)
        after_q1_H.append(Y(q1) ** 0.5)

    def swap_ty_tz() -> None:
        before_q0.append(V(q0))
        before_q1.append(V(q1))
        after_q0_H.append(V(q0))
        after_q1_H.append(V(q1))

    tx = tx % 2
    ty = ty % 2
    tz = tz % 2

    if tx > 1.0:
        tx -= 1
        tx_minus_1()

    if ty > 1.0:
        ty -= 1
        ty_minus_1()

    if tz > 1.0:
        tz -= 1
        tz_minus_1()

    if tx > 0.5 and ty > 0.5:
        tx = 1 - tx
        ty = 1 - ty
        tx_minus_1()
        ty_minus_1()
        flip_tx_ty()

    if tx > 0.5 and tz > 0.5:
        tx = 1 - tx
        tz = 1 - tz
        tx_minus_1()
        tz_minus_1()
        flip_tx_tz()

    if ty > 0.5 and tz > 0.5:
        ty = 1 - ty
        tz = 1 - tz
        ty_minus_1()
        tz_minus_1()
        flip_ty_tz()

    if ty > tx:
        tx, ty = ty, tx
        swap_tx_ty()

    if tz > tx:
        tx, tz = tz, tx
        swap_tx_tz()

    if tz > ty:
        tz, ty = ty, tz
        swap_ty_tz()

    if tx > 0.5 and (1 - tx) < ty:
        tx, ty = ty, tx
        swap_tx_ty()
        tx = 1 - tx
        ty = 1 - ty
        tx_minus_1()
        ty_minus_1()
        flip_tx_ty()

    if ty > tx:  # pragma: no cover   # Never happens?
        tx, ty = ty, tx
        swap_tx_ty()

    if tz > tx:  # pragma: no cover   # Never happens?
        tx, tz = tz, tx
        swap_tx_tz()

    if tz > ty:
        tz, ty = ty, tz
        swap_ty_tz()

    # Insanity check
    assert _in_weyl(tx, ty, tz)

    circ_before_q0 = Circuit(before_q0)
    circ_before_q1 = Circuit(before_q1)
    circ_after_q0_H = Circuit(after_q0_H)
    circ_after_q1_H = Circuit(after_q1_H)
    if euler is not None:
        circ_before_q0 = euler_decomposition(circ_before_q0.asgate(), euler=euler)
        circ_before_q1 = euler_decomposition(circ_before_q1.asgate(), euler=euler)
        circ_after_q0_H = euler_decomposition(circ_after_q0_H.asgate(), euler=euler)
        circ_after_q1_H = euler_decomposition(circ_after_q1_H.asgate(), euler=euler)

    circ = Circuit()

    circ += circ_before_q0
    circ += circ_before_q1
    circ += Can(tx, ty, tz, q0, q1)
    circ += circ_after_q0_H.H
    circ += circ_after_q1_H.H

    return circ


# fin
