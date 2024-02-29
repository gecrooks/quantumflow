# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# 2-qubit gates

from typing import Iterator, Union

from .. import var
from ..stdgates import (
    CH,
    CS,
    CT,
    CU3,
    CV,
    CV_H,
    CY,
    CZ,
    ECP,
    S_H,
    T_H,
    U3,
    V_H,
    XX,
    XY,
    YY,
    ZZ,
    A,
    B,
    Barenco,
    Can,
    CNot,
    CNotPow,
    CPhase,
    CPhase00,
    CPhase01,
    CPhase10,
    CrossResonance,
    CRx,
    CRy,
    CRz,
    CYPow,
    CZPow,
    Exch,
    FSim,
    FSwap,
    FSwapPow,
    Givens,
    H,
    ISwap,
    PhaseShift,
    PSwap,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    S,
    SqrtISwap,
    SqrtISwap_H,
    SqrtSwap,
    SqrtSwap_H,
    Swap,
    Sycamore,
    T,
    V,
    W,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
)
from .translations import register_translation


@register_translation
def translate_b_to_can(gate: B) -> Iterator[Union[Can, Y, Z]]:
    """Translate B gate to Canonical gate"""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield Y(q1)
    yield Can(1 / 2, 1 / 4, 0, q0, q1)
    yield Y(q0)
    yield Z(q1)


@register_translation
def translate_barenco_to_xx(gate: Barenco) -> Iterator[Union[XX, YPow, ZPow]]:
    """Translate a Barenco gate to XX plus local gates"""
    phi, alpha, theta = gate.params

    ct = theta / var.PI
    ca = alpha / var.PI
    cp = phi / var.PI

    q0, q1 = gate.qubits

    yield ZPow(-1 / 2 + cp, q0)
    yield YPow(1 / 2, q0)
    yield ZPow(-1, q0)
    yield ZPow(1 / 2 - cp, q1)
    yield YPow(1 / 2, q1)
    yield ZPow(3 / 2, q1)
    yield XX(ct, q0, q1)
    yield YPow(1 / 2, q0)
    yield ZPow(3 / 2 + ca - cp, q0)
    yield ZPow(-1 / 2, q1)
    yield YPow(1 / 2 - ct, q1)
    yield ZPow(-3 / 2 + cp, q1)


@register_translation
def translate_can_to_cnot(
    gate: Can,
) -> Iterator[Union[CNot, S, S_H, XPow, YPow, ZPow, V, Z, V_H]]:
    """Translate canonical gate to 3 CNots, using the Kraus-Cirac decomposition.

    ::

        ───Can(tx,ty,tz)───     ───────X───Z^tz-1/2───●──────────────X───S⁺───
            │                          │              │              │
        ───Can(tx,ty,tz)───     ───S───●───Y^tx-1/2───X───Y^1/2-ty───●────────

    Ref:
        Vatan and Williams. Optimal quantum circuits for general two-qubit gates.
        Phys. Rev. A, 69:032315, 2004. quant-ph/0308006 :cite:`Vatan2004a` Fig. 6

        B. Kraus and J. I. Cirac, Phys. Rev. A 63, 062309 (2001).
    """  # noqa: W291, E501
    # TODO: Other special cases

    # Note: sign flip on central ZPow, YPow, YPow because of differing sign
    # conventions for Canonical.
    tx, ty, tz = gate.params
    q0, q1 = gate.qubits

    if var.isclose(tz, 0.0):
        # If we know tz is close to zero we only need two CNot gates
        yield Z(q0)
        yield V(q0).H
        yield Z(q1)
        yield V(q1).H
        yield CNot(q0, q1)
        yield XPow(tx, q0)
        yield ZPow(ty, q1)
        yield CNot(q0, q1)
        yield V(q0)
        yield Z(q0)
        yield V(q1)
        yield Z(q1)
    else:
        yield S(q1)
        yield CNot(q1, q0)
        yield ZPow(tz - 0.5, q0)
        yield YPow(-0.5 + tx, q1)
        yield CNot(q0, q1)
        yield YPow(-ty + 0.5, q1)
        yield CNot(q1, q0)
        yield S_H(q0)


@register_translation
def translate_can_to_xx_yy_zz(gate: Can) -> Iterator[Union[XX, YY, ZZ]]:
    """Convert a canonical gate to a circuit with XX, YY, and ZZ gates."""
    tx, ty, tz = gate.params
    q0, q1 = gate.qubits

    if not var.isclose(tx, 0.0):
        yield XX(tx, q0, q1)
    if not var.isclose(ty, 0.0):
        yield YY(ty, q0, q1)
    if not var.isclose(tz, 0.0):
        yield ZZ(tz, q0, q1)


@register_translation
def translate_ch_to_cpt(gate: CH) -> Iterator[Union[CNot, S, T, S_H, T_H, H]]:
    """Decomposition of a controlled Hadamard-gate into the Clifford+T.
    ::

        ───●───     ───────────────●─────────────────
           │                       │
        ───H───     ───S───H───T───X───T⁺───H───S⁺───

    Ref:
        http://arxiv.org/abs/1206.0758v3, Figure 5(a)
    """
    # Kudos: Adapted from QuipperLib
    # https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html
    q0, q1 = gate.qubits
    yield S(q1)
    yield H(q1)
    yield T(q1)
    yield CNot(q0, q1)
    yield T_H(q1)
    yield H(q1)
    yield S_H(q1)


@register_translation
def translate_cnot_to_cz(gate: CNot) -> Iterator[Union[H, CZ]]:
    """Convert CNot gate to a CZ based circuit."""
    q0, q1 = gate.qubits
    yield H(q1)
    yield CZ(q0, q1)
    yield H(q1)


@register_translation
def translate_cnot_to_sqrtiswap(gate: CNot) -> Iterator[Union[SqrtISwap_H, X, S_H, H]]:
    """Translate a CNOT gate to a square-root-iswap sandwich"""
    q0, q1 = gate.qubits

    # TODO: simplify 1-qubit gates
    yield H(q1)
    yield S_H(q0)
    yield S_H(q1)
    yield H(q0)
    yield H(q1)
    yield SqrtISwap(q0, q1).H
    yield X(q0)
    yield SqrtISwap(q0, q1).H
    yield X(q0)
    yield H(q0)


@register_translation
def translate_cnot_to_sqrtswap(gate: CNot) -> Iterator[Union[SqrtSwap, YPow, ZPow, Z]]:
    """Translate square-root swap to canonical"""
    # https://qipc2011.ethz.ch/uploads/Schoolpresentations/berghaus2011_DiVincenzo.pdf
    q0, q1 = gate.qubits

    yield Y(q1) ** 0.5
    yield SqrtSwap(q0, q1)
    yield Z(q0)
    yield SqrtSwap(q0, q1)
    yield Z(q0) ** -0.5
    yield Z(q1) ** -0.5
    yield Y(q1) ** -0.5


@register_translation
def translate_cnot_to_xx(gate: CNot) -> Iterator[Union[XX, H, S_H]]:
    """Convert CNot to XX gate"""
    # TODO: simplify 1-qubit gates
    q0, q1 = gate.qubits
    yield H(q0)
    yield XX(0.5, q0, q1)
    yield H(q0)
    yield H(q1)
    yield S(q0).H
    yield S(q1).H
    yield H(q1)


@register_translation
def translate_cy_to_cnot(gate: CY) -> Iterator[Union[CNot, S, S_H]]:
    """Translate CY to CNot (CX)"""
    q0, q1 = gate.qubits
    yield S_H(q1)
    yield CNot(q0, q1)
    yield S(q1)


@register_translation
def translate_cypow_to_cxpow(gate: CYPow) -> Iterator[Union[CNotPow, S, S_H]]:
    """Translate powers of CY to powers of CNot (CX)"""
    (t,) = gate.params
    q0, q1 = gate.qubits
    yield S_H(q1)
    yield CNot(q0, q1) ** t
    yield S(q1)


@register_translation
def translate_cphase_to_zz(gate: CPhase) -> Iterator[Union[ZZ, ZPow]]:
    """Convert a CPhase gate to a ZZ based circuit."""
    t = -gate.param("theta") / (2 * var.PI)
    q0, q1 = gate.qubits
    yield ZZ(t, q0, q1)
    yield ZPow(-t, q0)
    yield ZPow(-t, q1)


@register_translation
def translate_cphase00_to_cphase(gate: CPhase00) -> Iterator[Union[X, CPhase]]:
    """Convert a CPhase00 gate to a CPhase."""
    theta = gate.param("theta")
    q0, q1 = gate.qubits
    yield X(q0)
    yield X(q1)
    yield CPhase(theta, q0, q1)
    yield X(q0)
    yield X(q1)


@register_translation
def translate_cphase01_to_cphase(gate: CPhase01) -> Iterator[Union[X, CPhase]]:
    """Convert a CPhase01 gate to a CPhase."""
    theta = gate.param("theta")
    q0, q1 = gate.qubits
    yield X(q0)
    yield CPhase(theta, q0, q1)
    yield X(q0)


@register_translation
def translate_cphase10_to_cphase(gate: CPhase10) -> Iterator[Union[X, CPhase]]:
    """Convert a CPhase10 gate to a CPhase."""
    theta = gate.param("theta")
    q0, q1 = gate.qubits
    yield X(q1)
    yield CPhase(theta, q0, q1)
    yield X(q1)


@register_translation
def translate_cross_resonance_to_xx(
    gate: CrossResonance,
) -> Iterator[Union[XX, XPow, YPow, X]]:
    """Translate a cross resonance gate to an XX based circuit"""
    s, b, c = gate.params
    q0, q1 = gate.qubits

    t7 = (
        var.arccos((1 + b**2 * var.cos(var.PI * var.sqrt(1 + b**2) * s)) / (1 + b**2))
        / var.PI
    )
    t4 = c * s
    t1 = (
        var.arccos(
            var.cos(0.5 * var.PI * var.sqrt(1 + b**2) * s) / var.cos(t7 * var.PI / 2)
        )
        / var.PI
    )

    a = var.sin(var.PI * var.sqrt(1 + b**2) * s / 2)
    t7 *= var.sign(a) * var.sign(b)
    t1 *= var.sign(a)

    yield XPow(t1, q0)
    yield YPow(1.5, q0)
    yield X(q0)
    yield XPow(t4, q1)
    yield XX(t7, q0, q1)
    yield YPow(1.5, q0)
    yield X(q0)
    yield XPow(t1, q0)


@register_translation
def translate_crx_to_cnotpow(gate: CRx) -> Iterator[Union[CNotPow, PhaseShift]]:
    """Translate QASM's CRx gate to powers of a CNot gate."""
    q0, q1 = gate.qubits
    (theta,) = gate.params

    yield CNotPow(theta / var.PI, q0, q1)
    yield PhaseShift(-theta / 2, q0)


@register_translation
def translate_cry_to_cypow(gate: CRy) -> Iterator[Union[CYPow, PhaseShift]]:
    """Translate QASM's CRy gate to powers of a CY gate."""
    q0, q1 = gate.qubits
    (theta,) = gate.params

    yield CYPow(theta / var.PI, q0, q1)
    yield PhaseShift(-theta / 2, q0)


@register_translation
def translate_crz_to_czpow(gate: CRz) -> Iterator[Union[CZPow, PhaseShift]]:
    """Translate QASM's CRz gate to powers of a CZ gate."""
    q0, q1 = gate.qubits
    (theta,) = gate.params

    yield CZPow(theta / var.PI, q0, q1)
    yield PhaseShift(-theta / 2, q0)


@register_translation
def translate_crz_to_cnot(gate: CRz) -> Iterator[Union[CNot, PhaseShift]]:
    """Translate QASM's CRZ gate to standard gates.

    Ref:
        https://arxiv.org/pdf/1707.03429.pdf
    """
    q0, q1 = gate.qubits
    (theta,) = gate.params

    yield PhaseShift(theta / 2, q1)
    yield CNot(q0, q1)
    yield PhaseShift(-theta / 2, q1)
    yield CNot(q0, q1)


@register_translation
def translate_cnotpow_to_zz(gate: CNotPow) -> Iterator[Union[ZZ, ZPow, H]]:
    """Convert a controlled X^t gate to a ZZ based circuit.
    ::

        ───●─────     ───────ZZ^-t/2───Z^t/2───────
           │       =          │
        ───X^t───     ───H───ZZ^-t/2───Z^t/2───H───

    """
    (t,) = gate.params
    q0, q1 = gate.qubits
    yield H(q1)
    yield ZZ(-t / 2, q0, q1)
    yield ZPow(t / 2, q0)
    yield ZPow(t / 2, q1)
    yield H(q1)


@register_translation
def translate_cz_to_zz(gate: CZ) -> Iterator[Union[ZZ, S_H]]:
    """Convert CZ gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    yield ZZ(0.5, q0, q1)
    yield S_H(q0)
    yield S_H(q1)


@register_translation
def translate_czpow_to_zz(gate: CZPow) -> Iterator[Union[ZZ, ZPow]]:
    """Convert a CZPow gate to a ZZ based circuit."""
    t = gate.param("t")
    q0, q1 = gate.qubits
    yield ZZ(-t / 2, q0, q1)
    yield ZPow(t / 2, q0)
    yield ZPow(t / 2, q1)


@register_translation
def translate_czpow_to_cphase(gate: CZPow) -> Iterator[CPhase]:
    """Convert a CZPow gate to CPhase."""
    theta = gate.param("t") * var.PI
    yield CPhase(theta, *gate.qubits)


@register_translation
def translate_cphase_to_czpow(gate: CPhase) -> Iterator[CZPow]:
    """Convert a CPhase gate to a CZPow."""
    (theta,) = gate.params
    t = theta / var.PI
    yield CZPow(t, *gate.qubits)


# TODO: cphase to fsim
# TODO: fsim specialize


@register_translation
def translate_cu3_to_cnot(gate: CU3) -> Iterator[Union[CNot, PhaseShift, U3]]:
    """Translate QASM's CU3 gate to standard gates"""
    # Kudos: Adapted from qiskit
    # https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/cu3.py
    q0, q1 = gate.qubits
    theta, phi, lam = gate.params

    yield PhaseShift((lam + phi) / 2, q0)
    yield PhaseShift((lam - phi) / 2, q1)
    yield CNot(q0, q1)
    yield U3(-theta / 2, 0, -(phi + lam) / 2, q1)
    yield CNot(q0, q1)
    yield U3(theta / 2, phi, 0, q1)


@register_translation
def translate_cv_to_cpt(gate: CV) -> Iterator[Union[CNot, T, T_H, H]]:
    """Decomposition of a controlled sqrt(X)-gate into the Clifford+T.
    ::

        ───●───     ───T───X───T⁺───X───────
           │               │        │
        ───V───     ───H───●───T────●───H───

    Ref:
        http://arxiv.org/abs/1206.0758v3, Figure 5(c)
    """
    q0, q1 = gate.qubits
    yield T(q0)
    yield H(q1)
    yield CNot(q1, q0)
    yield T_H(q0)
    yield T(q1)
    yield CNot(q1, q0)
    yield H(q1)


@register_translation
def translate_cvh_to_cpt(gate: CV_H) -> Iterator[Union[CNot, T, T_H, H]]:
    """Decomposition of a controlled sqrt(X)-gate into the Clifford+T.
    ::

        ───●────     ───────X───T────X───T⁺───
           │                │        │
        ───V⁺───     ───H───●───T⁺───●───H────

    Ref:
        http://arxiv.org/abs/1206.0758v3, Figure 5(c)
    """
    q0, q1 = gate.qubits
    yield H(q1)
    yield CNot(q1, q0)
    yield T_H(q1)
    yield T(q0)
    yield CNot(q1, q0)
    yield H(q1)
    yield T_H(q0)


@register_translation
def translate_ecp_to_can(gate: ECP) -> Iterator[Can]:
    """Translate an ECP gate to a Canonical gate"""
    yield Can(1 / 2, 1 / 4, 1 / 4, *gate.qubits)


@register_translation
def translate_ecp_to_sqrtiswap(gate: ECP) -> Iterator[Union[SqrtISwap_H, YPow, S, S_H]]:
    """Translate an ECP gate to a square-root-iswap sandwich"""
    q0, q1 = gate.qubits

    yield SqrtISwap(q0, q1).H
    yield S(q0)
    yield S(q1)
    yield YPow(0.5, q0)
    yield YPow(0.5, q1)
    yield SqrtISwap(q0, q1).H

    yield YPow(-0.5, q1)
    yield S_H(q1)

    yield YPow(-0.5, q0)
    yield S_H(q0)


@register_translation
def translate_exch_to_can(gate: Exch) -> Iterator[Can]:
    """Convert an exchange gate to a canonical based circuit"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(t, t, t, q0, q1)


@register_translation
def translate_exch_to_xy_zz(gate: Exch) -> Iterator[Union[XY, ZZ]]:
    """Convert an exchange gate to XY and ZZ gates"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield XY(t, q0, q1)
    yield ZZ(t, q0, q1)


@register_translation
def translate_fsim_to_xy_cz(gate: FSim) -> Iterator[Union[XY, CZ]]:
    """Convert the Cirq's FSim  gate to a canonical gate"""
    q0, q1 = gate.qubits
    theta, phi = gate.params

    yield XY(theta / var.PI, q0, q1)
    yield CZ(q0, q1) ** (-phi / var.PI)


@register_translation
def translate_fswap(gate: FSwap) -> Iterator[Union[Swap, CZ]]:
    """Translate fSwap gate to Swap and CV"""
    q0, q1 = gate.qubits
    yield Swap(q0, q1)
    yield CZ(q0, q1)


@register_translation
def translate_fswappow(gate: FSwapPow) -> Iterator[Union[Exch, CZPow]]:
    """Translate fSwap gate to XY and CVPow"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Swap(q0, q1) ** t
    yield CZ(q0, q1) ** t


# TODO: Other givens deke
@register_translation
def translate_givens_to_xy(gate: Givens) -> Iterator[Union[XY, T, T_H]]:
    """Convert a Givens  gate to an XY gate"""
    q0, q1 = gate.qubits
    (theta,) = gate.params

    yield T_H(q0)
    yield T(q1)
    yield XY(theta / var.PI, q0, q1)
    yield T(q0)
    yield T_H(q1)


@register_translation
def translate_iswap_to_can(gate: ISwap) -> Iterator[Union[Can, X]]:
    """Convert ISwap gate to a canonical gate within the Weyl chamber."""
    q0, q1 = gate.qubits
    yield X(q0)
    yield Can(0.5, 0.5, 0, q0, q1)
    yield X(q1)


@register_translation
def translate_iswap_to_swap_cz(gate: ISwap) -> Iterator[Union[Swap, CZ, S]]:
    """Convert ISwap gate to a Swap, CZ based circuit."""
    q0, q1 = gate.qubits
    yield Swap(q0, q1)
    yield CZ(q0, q1)
    yield S(q0)
    yield S(q1)


@register_translation
def translate_iswap_to_sqrtiswap(gate: ISwap) -> Iterator[SqrtISwap]:
    """Translate iswap gate to square-root iswaps"""
    q0, q1 = gate.qubits
    yield SqrtISwap(q0, q1)
    yield SqrtISwap(q0, q1)


@register_translation
def translate_iswap_to_xy(gate: ISwap) -> Iterator[XY]:
    """Convert ISwap gate to a XY gate."""
    q0, q1 = gate.qubits
    yield XY(-0.5, q0, q1)


@register_translation
def translate_pswap_to_canonical(gate: PSwap) -> Iterator[Union[Can, Y]]:
    """Translate parametric Swap to a canonical circuit"""

    q0, q1 = gate.qubits
    (theta,) = gate.params
    t = 0.5 - theta / var.PI
    yield Y(q0)
    yield Can(0.5, 0.5, t, q0, q1)
    yield Y(q1)


@register_translation
def translate_rxx_to_xx(gate: Rxx) -> Iterator[XX]:
    """Translate QASM's RXX gate to standard gates"""
    q0, q1 = gate.qubits
    (theta,) = gate.params
    yield XX(theta / var.PI, q0, q1)


@register_translation
def translate_ryy_to_yy(gate: Ryy) -> Iterator[YY]:
    """Translate QASM's RYY gate to standard gates"""
    q0, q1 = gate.qubits
    (theta,) = gate.params
    yield YY(theta / var.PI, q0, q1)


@register_translation
def translate_rzz_to_zz(gate: Rzz) -> Iterator[ZZ]:
    """Translate QASM's RZZ gate to standard gates"""
    q0, q1 = gate.qubits
    (theta,) = gate.params
    yield ZZ(theta / var.PI, q0, q1)


@register_translation
def translate_rzz_to_cnot(gate: Rzz) -> Iterator[Union[CNot, PhaseShift, U3]]:
    """Translate QASM's Rzz gate to standard gates"""
    q0, q1 = gate.qubits
    (theta,) = gate.params
    yield CNot(q0, q1)
    yield PhaseShift(theta, q1)
    yield CNot(q0, q1)


@register_translation
def translate_sqrtiswap_to_sqrtiswap_h(
    gate: SqrtISwap,
) -> Iterator[Union[SqrtISwap_H, Z]]:
    """Translate square-root-iswap to its inverse."""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield SqrtISwap_H(q0, q1)
    yield Z(q0)


@register_translation
def translate_sqrtiswap_h_to_can(gate: SqrtISwap_H) -> Iterator[Can]:
    """Translate square-root iswap to canonical"""
    yield Can(1 / 4, 1 / 4, 0, *gate.qubits)


@register_translation
def translate_sqrtiswap_h_to_sqrtiswap(
    gate: SqrtISwap_H,
) -> Iterator[Union[SqrtISwap, Z]]:
    """Translate square-root-iswap to it's inverse."""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield SqrtISwap(q0, q1)
    yield Z(q0)


@register_translation
def translate_sqrtswap_to_can(gate: SqrtSwap) -> Iterator[Can]:
    """Translate square-root swap to canonical"""
    yield Can(1 / 4, 1 / 4, 1 / 4, *gate.qubits)


@register_translation
def translate_sqrtswap_h_to_can(gate: SqrtSwap_H) -> Iterator[Can]:
    """Translate inv. square-root swap to canonical"""
    yield Can(-1 / 4, -1 / 4, -1 / 4, *gate.qubits)


@register_translation
def translate_swap_to_cnot(gate: Swap) -> Iterator[CNot]:
    """Convert a Swap gate to a circuit with 3 CNots."""
    q0, q1 = gate.qubits
    yield CNot(q0, q1)
    yield CNot(q1, q0)
    yield CNot(q0, q1)


@register_translation
def translate_swap_to_ecp_sqrtiswap(
    gate: Swap,
) -> Iterator[Union[ECP, SqrtISwap_H, H, ZPow, YPow]]:
    """Translate a Swap gate to an  ECP -- square-root-iswap sandwich.

    An intermediate step in translating swap to 3 square-root-iswap's.
    """
    q0, q1 = gate.qubits

    yield ECP(q0, q1)

    yield H(q0)
    yield H(q1)

    yield SqrtISwap(q0, q1).H

    yield YPow(-1 / 2, q1)
    yield ZPow(+1, q1)

    yield YPow(-1 / 2, q0)
    yield ZPow(+1, q0)


@register_translation
def translate_swap_to_iswap_cz(gate: Swap) -> Iterator[Union[ISwap, CZ, S_H]]:
    """Convert ISwap gate to a Swap, CZ based circuit."""
    q0, q1 = gate.qubits
    yield S_H(q0)
    yield S_H(q1)
    yield CZ(q0, q1)
    yield ISwap(q0, q1)


@register_translation
def translate_sycamore_to_fsim(gate: Sycamore) -> Iterator[FSim]:
    """Convert a Sycamore gate to an FSim gate"""
    yield FSim(var.PI / 2, var.PI / 6, *gate.qubits)


@register_translation
def translate_syc_to_can(gate: Sycamore) -> Iterator[Union[Can, ZPow]]:
    """Convert a Sycamore gate to an canonical gate"""
    q0, q1 = gate.qubits
    yield Can(1 / 2, 1 / 2, 1 / 12, q0, q1)
    yield Z(q0) ** (-1 / 12)
    yield Z(q1) ** (-1 / 12)


@register_translation
def translate_syc_to_cphase(gate: Sycamore) -> Iterator[Union[CPhase, ISwap, Z]]:
    """Convert a Sycamore gate to a CPhase gate"""
    q0, q1 = gate.qubits
    yield CPhase(-var.PI / 6, q0, q1)
    yield Z(q0)
    yield ISwap(q0, q1)
    yield Z(q0)


@register_translation
def translate_w_to_ecp(gate: W) -> Iterator[Union[ECP, H, S, S_H, T, T_H]]:
    """Translate W gate to ECP."""
    # TODO: Cite self
    q0, q1 = gate.qubits
    yield T(q0).H
    yield H(q0)
    yield T(q1).H
    yield S(q1).H
    yield H(q1)
    yield ECP(q0, q1)
    yield H(q0)
    yield S(q0)
    yield T(q0)
    yield H(q1)
    yield T(q1)


@register_translation
def translate_w_to_cnot(gate: W) -> Iterator[Union[CNot, H, S, S_H, T, T_H]]:
    """Translate W gate to CNot."""
    # Kudos: Decomposition given in Quipper
    # (Dek in paper far from optimal)
    # https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html
    q0, q1 = gate.qubits

    yield CNot(q0, q1)
    yield S(q0).H
    yield H(q0)
    yield T(q0).H
    yield CNot(q1, q0)
    yield T(q0)
    yield H(q0)
    yield S(q0)
    yield CNot(q0, q1)


@register_translation
def translate_w_to_ch_cnot(gate: W) -> Iterator[Union[CNot, CH]]:
    """Translate W gate to controlled-Hadamard and CNot."""
    # From https://arxiv.org/pdf/1505.06552.pdf
    q0, q1 = gate.qubits

    yield CNot(q1, q0)
    yield CNot(q0, q1)
    yield CH(q0, q1)
    yield CNot(q0, q1)
    yield CNot(q1, q0)


@register_translation
def translate_xx_to_can(gate: XX) -> Iterator[Can]:
    """Convert an XX gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(t, 0, 0, q0, q1)


@register_translation
def translate_xx_to_zz(gate: XX) -> Iterator[Union[H, ZZ]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield H(q0)
    yield H(q1)
    yield ZZ(t, q0, q1)
    yield H(q0)
    yield H(q1)


@register_translation
def translate_xy_to_can(gate: XY) -> Iterator[Can]:
    """Convert XY gate to a canonical gate."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(t, t, 0, q0, q1)


@register_translation
def translate_xy_to_sqrtiswap(
    gate: XY,
) -> Iterator[Union[Z, T, T_H, ZPow, SqrtISwap_H]]:
    """Translate an XY gate to sandwich of square-root iswaps"""

    q0, q1 = gate.qubits
    t = gate.param("t")

    yield Z(q0)
    yield T(q0)
    yield T_H(q1)
    yield SqrtISwap_H(q0, q1)
    yield ZPow(1 - t, q0)
    yield ZPow(t, q1)
    yield SqrtISwap_H(q0, q1)
    yield T_H(q0)
    yield T(q1)


@register_translation
def translate_yy_to_can(gate: YY) -> Iterator[Can]:
    """Convert an YY gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(0, t, 0, q0, q1)


@register_translation
def translate_yy_to_zz(gate: YY) -> Iterator[Union[XPow, ZZ]]:
    """Convert a YY gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield XPow(0.5, q0)
    yield XPow(0.5, q1)
    yield ZZ(t, q0, q1)
    yield XPow(-0.5, q0)
    yield XPow(-0.5, q1)


@register_translation
def translate_zz_to_can(gate: ZZ) -> Iterator[Can]:
    """Convert an ZZ gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(0, 0, t, q0, q1)


@register_translation
def translate_zz_to_cnot(gate: ZZ) -> Iterator[Union[CNot, ZPow]]:
    """Convert a ZZ gate to a CNot based circuit"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield CNot(q0, q1)
    yield ZPow(t, q1)
    yield CNot(q0, q1)


@register_translation
def translate_zz_to_xx(gate: ZZ) -> Iterator[Union[H, XX]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield H(q0)
    yield H(q1)
    yield XX(t, q0, q1)
    yield H(q0)
    yield H(q1)


@register_translation
def translate_zz_to_yy(gate: ZZ) -> Iterator[Union[XPow, YY]]:
    """Convert a YY gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield XPow(-0.5, q0)
    yield XPow(-0.5, q1)
    yield YY(t, q0, q1)
    yield XPow(0.5, q0)
    yield XPow(0.5, q1)


@register_translation
def translate_CS_to_CZPow(gate: CS) -> Iterator[CZPow]:
    """Convert a controlled-S to half power of CZ gate"""
    yield CZPow(0.5, *gate.qubits)


@register_translation
def translate_CT_to_CZPow(gate: CT) -> Iterator[CZPow]:
    """Convert a controlled-T to quarter power of CZ gate"""
    yield CZPow(0.25, *gate.qubits)


@register_translation
def translate_a_to_cnot(gate: A) -> Iterator[Union[CNot, Rz, Ry]]:
    """Translate the A-gate to 3 CNots.

    Ref:
        Fig. 2 :cite:`Gard2020a`
    """
    (q0, q1) = gate.qubits
    (theta, phi) = gate.params
    yield CNot(q1, q0)
    yield Rz(-phi - var.PI, q1)
    yield Ry(-theta - var.PI / 2, q1)
    yield CNot(q0, q1)
    yield Ry(theta + var.PI / 2, q1)
    yield Rz(phi + var.PI, q1)
    yield CNot(q1, q0)


@register_translation
def translate_a_to_can(gate: A) -> Iterator[Union[Can, ZPow]]:
    """Translate the A-gate to the canonical gate.

    Ref:
        Page 3
        :cite:`Gard2020a`
    """
    (q0, q1) = gate.qubits
    (theta, phi) = gate.params

    yield ZPow(1 / 2, q0)
    yield ZPow(-phi / var.PI, q1)

    yield Can(theta / var.PI, theta / var.PI, 0.5, q0, q1)

    # Note: There seems to be a sign error in the paper referenced above for
    # this part of the expression
    yield ZPow(phi / var.PI - 1 / 2, q1)


# fin
