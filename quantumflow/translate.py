# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Translate, transform, and compile circuits.
"""

# TODO: Split into separate modules
# TODO: Put translations into alphabetic order so they can be found

from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    Type,
    Union,
)

from . import var
from .circuits import Circuit
from .ops import Gate
from .stdgates import (
    CCZ,
    CH,
    CRZ,
    CU3,
    CV,
    CV_H,
    CY,
    CZ,
    ECP,
    RZZ,
    S_H,
    T_H,
    U2,
    U3,
    V_H,
    XX,
    XY,
    YY,
    ZZ,
    B,
    Barenco,
    Can,
    CCiX,
    CCNot,
    CCXPow,
    CNot,
    CNotPow,
    CPhase,
    CPhase00,
    CPhase01,
    CPhase10,
    CrossResonance,
    CSwap,
    CYPow,
    CZPow,
    Deutsch,
    Exch,
    FSim,
    FSwap,
    FSwapPow,
    Givens,
    H,
    HPow,
    I,
    ISwap,
    Ph,
    PhasedX,
    PhasedXPow,
    PhaseShift,
    PSwap,
    Rn,
    Rx,
    Ry,
    Rz,
    S,
    SqrtISwap,
    SqrtISwap_H,
    SqrtSwap,
    SqrtSwap_H,
    SqrtY,
    SqrtY_H,
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

# Implementation note:
# __all__ is auto-magically created at end of module
#
# These translations are all analytic, so that we can use symbolic
# parameters. Numerical decompositions can be found in decompositions.py
#
# Translations return an Iterator over gates, rather than a Circuit, in part
# so that we can use type annotations to keep track of the source and resultant
# gates of each translation.


def translation_source_gate(trans: Callable) -> Type[Gate]:
    return trans.__annotations__["gate"]


def translation_target_gates(trans: Callable) -> Tuple[Type[Gate]]:

    # FIXME: Fails with KeyError if no return type annotation.
    # Add a more informative error message.
    ret = trans.__annotations__["return"].__args__[0]
    if hasattr(ret, "__args__"):  # Union
        gates = ret.__args__
    else:
        gates = (ret,)

    return gates


# TODO: Rename
def select_translators(
    target_gates: Iterable[Type[Gate]], translators: Iterable[Callable] = None
) -> List[Callable]:
    """Return a list of translations that will translate source gates to target
    gates.

    If no translations are specified, we use all QuantumFlow translations
    listed in qf.TRANSLATORS

    For example, to convert a circuit to use gates understood by QUIL:
    ::

        trans = qf.select_translators(qf.QUIL_GATES)
        circ = qf.transform(circ, trans)

    """
    # Warning: Black Voodoo magic. We use python's type annotations to figure
    # out the source gate and target gates of a translation.

    if translators is None:
        source_trans = set(TRANSLATORS.values())
    else:
        source_trans = set(translators)

    out_trans = []
    target_gates = set(target_gates)

    while source_trans:  # Loop until we run out of translations
        for trans in list(source_trans):
            from_gate = translation_source_gate(trans)
            to_gates = translation_target_gates(trans)
            if from_gate in target_gates:
                # If translation's source gates are already in targets
                # then we don't need this translation. Discard.
                source_trans.remove(trans)
                break

            if target_gates.issuperset(to_gates):
                # If target gate of translation are already in
                # target list, and source gate isn't, move
                # translation to output list and source gate to targets.
                target_gates.add(from_gate)
                out_trans.append(trans)
                source_trans.remove(trans)
                break
        else:
            # If we got here, none of the remaining translations can be
            # used. Break out of while and return results.
            break  # pragma: no cover

    return out_trans


def circuit_translate(
    circ: Circuit,
    translators: Sequence = None,
    targets: Sequence[Type[Gate]] = None,
    recurse: bool = True,
) -> Circuit:
    """Apply a collection of translations to each gate in a circuit.
    If recurse, then apply translations to output of translations
    until translationally invariant.
    """
    if translators is not None and targets is not None:
        raise ValueError("Specify either targets or translators, not both")

    gates = list(reversed(list(circ)))
    translated: List[Gate] = []

    if translators is None:
        if targets is None:
            targets = [Can, XPow, ZPow, I, Ph]
        translators = select_translators(targets)

    # Use type annotations to do dynamic dispatch
    gateclass_translation = {
        translation_source_gate(trans): trans for trans in translators
    }

    while gates:
        gate = gates.pop()
        if type(gate) in gateclass_translation:
            trans = gateclass_translation[type(gate)](gate)
            if recurse:
                gates.extend(reversed(list(trans)))
            else:
                translated.extend(trans)
        else:
            translated += gate

    return Circuit(translated)


# 1-qubit gates


def translate_x_to_tx(gate: X) -> Iterator[XPow]:
    """Translate X gate to XPow"""
    (q0,) = gate.qubits
    yield XPow(1, q0)


def translate_y_to_ty(gate: Y) -> Iterator[YPow]:
    """Translate Y gate to YPow"""
    (q0,) = gate.qubits
    yield YPow(1, q0)


def translate_z_to_tz(gate: Z) -> Iterator[ZPow]:
    """Translate Z gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(1, q0)


def translate_s_to_tz(gate: S) -> Iterator[ZPow]:
    """Translate S gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(0.5, q0)


def translate_t_to_tz(gate: T) -> Iterator[ZPow]:
    """Translate T gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(0.25, q0)


def translate_invs_to_tz(gate: S_H) -> Iterator[ZPow]:
    """Translate S.H gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(-0.5, q0)


def translate_invt_to_tz(gate: T_H) -> Iterator[ZPow]:
    """Translate inverse T gate to Rz (a quil standard gate)"""
    (q0,) = gate.qubits
    yield ZPow(-0.25, q0)


def translate_rx_to_tx(gate: Rx) -> Iterator[XPow]:
    """Translate Rx gate to XPow"""
    (q0,) = gate.qubits
    (theta,) = gate.params
    t = theta / var.PI
    yield XPow(t, q0)


def translate_ry_to_ty(gate: Ry) -> Iterator[YPow]:
    """Translate Ry gate to YPow"""
    (q0,) = gate.qubits
    (theta,) = gate.params
    t = theta / var.PI
    yield YPow(t, q0)


def translate_rz_to_tz(gate: Rz) -> Iterator[ZPow]:
    """Translate Rz gate to ZPow"""
    (q0,) = gate.qubits
    (theta,) = gate.params
    t = theta / var.PI
    yield ZPow(t, q0)


def translate_rn_to_rz_ry(gate: Rn) -> Iterator[Union[Rz, Ry]]:
    """Translate Rn Bloch rotation to Rz Ry Rz Ry Rz.

    Refs:
        http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf
    """
    (q0,) = gate.qubits
    theta, nx, ny, nz = gate.params

    ang_y = var.arccos(nz)
    ang_z = var.arctan2(ny, nx)

    yield Rz(-ang_z, q0)
    yield Ry(-ang_y, q0)
    yield Rz(theta, q0)
    yield Ry(ang_y, q0)
    yield Rz(ang_z, q0)


def translate_phase_to_rz(gate: PhaseShift) -> Iterator[Rz]:
    """Translate Phase gate to Rz (ignoring global phase)"""
    (q0,) = gate.qubits
    theta = gate.param("theta")
    yield Rz(theta, q0)


def translate_sqrty_to_ty(gate: SqrtY) -> Iterator[YPow]:
    """Translate sqrt-Y gate to YPow"""
    (q0,) = gate.qubits
    yield YPow(0.5, q0)


def translate_sqrty_h_to_ty(gate: SqrtY_H) -> Iterator[YPow]:
    """Translate sqrt-Y gate to YPow"""
    (q0,) = gate.qubits
    yield YPow(-0.5, q0)


def translate_tx_to_rx(gate: XPow) -> Iterator[Rx]:
    """Translate XPow gate to Rx"""
    (q0,) = gate.qubits
    theta = gate.param("t") * var.PI
    yield Rx(theta, q0)


def translate_ty_to_ry(gate: YPow) -> Iterator[Ry]:
    """Translate YPow gate to Ry"""
    (q0,) = gate.qubits
    theta = gate.param("t") * var.PI
    yield Ry(theta, q0)


def translate_tz_to_rz(gate: ZPow) -> Iterator[Rz]:
    """Translate ZPow gate to Rz"""
    (q0,) = gate.qubits
    theta = gate.param("t") * var.PI
    yield Rz(theta, q0)


def translate_ty_to_xzx(gate: YPow) -> Iterator[Union[XPow, ZPow]]:
    """Translate YPow gate to XPow and ZPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield XPow(0.5, q0)
    yield ZPow(t, q0)
    yield XPow(-0.5, q0)


def translate_tx_to_zyz(gate: XPow) -> Iterator[Union[YPow, S, S_H]]:
    """Translate XPow gate to S and YPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield S(q0)
    yield YPow(t, q0)
    yield S_H(q0)


def translate_tz_to_xyx(gate: ZPow) -> Iterator[Union[YPow, V, V_H]]:
    """Translate ZPow gate to V and YPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield V_H(q0)
    yield YPow(t, q0)
    yield V(q0)


def translate_phased_x_to_zxz(gate: PhasedX) -> Iterator[Union[X, ZPow]]:
    """Translate YPow gate to XPow and ZPow gates"""
    (q0,) = gate.qubits
    p = gate.param("p")
    yield ZPow(-p, q0)
    yield X(q0)
    yield ZPow(p, q0)


def translate_phased_tx_to_zxz(gate: PhasedXPow) -> Iterator[Union[XPow, ZPow]]:
    """Translate YPow gate to XPow and ZPow gates"""
    (q0,) = gate.qubits
    p, t = gate.params
    yield ZPow(-p, q0)
    yield XPow(t, q0)
    yield ZPow(p, q0)


def translate_v_to_tx(gate: V) -> Iterator[XPow]:
    """Translate V gate to XPow"""
    (q0,) = gate.qubits
    yield XPow(0.5, q0)


def translate_invv_to_tx(gate: V_H) -> Iterator[XPow]:
    """Translate V_H gate to XPow"""
    (q0,) = gate.qubits
    yield XPow(-0.5, q0)


def translate_th_to_tx(gate: HPow) -> Iterator[Union[XPow, H, S, T, S_H, T_H]]:
    """Translate powers of the Hadamard gate to XPow and YPow"""
    (q0,) = gate.qubits
    (t,) = gate.params

    yield S(q0)
    yield H(q0)
    yield T(q0)
    yield XPow(t, q0)
    yield T(q0).H
    yield H(q0)
    yield S(q0).H


def translate_ty_to_zxz(gate: YPow) -> Iterator[Union[XPow, S, S_H]]:
    """Translate YPow gate to ZPow and XPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield S_H(q0)
    yield XPow(t, q0)
    yield S(q0)


def translate_tx_to_zxzxz(gate: XPow) -> Iterator[Union[XPow, ZPow]]:
    """Convert an arbitrary power of a Pauli-X gate to Z and V gates"""
    (q0,) = gate.qubits
    t = gate.param("t")

    if var.isclose(t, 0.5) or var.isclose(t, -0.5):
        yield gate
        return

    yield ZPow(0.5, q0)
    yield XPow(0.5, q0)
    yield ZPow(t, q0)
    yield XPow(-0.5, q0)
    yield ZPow(-0.5, q0)


def translate_hadamard_to_zxz(gate: H) -> Iterator[Union[XPow, ZPow]]:
    """Convert a Hadamard gate to a circuit with ZPow and XPow gates."""
    (q0,) = gate.qubits
    yield ZPow(0.5, q0)
    yield XPow(0.5, q0)
    yield ZPow(0.5, q0)


def translate_u3_to_zyz(gate: U3) -> Iterator[Union[Rz, Ry]]:
    """Translate QASMs U3 gate to Rz and Ry"""
    (q0,) = gate.qubits
    theta, phi, lam = gate.params
    yield Rz(lam, q0)
    yield Ry(theta, q0)
    yield Rz(phi, q0)


def translate_u2_to_zyz(gate: U2) -> Iterator[Union[Rz, Ry]]:
    """Translate QASMs U2 gate to Rz and Ry"""
    (q0,) = gate.qubits
    phi, lam = gate.params
    yield Rz(lam, q0)
    yield Ry(var.PI / 2, q0)
    yield Rz(phi, q0)


def translate_tx_to_hzh(gate: XPow) -> Iterator[Union[H, ZPow]]:
    """Convert a XPow gate to a circuit with Hadamard and ZPow gates"""
    (q0,) = gate.qubits
    (t,) = gate.params
    yield H(q0)
    yield ZPow(t, q0)
    yield H(q0)


# 2-qubit gates


def translate_b_to_can(gate: B) -> Iterator[Union[Can, Y, Z]]:
    """Translate B gate to Canonical gate"""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield Y(q1)
    yield Can(1 / 2, 1 / 4, 0, q0, q1)
    yield Y(q0)
    yield Z(q1)


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


def translate_can_to_cnot(
    gate: Can,
) -> Iterator[Union[CNot, S, S_H, XPow, YPow, ZPow, V, Z]]:
    """Translate canonical gate to 3 CNotS, using the Kraus-Cirac decomposition.

    ::

        ───Can(tx,ty,tz)───     ───────X───Z^tz-1/2───●──────────────X───S⁺───
            │                          │              │              │
        ───Can(tx,ty,tz)───     ───S───●───Y^tx-1/2───X───Y^1/2-ty───●────────

    Ref:
        Vatan and Williams. Optimal quantum circuits for general two-qubit gates.
        Phys. Rev. A, 69:032315, 2004. quant-ph/0308006 :cite:`Vatan2004` Fig. 6

        B. Kraus and J. I. Cirac, Phys. Rev. A 63, 062309 (2001).
    """  # noqa: W291, E501
    # TODO: Other special cases

    # Note: sign flip on central ZPow, YPow, YPow because of differing sign
    # conventions for Canonical.
    tx, ty, tz = gate.params
    q0, q1 = gate.qubits

    if var.isclose(tz, 0.0):
        # If we know tz is close to zero we only need two CNot gates
        yield V(q0)
        yield Z(q0)
        yield V(q1)
        yield Z(q1)
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


def translate_cnot_to_cz(gate: CNot) -> Iterator[Union[H, CZ]]:
    """Convert CNot gate to a CZ based circuit."""
    q0, q1 = gate.qubits
    yield H(q1)
    yield CZ(q0, q1)
    yield H(q1)


def translate_cnot_to_sqrtiswap(gate: CNot) -> Iterator[Union[SqrtISwap_H, X, S_H, H]]:
    """Translate an ECP gate to a square-root-iswap sandwich"""
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


def translate_cy_to_cnot(gate: CY) -> Iterator[Union[CNot, S, S_H]]:
    """Translate CY to CNot (CX)"""
    q0, q1 = gate.qubits
    yield S_H(q1)
    yield CNot(q0, q1)
    yield S(q1)


def translate_cypow_to_cxpow(gate: CYPow) -> Iterator[Union[CNotPow, S, S_H]]:
    """Translate powers of CY to powers of CNot (CX)"""
    yield from gate.decompose()


def translate_cphase_to_zz(gate: CPhase) -> Iterator[Union[ZZ, ZPow]]:
    """Convert a CPhase gate to a ZZ based circuit."""
    t = -gate.param("theta") / (2 * var.PI)
    q0, q1 = gate.qubits
    yield ZZ(t, q0, q1)
    yield ZPow(-t, q0)
    yield ZPow(-t, q1)


def translate_cphase00_to_zz(gate: CPhase00) -> Iterator[Union[X, ZZ, ZPow]]:
    """Convert a CPhase00 gate to a ZZ based circuit."""
    t = -gate.param("theta") / (2 * var.PI)
    q0, q1 = gate.qubits
    yield X(q0)
    yield X(q1)
    yield ZZ(t, q0, q1)
    yield ZPow(-t, q0)
    yield ZPow(-t, q1)
    yield X(q0)
    yield X(q1)


def translate_cphase01_to_zz(gate: CPhase01) -> Iterator[Union[X, ZZ, ZPow]]:
    """Convert a CPhase01 gate to a ZZ based circuit."""
    t = -gate.param("theta") / (2 * var.PI)
    q0, q1 = gate.qubits
    yield X(q0)
    yield ZZ(t, q0, q1)
    yield ZPow(-t, q0)
    yield ZPow(-t, q1)
    yield X(q0)


def translate_cphase10_to_zz(gate: CPhase10) -> Iterator[Union[X, ZZ, ZPow]]:
    """Convert a CPhase10 gate to a ZZ based circuit."""
    t = -gate.param("theta") / (2 * var.PI)
    q0, q1 = gate.qubits

    yield X(q1)
    yield ZZ(t, q0, q1)
    yield ZPow(-t, q0)
    yield ZPow(-t, q1)
    yield X(q1)


def translate_cross_resonance_to_xx(
    gate: CrossResonance,
) -> Iterator[Union[XX, XPow, YPow, X]]:
    """Translate a cross resonance gate to an XX based circuit"""
    s, b, c = gate.params
    q0, q1 = gate.qubits

    t7 = (
        var.arccos(
            ((1 + b ** 2 * var.cos(var.PI * var.sqrt(1 + b ** 2) * s))) / (1 + b ** 2)
        )
        / var.PI
    )
    t4 = c * s
    t1 = (
        var.arccos(
            var.cos(0.5 * var.PI * var.sqrt(1 + b ** 2) * s) / var.cos(t7 * var.PI / 2)
        )
        / var.PI
    )

    a = var.sin(var.PI * var.sqrt(1 + b ** 2) * s / 2)
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


def translate_crz_to_cnot(gate: CRZ) -> Iterator[Union[CNot, PhaseShift, U3]]:
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


def translate_cz_to_zz(gate: CZ) -> Iterator[Union[ZZ, S_H]]:
    """Convert CZ gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    yield ZZ(0.5, q0, q1)
    yield S_H(q0)
    yield S_H(q1)


def translate_czpow_to_zz(gate: CZPow) -> Iterator[Union[ZZ, ZPow]]:
    """Convert a CZPow gate to a ZZ based circuit."""
    t = gate.param("t")
    q0, q1 = gate.qubits
    yield ZZ(-t / 2, q0, q1)
    yield ZPow(t / 2, q0)
    yield ZPow(t / 2, q1)


def translate_czpow_to_cphase(gate: CZPow) -> Iterator[CPhase]:
    """Convert a CZPow gate to CPhase."""
    theta = gate.param("t") * var.PI
    yield CPhase(theta, *gate.qubits)


def translate_cphase_to_czpow(gate: CPhase) -> Iterator[CZPow]:
    """Convert a CPhase gate to a CZPow."""
    (theta,) = gate.params
    t = theta / var.PI
    yield CZPow(t, *gate.qubits)


# TODO: cphase to fsim
# TODO: fsim specialize


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


def translate_ecp_to_can(gate: ECP) -> Iterator[Can]:
    """Translate an ECP gate to a Canonical gate"""
    yield Can(1 / 2, 1 / 4, 1 / 4, *gate.qubits)


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


def translate_exch_to_can(gate: Exch) -> Iterator[Can]:
    """Convert an exchange gate to a canonical based circuit"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(t, t, t, q0, q1)


def translate_exch_to_xy_zz(gate: Exch) -> Iterator[Union[XY, ZZ]]:
    """Convert an exchange gate to XY and ZZ gates"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield XY(t, q0, q1)
    yield ZZ(t, q0, q1)


def translate_fsim_to_xy_cz(gate: FSim) -> Iterator[Union[XY, CZ]]:
    """Convert the Cirq's FSim  gate to a canonical gate"""
    q0, q1 = gate.qubits
    theta, phi = gate.params

    yield XY(theta / var.PI, q0, q1)
    yield CZ(q0, q1) ** (-phi / var.PI)


def translate_fswap(gate: FSwap) -> Iterator[Union[Swap, CZ]]:
    """Translate fSwap gate to Swap and CV"""
    q0, q1 = gate.qubits
    yield Swap(q0, q1)
    yield CZ(q0, q1)


def translate_fswappow(gate: FSwapPow) -> Iterator[Union[Exch, CZPow]]:
    """Translate fSwap gate to XY and CVPow"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Swap(q0, q1) ** t
    yield CZ(q0, q1) ** t


# TODO: Other givens deke
def translate_givens_to_xy(gate: Givens) -> Iterator[Union[XY, T, T_H]]:
    """Convert a Givens  gate to an XY gate"""
    q0, q1 = gate.qubits
    (theta,) = gate.params

    yield T_H(q0)
    yield T(q1)
    yield XY(theta / var.PI, q0, q1)
    yield T(q0)
    yield T_H(q1)


def translate_iswap_to_can(gate: ISwap) -> Iterator[Union[Can, X]]:
    """Convert ISwap gate to a canonical gate within the Weyl chamber."""
    q0, q1 = gate.qubits
    yield X(q0)
    yield Can(0.5, 0.5, 0, q0, q1)
    yield X(q1)


def translate_iswap_to_swap_cz(gate: ISwap) -> Iterator[Union[Swap, CZ, S]]:
    """Convert ISwap gate to a Swap, CZ based circuit."""
    q0, q1 = gate.qubits
    yield Swap(q0, q1)
    yield CZ(q0, q1)
    yield S(q0)
    yield S(q1)


def translate_iswap_to_sqrtiswap(gate: ISwap) -> Iterator[SqrtISwap]:
    """Translate iswap gate to square-root iswaps"""
    q0, q1 = gate.qubits
    yield SqrtISwap(q0, q1)
    yield SqrtISwap(q0, q1)


def translate_iswap_to_xy(gate: ISwap) -> Iterator[Union[XY]]:
    """Convert ISwap gate to a XY gate."""
    q0, q1 = gate.qubits
    yield XY(-0.5, q0, q1)


def translate_pswap_to_canonical(gate: PSwap) -> Iterator[Union[Can, Y]]:
    """Translate parametric Swap to a canonical circuit"""

    q0, q1 = gate.qubits
    (theta,) = gate.params
    t = 0.5 - theta / var.PI
    yield Y(q0)
    yield Can(0.5, 0.5, t, q0, q1)
    yield Y(q1)


def translate_rzz_to_cnot(gate: RZZ) -> Iterator[Union[CNot, PhaseShift, U3]]:
    """Translate QASM's RZZ gate to standard gates"""
    q0, q1 = gate.qubits
    (theta,) = gate.params
    yield CNot(q0, q1)
    yield PhaseShift(theta, q1)
    yield CNot(q0, q1)


def translate_sqrtiswap_to_sqrtiswap_h(
    gate: SqrtISwap,
) -> Iterator[Union[SqrtISwap_H, Z]]:
    """Translate square-root-iswap to its inverse."""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield SqrtISwap_H(q0, q1)
    yield Z(q0)


def translate_sqrtiswap_h_to_can(gate: SqrtISwap_H) -> Iterator[Can]:
    """Translate square-root iswap to canonical"""
    yield Can(1 / 4, 1 / 4, 0, *gate.qubits)


def translate_sqrtiswap_h_to_sqrtiswap(
    gate: SqrtISwap_H,
) -> Iterator[Union[SqrtISwap, Z]]:
    """Translate square-root-iswap to it's inverse."""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield SqrtISwap(q0, q1)
    yield Z(q0)


def translate_sqrtswap_to_can(gate: SqrtSwap) -> Iterator[Can]:
    """Translate square-root swap to canonical"""
    yield Can(1 / 4, 1 / 4, 1 / 4, *gate.qubits)


def translate_sqrtswap_h_to_can(gate: SqrtSwap_H) -> Iterator[Can]:
    """Translate inv. square-root swap to canonical"""
    yield Can(-1 / 4, -1 / 4, -1 / 4, *gate.qubits)


def translate_swap_to_cnot(gate: Swap) -> Iterator[CNot]:
    """Convert a Swap gate to a circuit with 3 CNots."""
    q0, q1 = gate.qubits
    yield CNot(q0, q1)
    yield CNot(q1, q0)
    yield CNot(q0, q1)


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


def translate_swap_to_iswap_cz(gate: Swap) -> Iterator[Union[ISwap, CZ, S_H]]:
    """Convert ISwap gate to a Swap, CZ based circuit."""
    q0, q1 = gate.qubits
    yield S_H(q0)
    yield S_H(q1)
    yield CZ(q0, q1)
    yield ISwap(q0, q1)


def translate_sycamore_to_fsim(gate: Sycamore) -> Iterator[FSim]:
    """Convert a Sycamore gate to an FSim gate"""
    yield FSim(var.PI / 2, var.PI / 6, *gate.qubits)


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


def translate_w_to_ch_cnot(gate: W) -> Iterator[Union[CNot, CH]]:
    """Translate W gate to controlled-Hadamard and CNot."""
    # From https://arxiv.org/pdf/1505.06552.pdf
    q0, q1 = gate.qubits

    yield CNot(q1, q0)
    yield CNot(q0, q1)
    yield CH(q0, q1)
    yield CNot(q0, q1)
    yield CNot(q1, q0)


def translate_xx_to_can(gate: XX) -> Iterator[Union[Can]]:
    """Convert an XX gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(t, 0, 0, q0, q1)


def translate_xx_to_zz(gate: XX) -> Iterator[Union[H, ZZ]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield H(q0)
    yield H(q1)
    yield ZZ(t, q0, q1)
    yield H(q0)
    yield H(q1)


def translate_xy_to_can(gate: XY) -> Iterator[Can]:
    """Convert XY gate to a canonical gate."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(t, t, 0, q0, q1)


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


def translate_yy_to_can(gate: YY) -> Iterator[Union[Can]]:
    """Convert an YY gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(0, t, 0, q0, q1)


def translate_yy_to_zz(gate: YY) -> Iterator[Union[XPow, ZZ]]:
    """Convert a YY gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield XPow(0.5, q0)
    yield XPow(0.5, q1)
    yield ZZ(t, q0, q1)
    yield XPow(-0.5, q0)
    yield XPow(-0.5, q1)


def translate_zz_to_can(gate: ZZ) -> Iterator[Union[Can]]:
    """Convert an ZZ gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield Can(0, 0, t, q0, q1)


def translate_zz_to_cnot(gate: ZZ) -> Iterator[Union[CNot, ZPow]]:
    """Convert a ZZ gate to a CNot based circuit"""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield CNot(q0, q1)
    yield ZPow(t, q1)
    yield CNot(q0, q1)


def translate_zz_to_xx(gate: ZZ) -> Iterator[Union[H, XX]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield H(q0)
    yield H(q1)
    yield XX(t, q0, q1)
    yield H(q0)
    yield H(q1)


def translate_zz_to_yy(gate: ZZ) -> Iterator[Union[XPow, YY]]:
    """Convert a YY gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.param("t")
    yield XPow(-0.5, q0)
    yield XPow(-0.5, q1)
    yield YY(t, q0, q1)
    yield XPow(0.5, q0)
    yield XPow(0.5, q1)


# 3-qubit gates


def translate_ccix_to_cnot(gate: CCiX) -> Iterator[Union[CNot, H, T, T_H]]:
    """Decompose doubly-controlled iX-gate to 4 CNots.

    ::

        ───●────     ───────────────●────────────────●────────────
           │                        │                │
        ───●────  =  ───────●───────┼────────●───────┼────────────
           │                │       │        │       │
        ───iX───     ───H───X───T───X───T⁺───X───T───X───T⁺───H───


    Refs:
         Nielsen and Chuang (Figure 4.9)

    """
    # Kudos: Adapted from Quipper
    # https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html
    q0, q1, q2 = gate.qubits

    yield H(q2)
    yield CNot(q1, q2)
    yield T(q2)
    yield CNot(q0, q2)
    yield T(q2).H
    yield CNot(q1, q2)
    yield T(q2)
    yield CNot(q0, q2)
    yield T(q2).H
    yield H(q2)


def translate_ccix_to_cnot_adjacent(gate: CCiX) -> Iterator[Union[CNot, H, T, T_H]]:
    """Decompose doubly-controlled iX-gate to 8 CNots, respecting adjacency.

    ::

        ───●────     ───────────X───T⁺───────X────T───X───────X───
           │                    │            │        │       │
        ───●────  =  ───────X───●───T────X───●────X───●───X───●───
           │                │            │        │       │
        ───iX───     ───H───●────────────●───T⁺───●───────●───H───

    Refs:
          http://arxiv.org/abs/1210.0974, Figure 10
    """
    # Kudos: Adapted from 1210.0974, via Quipper
    q0, q1, q2 = gate.qubits

    yield H(q2)
    yield CNot(q2, q1)
    yield CNot(q1, q0)
    yield T(q0).H
    yield T(q1)
    yield CNot(q2, q1)
    yield CNot(q1, q0)
    yield T(q0)
    yield T(q2).H
    yield CNot(q2, q1)
    yield CNot(q1, q0)
    yield CNot(q2, q1)
    yield CNot(q1, q0)
    yield H(q2)


def translate_ccnot_to_ccz(gate: CCNot) -> Iterator[Union[H, CCZ]]:
    """Convert CCNot (Toffoli) gate to CCZ using Hadamards
    ::

        ───●───     ───────●───────
           │               │
        ───●───  =  ───────●───────
           │               │
        ───X───     ───H───●───H───

    """
    q0, q1, q2 = gate.qubits

    yield H(q2)
    yield CCZ(q0, q1, q2)
    yield H(q2)


def translate_ccnot_to_cnot(gate: CCNot) -> Iterator[Union[H, T, T_H, CNot]]:
    """Standard decomposition of CCNot (Toffoli) gate into six CNot gates.
    ::

        ───●───     ────────────────●────────────────●───●───T────●───
           │                        │                │   │        │
        ───●───  =  ───────●────────┼───────●───T────┼───X───T⁺───X───
           │               │        │       │        │
        ───X───     ───H───X───T⁺───X───T───X───T⁺───X───T───H────────

    Refs:
        M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum
        Information, Cambridge University Press (2000).
    """
    q0, q1, q2 = gate.qubits

    yield H(q2)
    yield CNot(q1, q2)
    yield T_H(q2)
    yield CNot(q0, q2)
    yield T(q2)
    yield CNot(q1, q2)
    yield T_H(q2)
    yield CNot(q0, q2)
    yield T(q1)
    yield T(q2)
    yield H(q2)
    yield CNot(q0, q1)
    yield T(q0)
    yield T_H(q1)
    yield CNot(q0, q1)


def translate_ccnot_to_cnot_AMMR(gate: CCNot) -> Iterator[Union[H, T, T_H, CNot]]:
    """Depth 9 decomposition of CCNot (Toffoli) gate into 7 CNot gates.
    ::

        ───●───     ───T───X───────●────────●────────T⁺───●───X───
           │               │       │        │             │   │
        ───●───  =  ───T───●───X───┼───T⁺───X───T⁺───X────┼───●───
           │                   │   │                 │    │
        ───X───     ───H───T───●───X────────────T────●────X───H───


    Refs:
        M. Amy, D. Maslov, M. Mosca, and M. Roetteler, A meet-in-the-middle
        algorithm for fast synthesis of depth-optimal quantum circuits, IEEE
        Transactions on Computer-Aided Design of Integrated Circuits and
        Systems 32(6):818-830.  http://arxiv.org/abs/1206.0758.
    """
    # Kudos: Adapted from QuipperLib
    # https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html

    q0, q1, q2 = gate.qubits

    yield H(q2)
    yield T(q0)
    yield T(q1)
    yield T(q2)
    yield CNot(q1, q0)
    yield CNot(q2, q1)
    yield CNot(q0, q2)
    yield T_H(q1)
    yield T(q2)
    yield CNot(q0, q1)
    yield T_H(q0)
    yield T_H(q1)
    yield CNot(q2, q1)
    yield CNot(q0, q2)
    yield CNot(q1, q0)
    yield H(q2)


def translate_ccnot_to_cv(gate: CCNot) -> Iterator[Union[CV, CV_H, CNot]]:
    """Decomposition of CCNot (Toffoli) gate into 3 CV and 2 CNots.
    ::

        ───●───     ───────●────────●───●───
           │               │        │   │
        ───●───  =  ───●───X───●────X───┼───
           │           │       │        │
        ───X───     ───V───────V⁺───────V───

    Refs:
         Barenco, Adriano; Bennett, Charles H.; Cleve, Richard;
         DiVincenzo, David P.; Margolus, Norman; Shor, Peter; Sleator, Tycho;
         Smolin, John A.; Weinfurter, Harald (Nov 1995). "Elementary gates for
         quantum computation". Physical Review A. American Physical Society. 52
         (5): 3457–3467. arXiv:quant-ph/9503016
    """

    q0, q1, q2 = gate.qubits
    yield CV(q1, q2)
    yield CNot(q0, q1)
    yield CV_H(q1, q2)
    yield CNot(q0, q1)
    yield CV(q0, q2)


def translate_ccxpow_to_cnotpow(gate: CCXPow) -> Iterator[Union[CNot, CNotPow]]:
    """Decomposition of powers of CCNot gates to powers of CNot gates."""
    q0, q1, q2 = gate.qubits
    (t,) = gate.params
    yield CNotPow(t / 2, q1, q2)
    yield CNot(q0, q1)
    yield CNotPow(-t / 2, q1, q2)
    yield CNot(q0, q1)
    yield CNotPow(t / 2, q0, q2)


def translate_ccz_to_adjacent_cnot(gate: CCZ) -> Iterator[Union[T, CNot, T_H]]:
    """Decomposition of CCZ gate into 8 CNot gates.
    Respects linear adjacency of qubits.
    ::

        ───●───     ───T───●────────────●───────●────────●────────
           │               │            │       │        │
        ───●───  =  ───T───X───●───T⁺───X───●───X────●───X────●───
           │                   │            │        │        │
        ───●───     ───T───────X───T────────X───T⁺───X───T⁺───X───
    """
    # Kudos: adapted from Cirq

    q0, q1, q2 = gate.qubits

    yield T(q0)
    yield T(q1)
    yield T(q2)

    yield CNot(q0, q1)
    yield CNot(q1, q2)

    yield T_H(q1)
    yield T(q2)

    yield CNot(q0, q1)
    yield CNot(q1, q2)

    yield T_H(q2)

    yield CNot(q0, q1)
    yield CNot(q1, q2)

    yield T_H(q2)

    yield CNot(q0, q1)
    yield CNot(q1, q2)


def translate_ccz_to_ccnot(gate: CCZ) -> Iterator[Union[H, CCNot]]:
    """Convert  CCZ gate to CCNot gate using Hadamards
    ::

        ───●───     ───────●───────
           │               │
        ───●───  =  ───────●───────
           │               │
        ───X───     ───H───●───H───

    """
    q0, q1, q2 = gate.qubits

    yield H(q2)
    yield CCNot(q0, q1, q2)
    yield H(q2)


def translate_cswap_to_ccnot(gate: CSwap) -> Iterator[Union[CNot, CCNot]]:
    """Convert a CSwap gate to a circuit with a CCNot and 2 CNots"""
    q0, q1, q2 = gate.qubits
    yield CNot(q2, q1)
    yield CCNot(q0, q1, q2)
    yield CNot(q2, q1)


def translate_cswap_to_cnot(
    gate: CSwap,
) -> Iterator[Union[CNot, H, T, T_H, V, V_H, S, S_H]]:
    """Adjacency respecting decomposition of CSwap to CNot.
    Assumes that q0 (control) is next to q1 is next to q2.
    ::

        ───●───     ───T───────────●────────────●───────●────────────────●───────────────────────
           │                       │            │       │                │
        ───x───  =  ───X───T───────X───●───T⁺───X───●───X────●───X^1/2───X───●───X^1/2───────────
           │           │               │            │        │               │
        ───x───     ───●───Y^3/2───T───X───T────────X───T⁺───X───T⁺──────────X───S───────X^3/2───
    """  # noqa: W291, E501
    # Kudos: Adapted from Cirq
    q0, q1, q2 = gate.qubits

    yield CNot(q2, q1)
    yield H(q2)
    yield S(q2).H
    yield T(q0)
    yield T(q1)
    yield T(q2).H
    yield CNot(q0, q1)
    yield CNot(q1, q2)
    yield T(q1).H
    yield T(q2)
    yield CNot(q0, q1)
    yield CNot(q1, q2)
    yield T(q2).H
    yield CNot(q0, q1)
    yield CNot(q1, q2)
    yield T(q2).H
    yield V(q1)
    yield CNot(q0, q1)
    yield CNot(q1, q2)
    yield S(q2)
    yield V(q1)
    yield V(q2).H


def translate_cswap_inside_to_cnot(
    gate: CSwap,
) -> Iterator[Union[CNot, H, T, T_H, V, V_H, S, S_H]]:
    """Adjacency respecting decomposition of centrally controlled CSwap to CNot.
    Assumes that q0 (control) is next to q1 is next to q2.
    ::

        ───x───      ───●───X───T────────────────────●────────────●────────────X───────●───V⁺────────────
           │            │   │                        │            │            │       │
        ───●───  =   ───X───●───X───────────●───T⁺───X───●───T────X───●───V────●───●───X───●─────────────
           │                    │           │            │            │            │       │
        ───x───      ───────────●───H───T───X───T────────X───T⁺───────X───T⁺───────X───────X────H───S⁺───
    """  # noqa: W291, E501
    # Kudos: Adapted from Cirq
    q0, q1, q2 = gate.qubits

    yield CNot(q1, q0)
    yield CNot(q0, q1)
    yield CNot(q2, q0)
    yield H(q2)
    yield T(q2)
    yield CNot(q0, q2)
    yield T(q1)
    yield T_H(q0)
    yield T(q2)
    yield CNot(q1, q0)
    yield CNot(q0, q2)
    yield T(q0)
    yield T_H(q2)
    yield CNot(q1, q0)
    yield CNot(q0, q2)
    yield V(q0)
    yield T_H(q2)
    yield CNot(q0, q1)
    yield CNot(q0, q2)
    yield CNot(q1, q0)
    yield CNot(q0, q2)
    yield H(q2)
    yield S_H(q2)
    yield V_H(q1)


def translate_deutsch_to_barenco(gate: Deutsch) -> Iterator[Barenco]:
    """Translate a 3-qubit Deutsch gate to five 2-qubit Barenco gates.

    Ref:
        A Universal Two–Bit Gate for Quantum Computation, A. Barenco (1995)
        https://arxiv.org/pdf/quant-ph/9505016.pdf  :cite:`Barenco1995a`
    """
    q0, q1, q2 = gate.qubits
    (theta,) = gate.params
    yield Barenco(0, var.PI / 4, theta / 2, q1, q2)
    yield Barenco(0, var.PI / 4, theta / 2, q0, q2)
    yield Barenco(0, var.PI / 2, var.PI / 2, q0, q1)
    yield Barenco(var.PI, -var.PI / 4, theta / 2, q1, q2)
    yield Barenco(0, var.PI / 2, var.PI / 2, q0, q1)


TRANSLATORS: Dict[str, Callable] = {}
TRANSLATORS = {
    name: func for name, func in globals().items() if name.startswith("translate_")
}


# Note: Translators auto-magically added
__all__ = ("circuit_translate", "select_translators", "TRANSLATORS") + tuple(
    TRANSLATORS.keys()
)

# fin
