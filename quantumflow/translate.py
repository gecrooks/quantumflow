
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Translate, transform, and comPIle circuits.
"""

# TODO: Split into separate modules
# TODO: Put translations into alphabetic order so they can be found

from typing import (
    Sequence, Iterator, Union, Callable, Tuple, Dict, Type, List, Iterable)
import numpy as np

import sympy
from sympy import Symbol
from sympy import pi as PI

from .ops import Gate
from .circuits import Circuit

from .gates import (
    # one qubit
    I, X, Y, Z, H, S, T, PhaseShift, RX, RY, RZ,
    RN,
    TX, TY, TZ, TH,
    S_H, T_H, V, V_H, SqrtY, SqrtY_H, Ph,
    # two qubit
    Barenco, CZ, CZPow, CNOT, ECP, SWAP, ISWAP, XY,
    CPHASE00, CPHASE01, CPHASE10, CPHASE,
    CrossResonance, FSwap, FSwapPow, Givens, PSWAP, Can, XX, YY, ZZ, EXCH,
    CNotPow,
    CV, CV_H, CY, CYPow, CH, B, SqrtISwap, SqrtISwap_H, SqrtSwap, SqrtSwap_H,
    W,
    # three qubit
    CCNOT, CCXPow, CSWAP, CCZ, Deutsch, IDEN, CCiX,
    # qasm gates
    U3, U2, CU3, CRZ, RZZ,
    # cirq gates
    PhasedX, PhasedXPow, FSim, Sycamore
    )

# from .variables import variable_is_symbolic


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
    return trans.__annotations__['gate']


def translation_target_gates(trans: Callable) -> Tuple[Type[Gate]]:

    # FIXME: Fails with KeyError if no return type annotation.
    # Add a more informative error message.
    ret = trans.__annotations__['return'].__args__[0]
    if hasattr(ret, '__args__'):  # Union
        gates = ret.__args__
    else:
        gates = (ret,)

    return gates


# TODO: Rename
def select_translators(target_gates: Iterable[Type[Gate]],
                       translators: Iterable[Callable] = None
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


def circuit_translate(circ: Circuit,
                      translators: Sequence = None,
                      targets: Sequence[Type[Gate]] = None,
                      recurse: bool = True) -> Circuit:
    """Apply a collection of translations to each gate in a circuit.
    If recurse, then apply translations to output of translations
    until translationally invariant.
    """
    if translators is not None and targets is not None:
        raise ValueError('Specify either targets or translators, not both')

    gates = list(reversed(list(circ)))
    translated = Circuit()

    if translators is None:
        if targets is None:
            targets = [Can, TX, TZ, I, Ph]
        translators = select_translators(targets)

    # Use type annotations to do dynamic dispatch
    gateclass_translation = {translation_source_gate(trans): trans
                             for trans in translators}

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

    return translated


# 1-qubit gates

def translate_x_to_tx(gate: X) -> Iterator[TX]:
    """Translate X gate to TX"""
    q0, = gate.qubits
    yield TX(1, q0)


def translate_y_to_ty(gate: Y) -> Iterator[TY]:
    """Translate Y gate to TY"""
    q0, = gate.qubits
    yield TY(1, q0)


def translate_z_to_tz(gate: Z) -> Iterator[TZ]:
    """Translate Z gate to TZ"""
    q0, = gate.qubits
    yield TZ(1, q0)


def translate_s_to_tz(gate: S) -> Iterator[TZ]:
    """Translate S gate to TZ"""
    q0, = gate.qubits
    yield TZ(0.5, q0)


def translate_t_to_tz(gate: T) -> Iterator[TZ]:
    """Translate T gate to TZ"""
    q0, = gate.qubits
    yield TZ(0.25, q0)


def translate_invs_to_tz(gate: S_H) -> Iterator[TZ]:
    """Translate S.H gate to TZ"""
    q0, = gate.qubits
    yield TZ(-0.5, q0)


def translate_invt_to_tz(gate: T_H) -> Iterator[TZ]:
    """Translate inverse T gate to RZ (a quil standard gate)"""
    q0, = gate.qubits
    yield TZ(-0.25, q0)


def translate_rx_to_tx(gate: RX) -> Iterator[TX]:
    """Translate RX gate to TX"""
    q0, = gate.qubits
    theta, = gate.parameters()
    t = theta/PI
    yield TX(t, q0)


def translate_ry_to_ty(gate: RY) -> Iterator[TY]:
    """Translate RY gate to TY"""
    q0, = gate.qubits
    theta, = gate.parameters()
    t = theta/PI
    yield TY(t, q0)


def translate_rz_to_tz(gate: RZ) -> Iterator[TZ]:
    """Translate RZ gate to TZ"""
    q0, = gate.qubits
    theta, = gate.parameters()
    t = theta/PI
    yield TZ(t, q0)


def translate_rn_to_rz_ry(gate: RN) -> Iterator[Union[RZ, RY]]:
    """Translate RN Bloch rotation to Rz Ry Rz Ry Rz.

    Refs:
        http://www.vcpc.univie.ac.at/~ian/hotlist/qc/talks/bloch-sphere-rotations.pdf
    """
    q0, = gate.qubits
    theta, nx, ny, nz = gate.parameters()

    ang_y = np.arccos(nz)
    ang_z = np.arctan2(ny, nx)

    yield RZ(-ang_z, q0)
    yield RY(-ang_y, q0)
    yield RZ(theta, q0)
    yield RY(ang_y, q0)
    yield RZ(ang_z, q0)


def translate_phase_to_rz(gate: PhaseShift) -> Iterator[RZ]:
    """Translate PHASE gate to RZ (ignoring global phase)"""
    q0, = gate.qubits
    theta = gate.params['theta']
    yield RZ(theta, q0)


def translate_sqrty_to_ty(gate: SqrtY) -> Iterator[TY]:
    """Translate sqrt-Y gate to TY"""
    yield from gate.decompose()


def translate_sqrty_h_to_ty(gate: SqrtY_H) -> Iterator[TY]:
    """Translate sqrt-Y gate to TY"""
    yield from gate.decompose()


def translate_tx_to_rx(gate: TX) -> Iterator[RX]:
    """Translate TX gate to RX"""
    q0, = gate.qubits
    theta = gate.params['t'] * PI
    yield RX(theta, q0)


def translate_ty_to_ry(gate: TY) -> Iterator[RY]:
    """Translate TY gate to RY"""
    q0, = gate.qubits
    theta = gate.params['t'] * PI
    yield RY(theta, q0)


def translate_tz_to_rz(gate: TZ) -> Iterator[RZ]:
    """Translate TZ gate to RZ"""
    q0, = gate.qubits
    theta = gate.params['t'] * PI
    yield RZ(theta, q0)


def translate_ty_to_xzx(gate: TY) -> Iterator[Union[TX, TZ]]:
    """Translate TY gate to TX and TZ gates"""
    q0, = gate.qubits
    t = gate.params['t']
    yield TX(0.5, q0)
    yield TZ(t, q0)
    yield TX(-0.5, q0)


def translate_tx_to_zyz(gate: TX) -> Iterator[Union[TY, S, S_H]]:
    """Translate TX gate to S and TY gates"""
    q0, = gate.qubits
    t = gate.params['t']
    yield S(q0)
    yield TY(t, q0)
    yield S_H(q0)


def translate_tz_to_xyx(gate: TZ) -> Iterator[Union[TY, V, V_H]]:
    """Translate TZ gate to V and TY gates"""
    q0, = gate.qubits
    t = gate.params['t']
    yield V_H(q0)
    yield TY(t, q0)
    yield V(q0)


def translate_phased_x_to_zxz(gate: PhasedX) -> Iterator[Union[X, TZ]]:
    """Translate TY gate to TX and TZ gates"""
    q0, = gate.qubits
    p = gate.params['p']
    yield TZ(-p, q0)
    yield X(q0)
    yield TZ(p, q0)


def translate_phased_tx_to_zxz(gate: PhasedXPow) -> Iterator[Union[TX, TZ]]:
    """Translate TY gate to TX and TZ gates"""
    q0, = gate.qubits
    p, t = gate.parameters()
    yield TZ(-p, q0)
    yield TX(t, q0)
    yield TZ(p, q0)


def translate_v_to_tx(gate: V) -> Iterator[TX]:
    """Translate V gate to TX"""
    q0, = gate.qubits
    yield TX(0.5, q0)


def translate_invv_to_tx(gate: V_H) -> Iterator[TX]:
    """Translate V_H gate to TX"""
    q0, = gate.qubits
    yield TX(-0.5, q0)


def translate_th_to_tx(gate: TH) -> Iterator[Union[TX, H, S, T, S_H, T_H]]:
    """Translate powers of the Hadamard gate to TX and TY"""
    q0, = gate.qubits
    t, = gate.parameters()

    yield S(q0)
    yield H(q0)
    yield T(q0)
    yield TX(t, q0)
    yield T(q0).H
    yield H(q0)
    yield S(q0).H


def translate_ty_to_zxz(gate: TY) -> Iterator[Union[TX, S, S_H]]:
    """Translate TY gate to TZ and TX gates"""
    q0, = gate.qubits
    t = gate.params['t']
    yield S_H(q0)
    yield TX(t, q0)
    yield S(q0)


def translate_tx_to_zxzxz(gate: TX) -> Iterator[Union[TX, TZ]]:
    """Convert an arbitrary power of a Pauli-X gate to Z and V gates"""
    q0, = gate.qubits
    t = gate.params['t']

    if t == 0.5 or t == -0.5:   # FIXME: isclose
        yield gate
        return

    yield TZ(0.5, q0)
    yield TX(0.5, q0)
    yield TZ(t, q0)
    yield TX(-0.5, q0)
    yield TZ(-0.5, q0)


def translate_hadamard_to_zxz(gate: H) -> Iterator[Union[TX, TZ]]:
    """Convert a Hadamard gate to a circuit with TZ and TX gates."""
    q0, = gate.qubits
    yield TZ(0.5, q0)
    yield TX(0.5, q0)
    yield TZ(0.5, q0)


def translate_u3_to_zyz(gate: U3) -> Iterator[Union[RZ, RY]]:
    """Translate QASMs U3 gate to RZ and RY"""
    q0, = gate.qubits
    theta, phi, lam = gate.parameters()
    yield RZ(lam, q0)
    yield RY(theta, q0)
    yield RZ(phi, q0)


def translate_u2_to_zyz(gate: U2) -> Iterator[Union[RZ, RY]]:
    """Translate QASMs U2 gate to RZ and RY"""
    q0, = gate.qubits
    phi, lam = gate.parameters()
    yield RZ(lam, q0)
    yield RY(PI/2, q0)
    yield RZ(phi, q0)


def translate_tx_to_hzh(gate: TX) -> Iterator[Union[H, TZ]]:
    """Convert a TX gate to a circuit with Hadamard and TZ gates"""
    q0, = gate.qubits
    t, = gate.parameters()
    yield H(q0)
    yield TZ(t, q0)
    yield H(q0)


# 2-qubit gates


def translate_b_to_can(gate: B) -> Iterator[Union[Can, Y, Z]]:
    """Translate B gate to Canonical gate"""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield Y(q1)
    yield Can(1/2, 1/4, 0, q0, q1)
    yield Y(q0)
    yield Z(q1)


def translate_barenco_to_xx(gate: Barenco) -> Iterator[Union[XX, TY, TZ]]:
    """Translate a Barenco gate to XX plus local gates"""
    phi, alpha, theta = gate.parameters()

    ct = theta/PI
    ca = alpha/PI
    cp = phi/PI

    q0, q1 = gate.qubits

    yield TZ(-1/2+cp, q0)
    yield TY(1/2, q0)
    yield TZ(-1, q0)
    yield TZ(1/2-cp, q1)
    yield TY(1/2, q1)
    yield TZ(3/2, q1)
    yield XX(ct, q0, q1)
    yield TY(1/2, q0)
    yield TZ(3/2+ca-cp, q0)
    yield TZ(-1/2, q1)
    yield TY(1/2-ct, q1)
    yield TZ(-3/2+cp, q1)


def translate_can_to_cnot(gate: Can) -> \
        Iterator[Union[CNOT, S, S_H, TX, TY, TZ, V, Z]]:
    """Translate canonical gate to 3 CNOTS, using the Kraus-Cirac decomposition.

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

    # Note: sign flip on central TZ, TY, TY because of differing sign
    # conventions for Canonical.
    tx, ty, tz = gate.parameters()
    q0, q1 = gate.qubits

    if not isinstance(tz, Symbol) and np.isclose(tz, 0.0):
        # If we know tz is close to zero we only need two CNOT gates
        yield V(q0)
        yield Z(q0)
        yield V(q1)
        yield Z(q1)
        yield CNOT(q0, q1)
        yield TX(tx, q0)
        yield TZ(ty, q1)
        yield CNOT(q0, q1)
        yield V(q0)
        yield Z(q0)
        yield V(q1)
        yield Z(q1)
    else:
        yield S(q1)
        yield CNOT(q1, q0)
        yield TZ(tz-0.5, q0)
        yield TY(-0.5+tx, q1)
        yield CNOT(q0, q1)
        yield TY(-ty+0.5, q1)
        yield CNOT(q1, q0)
        yield S_H(q0)


def translate_can_to_xx_yy_zz(gate: Can) -> Iterator[Union[XX, YY, ZZ]]:
    """Convert a canonical gate to a circuit with XX, YY, and ZZ gates."""
    tx, ty, tz = gate.parameters()
    q0, q1 = gate.qubits

    if isinstance(tx, Symbol) or not np.isclose(tx, 0.0):
        yield XX(tx, q0, q1)
    if isinstance(tx, Symbol) or not np.isclose(ty, 0.0):
        yield YY(ty, q0, q1)
    if isinstance(tx, Symbol) or not np.isclose(tz, 0.0):
        yield ZZ(tz, q0, q1)


def translate_ch_to_cpt(gate: CH) -> Iterator[Union[CNOT, S, T, S_H, T_H, H]]:
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
    yield CNOT(q0, q1)
    yield T_H(q1)
    yield H(q1)
    yield S_H(q1)


def translate_cnot_to_cz(gate: CNOT) -> Iterator[Union[H, CZ]]:
    """Convert CNOT gate to a CZ based circuit."""
    q0, q1 = gate.qubits
    yield H(q1)
    yield CZ(q0, q1)
    yield H(q1)


def translate_cnot_to_sqrtiswap(gate: CNOT) \
        -> Iterator[Union[SqrtISwap_H, X, S_H, H]]:
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


def translate_cnot_to_sqrtswap(gate: CNOT) \
        -> Iterator[Union[SqrtSwap, TY, TZ, Z]]:
    """Translate square-root swap to canonical"""
    # https://qipc2011.ethz.ch/uploads/Schoolpresentations/berghaus2011_DiVincenzo.pdf
    q0, q1 = gate.qubits

    yield Y(q1)**0.5
    yield SqrtSwap(q0, q1)
    yield Z(q0)
    yield SqrtSwap(q0, q1)
    yield Z(q0)**-0.5
    yield Z(q1)**-0.5
    yield Y(q1)**-0.5


def translate_cnot_to_xx(gate: CNOT) -> Iterator[Union[XX, H, S_H]]:
    """Convert CNOT to XX gate"""
    # TODO: simplify 1-qubit gates
    q0, q1 = gate.qubits
    yield H(q0)
    yield XX(0.5, q0, q1)
    yield H(q0)
    yield H(q1)
    yield S(q0).H
    yield S(q1).H
    yield H(q1)


def translate_cy_to_cnot(gate: CY) -> Iterator[Union[CNOT, S, S_H]]:
    """Translate CY to CNOT (CX)"""
    q0, q1 = gate.qubits
    yield S_H(q1)
    yield CNOT(q0, q1)
    yield S(q1)


def translate_cypow_to_cxpow(gate: CYPow) -> Iterator[Union[CNotPow, S, S_H]]:
    """Translate powers of CY to powers of CNOT (CX)"""
    yield from gate.decompose()


def translate_cphase_to_zz(gate: CPHASE) -> Iterator[Union[ZZ, TZ]]:
    """Convert a CPHASE gate to a ZZ based circuit."""
    t = - gate.params['theta'] / (2 * PI)
    q0, q1 = gate.qubits
    yield ZZ(t, q0, q1)
    yield TZ(-t, q0)
    yield TZ(-t, q1)


def translate_cphase00_to_zz(gate: CPHASE00) -> Iterator[Union[X, ZZ, TZ]]:
    """Convert a CPHASE00 gate to a ZZ based circuit."""
    t = - gate.params['theta'] / (2 * PI)
    q0, q1 = gate.qubits
    yield X(q0)
    yield X(q1)
    yield ZZ(t, q0, q1)
    yield TZ(-t, q0)
    yield TZ(-t, q1)
    yield X(q0)
    yield X(q1)


def translate_cphase01_to_zz(gate: CPHASE01) -> Iterator[Union[X, ZZ, TZ]]:
    """Convert a CPHASE01 gate to a ZZ based circuit."""
    t = - gate.params['theta'] / (2 * PI)
    q0, q1 = gate.qubits
    yield X(q0)
    yield ZZ(t, q0, q1)
    yield TZ(-t, q0)
    yield TZ(-t, q1)
    yield X(q0)


def translate_cphase10_to_zz(gate: CPHASE10) -> Iterator[Union[X, ZZ, TZ]]:
    """Convert a CPHASE10 gate to a ZZ based circuit."""
    t = - gate.params['theta'] / (2 * PI)
    q0, q1 = gate.qubits

    yield X(q1)
    yield ZZ(t, q0, q1)
    yield TZ(-t, q0)
    yield TZ(-t, q1)
    yield X(q1)


def translate_cross_resonance_to_xx(gate: CrossResonance) \
        -> Iterator[Union[XX, TX, TY, X]]:
    """Translate a cross resonance gate to an XX based circuit"""
    s, b, c = gate.parameters()
    q0, q1 = gate.qubits

    t7 = np.arccos(((1 + b**2 * np.cos(PI * np.sqrt(1 + b**2) * s)))
                   / (1 + b**2)) / PI
    t4 = c * s
    t1 = np.arccos(np.cos(0.5*PI * np.sqrt(1 + b**2) * s)
                   / np.cos(t7*PI/2))/PI

    a = np.sin(PI * np.sqrt(1 + b**2) * s / 2)
    t7 *= sympy.sign(a) * sympy.sign(b)
    t1 *= sympy.sign(a)

    yield TX(t1, q0)
    yield TY(1.5, q0)
    yield X(q0)
    yield TX(t4, q1)
    yield XX(t7, q0, q1)
    yield TY(1.5, q0)
    yield X(q0)
    yield TX(t1, q0)


def translate_crz_to_cnot(gate: CRZ) -> Iterator[Union[CNOT, PhaseShift, U3]]:
    """Translate QASM's CRZ gate to standard gates.

    Ref:
        https://arxiv.org/pdf/1707.03429.pdf
    """
    q0, q1 = gate.qubits
    theta, = gate.parameters()

    yield PhaseShift(theta/2, q1)
    yield CNOT(q0, q1)
    yield PhaseShift(-theta/2, q1)
    yield CNOT(q0, q1)


def translate_cnotpow_to_zz(gate: CNotPow) -> Iterator[Union[ZZ, TZ, H]]:
    """Convert a controlled X^t gate to a ZZ based circuit.
    ::

        ───●─────     ───────ZZ^-t/2───Z^t/2───────
           │       =          │
        ───X^t───     ───H───ZZ^-t/2───Z^t/2───H───

    """
    t, = gate.parameters()
    q0, q1 = gate.qubits
    yield H(q1)
    yield ZZ(-t/2, q0, q1)
    yield TZ(t/2, q0)
    yield TZ(t/2, q1)
    yield H(q1)


def translate_cz_to_zz(gate: CZ) -> Iterator[Union[ZZ, S_H]]:
    """Convert CZ gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    yield ZZ(0.5, q0, q1)
    yield S_H(q0)
    yield S_H(q1)


def translate_czpow_to_zz(gate: CZPow) -> Iterator[Union[ZZ, TZ]]:
    """Convert a CZPow gate to a ZZ based circuit."""
    t = gate.params['t']
    q0, q1 = gate.qubits
    yield ZZ(-t/2, q0, q1)
    yield TZ(t/2, q0)
    yield TZ(t/2, q1)


def translate_czpow_to_cphase(gate: CZPow) -> Iterator[CPHASE]:
    """Convert a CZPow gate to CPHASE."""
    theta = gate.params['t'] * PI
    yield CPHASE(theta, * gate.qubits)


def translate_cphase_to_czpow(gate: CPHASE) -> Iterator[CZPow]:
    """Convert a CPHASE gate to a CZPow."""
    theta, = gate.parameters()
    t = theta/PI
    yield CZPow(t, * gate.qubits)

# TODO: cphase to fsim
# TODO: fsim specialize


def translate_cu3_to_cnot(gate: CU3) -> Iterator[Union[CNOT, PhaseShift, U3]]:
    """Translate QASM's CU3 gate to standard gates"""
    # Kudos: Adapted from qiskit
    # https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/cu3.py
    q0, q1 = gate.qubits
    theta, phi, lam = gate.parameters()

    yield PhaseShift((lam+phi)/2, q0)
    yield PhaseShift((lam-phi)/2, q1)
    yield CNOT(q0, q1)
    yield U3(-theta / 2, 0, -(phi+lam)/2, q1)
    yield CNOT(q0, q1)
    yield U3(theta / 2, phi, 0, q1)


def translate_cv_to_cpt(gate: CV) -> Iterator[Union[CNOT, T, T_H, H]]:
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
    yield CNOT(q1, q0)
    yield T_H(q0)
    yield T(q1)
    yield CNOT(q1, q0)
    yield H(q1)


def translate_cvh_to_cpt(gate: CV_H) -> Iterator[Union[CNOT, T, T_H, H]]:
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
    yield CNOT(q1, q0)
    yield T_H(q1)
    yield T(q0)
    yield CNOT(q1, q0)
    yield H(q1)
    yield T_H(q0)


def translate_ecp_to_can(gate: ECP) -> Iterator[Can]:
    """Translate an ECP gate to a Canonical gate"""
    yield Can(1/2, 1/4, 1/4, *gate.qubits)


def translate_ecp_to_sqrtiswap(gate: ECP) \
        -> Iterator[Union[SqrtISwap_H, TY, S, S_H]]:
    """Translate an ECP gate to a square-root-iswap sandwich"""
    q0, q1 = gate.qubits

    yield SqrtISwap().H.on(q0, q1)
    yield S().on(q0)
    yield S().on(q1)
    yield TY(0.5).on(q0)
    yield TY(0.5).on(q1)
    yield SqrtISwap().H.on(q0, q1)

    yield TY(-0.5).on(q1)
    yield S_H().on(q1)

    yield TY(-0.5).on(q0)
    yield S_H().on(q0)


def translate_exch_to_can(gate: EXCH) -> Iterator[Can]:
    """Convert an exchange gate to a canonical based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield Can(t, t, t, q0, q1)


def translate_exch_to_xy_zz(gate: EXCH) -> Iterator[Union[XY, ZZ]]:
    """Convert an exchange gate to XY and ZZ gates"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield XY(t, q0, q1)
    yield ZZ(t, q0, q1)


def translate_fsim_to_xy_cz(gate: FSim) -> Iterator[Union[XY, CZ]]:
    """Convert the Cirq's FSim  gate to a canonical gate"""
    q0, q1 = gate.qubits
    theta, phi = gate.parameters()

    yield XY(theta / PI, q0, q1)
    yield CZ(q0, q1) ** (-phi/PI)


def translate_fswap(gate: FSwap) -> Iterator[Union[SWAP, CZ]]:
    """Translate fSwap gate to Swap and CV"""
    yield from gate.decompose()


def translate_fswappow(gate: FSwapPow) -> Iterator[Union[EXCH, CZPow]]:
    """Translate fSwap gate to XY and CVPow"""
    yield from gate.decompose()


def translate_givens_to_xy(gate: Givens) -> Iterator[Union[XY, T, T_H]]:
    """Convert a Givens  gate to an XY gate"""
    q0, q1 = gate.qubits
    theta, = gate.parameters()

    yield T_H(q0)
    yield T(q1)
    yield XY(theta / PI, q0, q1)
    yield T(q0)
    yield T_H(q1)


def translate_iswap_to_can(gate: ISWAP) -> Iterator[Union[Can, X]]:
    """Convert ISWAP gate to a canonical gate within the Weyl chamber."""
    q0, q1 = gate.qubits
    yield X(q0)
    yield Can(0.5, 0.5, 0, q0, q1)
    yield X(q1)


def translate_iswap_to_swap_cz(gate: ISWAP) -> Iterator[Union[SWAP, CZ, S]]:
    """Convert ISWAP gate to a SWAP, CZ based circuit."""
    q0, q1 = gate.qubits
    yield SWAP(q0, q1)
    yield CZ(q0, q1)
    yield S(q0)
    yield S(q1)


def translate_iswap_to_sqrtiswap(gate: ISWAP) -> Iterator[SqrtISwap]:
    """Translate iswap gate to square-root iswaps"""
    q0, q1 = gate.qubits
    yield SqrtISwap().on(q0, q1)
    yield SqrtISwap().on(q0, q1)


def translate_iswap_to_xy(gate: ISWAP) -> Iterator[Union[XY]]:
    """Convert ISWAP gate to a XY gate."""
    q0, q1 = gate.qubits
    yield XY(-0.5, q0, q1)


def translate_pswap_to_canonical(gate: PSWAP) -> Iterator[Union[Can, Y]]:
    """Translate parametric SWAP to a canonical circuit"""

    q0, q1 = gate.qubits
    theta, = gate.parameters()
    t = 0.5 - theta / PI
    yield Y(q0)
    yield Can(0.5, 0.5, t, q0, q1)
    yield Y(q1)


def translate_rzz_to_cnot(gate: RZZ) -> Iterator[Union[CNOT, PhaseShift, U3]]:
    """Translate QASM's RZZ gate to standard gates"""
    q0, q1 = gate.qubits
    theta, = gate.parameters()
    yield CNOT(q0, q1)
    yield PhaseShift(theta, q1)
    yield CNOT(q0, q1)


def translate_sqrtiswap_to_sqrtiswap_h(gate: SqrtISwap) \
        -> Iterator[Union[SqrtISwap_H, Z]]:
    """Translate square-root-iswap to its inverse."""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield SqrtISwap_H().on(q0, q1)
    yield Z(q0)


def translate_sqrtiswap_h_to_can(gate: SqrtISwap_H) -> Iterator[Can]:
    """Translate square-root iswap to canonical"""
    yield Can(1/4, 1/4, 0).on(*gate.qubits)


def translate_sqrtiswap_h_to_sqrtiswap(gate: SqrtISwap_H) \
        -> Iterator[Union[SqrtISwap, Z]]:
    """Translate square-root-iswap to it's inverse."""
    q0, q1 = gate.qubits
    yield Z(q0)
    yield SqrtISwap().on(q0, q1)
    yield Z(q0)


def translate_sqrtswap_to_can(gate: SqrtSwap) -> Iterator[Can]:
    """Translate square-root swap to canonical"""
    yield Can(1/4, 1/4, 1/4).on(*gate.qubits)


def translate_sqrtswap_h_to_can(gate: SqrtSwap_H) -> Iterator[Can]:
    """Translate inv. square-root swap to canonical"""
    yield Can(-1/4, -1/4, -1/4).on(*gate.qubits)


def translate_swap_to_cnot(gate: SWAP) -> Iterator[CNOT]:
    """Convert a SWAP gate to a circuit with 3 CNOTs."""
    q0, q1 = gate.qubits
    yield CNOT(q0, q1)
    yield CNOT(q1, q0)
    yield CNOT(q0, q1)


def translate_swap_to_ecp_sqrtiswap(gate: SWAP) \
        -> Iterator[Union[ECP, SqrtISwap_H, H, TZ, TY]]:
    """Translate a SWAP gate to an  ECP -- square-root-iswap sandwich.

    An intermediate step in translating swap to 3 square-root-iswap's.
    """
    q0, q1 = gate.qubits

    yield ECP().on(q0, q1)

    yield H().on(q0)
    yield H().on(q1)

    yield SqrtISwap().H.on(q0, q1)

    yield TY(-1/2).on(q1)
    yield TZ(+1).on(q1)

    yield TY(-1/2).on(q0)
    yield TZ(+1).on(q0)


def translate_swap_to_iswap_cz(gate: SWAP) -> Iterator[Union[ISWAP, CZ, S_H]]:
    """Convert ISWAP gate to a SWAP, CZ based circuit."""
    q0, q1 = gate.qubits
    yield S_H(q0)
    yield S_H(q1)
    yield CZ(q0, q1)
    yield ISWAP(q0, q1)


def translate_sycamore_to_fsim(gate: Sycamore) -> Iterator[FSim]:
    """Convert a Sycamore gate to an FSim gate"""
    yield FSim(PI/2, PI/6, *gate.qubits)


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


def translate_w_to_cnot(gate: W) -> Iterator[Union[CNOT, H, S, S_H, T, T_H]]:
    """Translate W gate to CNOT."""
    # Kudos: Decomposition given in Quipper
    # (Dek in paper far from optimal)
    # https://www.mathstat.dal.ca/~selinger/quipper/doc/QuipperLib-GateDecompositions.html
    q0, q1 = gate.qubits

    yield CNOT(q0, q1)
    yield S(q0).H
    yield H(q0)
    yield T(q0).H
    yield CNOT(q1, q0)
    yield T(q0)
    yield H(q0)
    yield S(q0)
    yield CNOT(q0, q1)


def translate_w_to_ch_cnot(gate: W) -> Iterator[Union[CNOT, CH]]:
    """Translate W gate to controlled-Hadamard and CNOT."""
    # From https://arxiv.org/pdf/1505.06552.pdf
    q0, q1 = gate.qubits

    yield CNOT(q1, q0)
    yield CNOT(q0, q1)
    yield CH(q0, q1)
    yield CNOT(q0, q1)
    yield CNOT(q1, q0)


def translate_xx_to_can(gate: XX) -> Iterator[Union[Can]]:
    """Convert an XX gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield Can(t, 0, 0, q0, q1)


def translate_xx_to_zz(gate: XX) -> Iterator[Union[H, ZZ]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield H(q0)
    yield H(q1)
    yield ZZ(t, q0, q1)
    yield H(q0)
    yield H(q1)


def translate_xy_to_can(gate: XY) -> Iterator[Can]:
    """Convert XY gate to a canonical gate."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield Can(t, t, 0, q0, q1)


def translate_xy_to_sqrtiswap(gate: XY) \
        -> Iterator[Union[Z, T, T_H, TZ, SqrtISwap_H]]:
    """Translate an XY gate to sandwich of square-root iswaps"""

    q0, q1 = gate.qubits
    t = gate.params['t']

    yield Z().on(q0)
    yield T().on(q0)
    yield T_H().on(q1)
    yield SqrtISwap_H().on(q0, q1)
    yield TZ(1-t).on(q0)
    yield TZ(t).on(q1)
    yield SqrtISwap_H().on(q0, q1)
    yield T_H().on(q0)
    yield T().on(q1)


def translate_yy_to_can(gate: YY) -> Iterator[Union[Can]]:
    """Convert an YY gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield Can(0, t, 0, q0, q1)


def translate_yy_to_zz(gate: YY) -> Iterator[Union[TX, ZZ]]:
    """Convert a YY gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield TX(0.5, q0)
    yield TX(0.5, q1)
    yield ZZ(t, q0, q1)
    yield TX(-0.5, q0)
    yield TX(-0.5, q1)


def translate_zz_to_can(gate: ZZ) -> Iterator[Union[Can]]:
    """Convert an ZZ gate to a canonical circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield Can(0, 0, t, q0, q1)


def translate_zz_to_cnot(gate: ZZ) -> Iterator[Union[CNOT, TZ]]:
    """Convert a ZZ gate to a CNOT based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield CNOT(q0, q1)
    yield TZ(t, q1)
    yield CNOT(q0, q1)


def translate_zz_to_xx(gate: ZZ) -> Iterator[Union[H, XX]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield H(q0)
    yield H(q1)
    yield XX(t, q0, q1)
    yield H(q0)
    yield H(q1)


def translate_zz_to_yy(gate: ZZ) -> Iterator[Union[TX, YY]]:
    """Convert a YY gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield TX(-0.5, q0)
    yield TX(-0.5, q1)
    yield YY(t, q0, q1)
    yield TX(0.5, q0)
    yield TX(0.5, q1)


# 3-qubit gates

def translate_ccix_to_cnot(gate: CCiX) -> Iterator[Union[CNOT, H, T, T_H]]:
    """Decompose doubly-controlled iX-gate to 4 CNOTs.

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
    yield CNOT(q1, q2)
    yield T(q2)
    yield CNOT(q0, q2)
    yield T(q2).H
    yield CNOT(q1, q2)
    yield T(q2)
    yield CNOT(q0, q2)
    yield T(q2).H
    yield H(q2)


def translate_ccix_to_cnot_adjacent(gate: CCiX) \
        -> Iterator[Union[CNOT, H, T, T_H]]:
    """Decompose doubly-controlled iX-gate to 8 CNOTs, respecting adjacency.

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
    yield CNOT(q2, q1)
    yield CNOT(q1, q0)
    yield T(q0).H
    yield T(q1)
    yield CNOT(q2, q1)
    yield CNOT(q1, q0)
    yield T(q0)
    yield T(q2).H
    yield CNOT(q2, q1)
    yield CNOT(q1, q0)
    yield CNOT(q2, q1)
    yield CNOT(q1, q0)
    yield H(q2)


def translate_ccnot_to_ccz(gate: CCNOT) -> Iterator[Union[H, CCZ]]:
    """Convert CCNOT (Toffoli) gate to CCZ using Hadamards
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


def translate_ccnot_to_cnot(gate: CCNOT) -> Iterator[Union[H, T, T_H, CNOT]]:
    """Standard decomposition of CCNOT (Toffoli) gate into six CNOT gates.
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
    yield CNOT(q1, q2)
    yield T_H(q2)
    yield CNOT(q0, q2)
    yield T(q2)
    yield CNOT(q1, q2)
    yield T_H(q2)
    yield CNOT(q0, q2)
    yield T(q1)
    yield T(q2)
    yield H(q2)
    yield CNOT(q0, q1)
    yield T(q0)
    yield T_H(q1)
    yield CNOT(q0, q1)


def translate_ccnot_to_cnot_AMMR(gate: CCNOT) \
        -> Iterator[Union[H, T, T_H, CNOT]]:
    """Depth 9 decomposition of CCNOT (Toffoli) gate into 7 CNOT gates.
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
    yield CNOT(q1, q0)
    yield CNOT(q2, q1)
    yield CNOT(q0, q2)
    yield T_H(q1)
    yield T(q2)
    yield CNOT(q0, q1)
    yield T_H(q0)
    yield T_H(q1)
    yield CNOT(q2, q1)
    yield CNOT(q0, q2)
    yield CNOT(q1, q0)
    yield H(q2)


def translate_ccnot_to_cv(gate: CCNOT) -> Iterator[Union[CV, CV_H, CNOT]]:
    """Decomposition of CCNOT (Toffoli) gate into 3 CV and 2 CNOTs.
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
    yield CNOT(q0, q1)
    yield CV_H(q1, q2)
    yield CNOT(q0, q1)
    yield CV(q0, q2)


def translate_ccxpow_to_cnotpow(gate: CCXPow) \
        -> Iterator[Union[CNOT, CNotPow]]:
    """Decomposition of powers of CCNot gates to powers of CNOT gates."""
    q0, q1, q2 = gate.qubits
    t, = gate.parameters()
    yield CNotPow(t/2, q1, q2)
    yield CNOT(q0, q1)
    yield CNotPow(-t/2, q1, q2)
    yield CNOT(q0, q1)
    yield CNotPow(t/2, q0, q2)


def translate_ccz_to_adjacent_cnot(gate: CCZ) -> Iterator[Union[T, CNOT, T_H]]:
    """Decomposition of CCZ gate into 8 CNOT gates.
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

    yield CNOT(q0, q1)
    yield CNOT(q1, q2)

    yield T_H(q1)
    yield T(q2)

    yield CNOT(q0, q1)
    yield CNOT(q1, q2)

    yield T_H(q2)

    yield CNOT(q0, q1)
    yield CNOT(q1, q2)

    yield T_H(q2)

    yield CNOT(q0, q1)
    yield CNOT(q1, q2)


def translate_ccz_to_ccnot(gate: CCZ) -> Iterator[Union[H, CCNOT]]:
    """Convert  CCZ gate to CCNOT gate using Hadamards
    ::

        ───●───     ───────●───────
           │               │
        ───●───  =  ───────●───────
           │               │
        ───X───     ───H───●───H───

    """
    q0, q1, q2 = gate.qubits

    yield H(q2)
    yield CCNOT(q0, q1, q2)
    yield H(q2)


def translate_cswap_to_ccnot(gate: CSWAP) -> Iterator[Union[CNOT, CCNOT]]:
    """Convert a CSWAP gate to a circuit with a CCNOT and 2 CNOTs"""
    q0, q1, q2 = gate.qubits
    yield CNOT(q2, q1)
    yield CCNOT(q0, q1, q2)
    yield CNOT(q2, q1)


def translate_cswap_to_cnot(gate: CSWAP) -> \
        Iterator[Union[CNOT, H, T, T_H, V, V_H, S, S_H]]:
    """Adjacency respecting decomposition of CSWAP to CNOT. 
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

    yield CNOT(q2, q1)
    yield H(q2)
    yield S(q2).H
    yield T(q0)
    yield T(q1)
    yield T(q2).H
    yield CNOT(q0, q1)
    yield CNOT(q1, q2)
    yield T(q1).H
    yield T(q2)
    yield CNOT(q0, q1)
    yield CNOT(q1, q2)
    yield T(q2).H
    yield CNOT(q0, q1)
    yield CNOT(q1, q2)
    yield T(q2).H
    yield V(q1)
    yield CNOT(q0, q1)
    yield CNOT(q1, q2)
    yield S(q2)
    yield V(q1)
    yield V(q2).H


def translate_cswap_inside_to_cnot(gate: CSWAP) -> \
        Iterator[Union[CNOT, H, T, T_H, V, V_H, S, S_H]]:
    """Adjacency respecting decomposition of centrally controlled CSWAP to CNOT. 
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

    yield CNOT(q1, q0)
    yield CNOT(q0, q1)
    yield CNOT(q2, q0)
    yield H(q2)
    yield T(q2)
    yield CNOT(q0, q2)
    yield T(q1)
    yield T_H(q0)
    yield T(q2)
    yield CNOT(q1, q0)
    yield CNOT(q0, q2)
    yield T(q0)
    yield T_H(q2)
    yield CNOT(q1, q0)
    yield CNOT(q0, q2)
    yield V(q0)
    yield T_H(q2)
    yield CNOT(q0, q1)
    yield CNOT(q0, q2)
    yield CNOT(q1, q0)
    yield CNOT(q0, q2)
    yield H(q2)
    yield S_H(q2)
    yield V_H(q1)


def translate_deutsch_to_barenco(gate: Deutsch) -> Iterator[Barenco]:
    """Translate a 3-qubit Deutsch gate to five 2-qubit Barenco gates.

    Ref:
        A Universal Two–Bit Gate for Quantum Computation, A. Barenco (1996)
        https://arxiv.org/pdf/quant-ph/9505016.pdf  :cite:`Barenco1996`
    """
    q0, q1, q2 = gate.qubits
    theta, = gate.parameters()
    yield Barenco(0, PI/4, theta/2,   q1, q2)
    yield Barenco(0, PI/4, theta/2,   q0, q2)
    yield Barenco(0, PI/2, PI/2,      q0, q1)
    yield Barenco(PI, -PI/4, theta/2, q1, q2)
    yield Barenco(0, PI/2, PI/2,      q0, q1)


# Multi qubit gates

def translate_iden(gate: IDEN) -> Iterator[I]:
    """Translate multi-qubit identity gate to single qubit identities"""
    for q in gate.qubits:
        yield I(q)


TRANSLATORS: Dict[str, Callable] = {}
TRANSLATORS = {name: func for name, func in globals().items()
               if name.startswith('translate_')}


# Note: Translators auto-magically added
__all__ = (
    'circuit_translate',
    'select_translators',
    'TRANSLATORS') + tuple(TRANSLATORS.keys())

# fin
