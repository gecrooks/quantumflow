
# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Translate, transform, and compile circuits.
"""

from typing import (
    Sequence, Iterator, Union, Callable, Tuple, Dict, Type, List, Iterable)
import numpy as np
from numpy import pi
# FIXME: Allows more symbolic transforms, but messes things up elsewhere
# from sympy import pi
from sympy import Symbol

from .ops import Gate
from .circuits import Circuit

from .gates import (
    # one qubit
    I, X, Y, Z, H, S, T, PHASE, RX, RY, RZ,
    # RN,
    TX, TY, TZ, TH,
    # ZYZ,
    S_H, T_H, V, V_H, W, TW,
    # two qubit
    CZ, CNOT, SWAP, ISWAP, CPHASE00, CPHASE01, CPHASE10,
    CPHASE, PSWAP, CAN, XX, YY, ZZ, PISWAP, EXCH, CTX,
    # BARENCO,
    CV, CV_H, CY, CH, FSIM,
    # three qubit
    CCNOT, CSWAP, CCZ, IDEN,
    # qasm gates
    U3, U2, CU3, CRZ, RZZ
    )

# Note: __all__ at end of module


def translation_source_gate(trans: Callable) -> Type[Gate]:
    return trans.__annotations__['gate']


def translation_target_gates(trans: Callable) -> Tuple[Type[Gate]]:
    ret = trans.__annotations__['return'].__args__[0]
    if hasattr(ret, '__args__'):  # Union
        gates = ret.__args__
    else:
        gates = (ret,)

    return gates


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
    # Black Voodoo magic. We use python's type annotations to figure out the
    # source gate and target gates of a translation.

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
            # If we got here, none of the remaing translations can be
            # used. Break out of while and return results.
            break  # pragma: no cover

    return out_trans


# TODO: Rename? Confusing that different signature from everything else called
# a translation?
def translate(circ: Circuit,
              translators: Sequence,
              recurse: bool = True) -> Circuit:
    """Apply a collection of translations to each gate in circuit.
    If recurse, then apply translations to output of translations
    until translationally invariant.
    """
    gates = list(reversed(list(circ)))
    translated = Circuit()

    # Use type annotations to do dynamics dispatchm
    gateclass_translation = {translation_source_gate(trans): trans
                             for trans in translators}

    while gates:
        gate = gates.pop()
        if type(gate) in gateclass_translation:
            trans = gateclass_translation[type(gate)](gate)
            if recurse:
                gates.extend(reversed(list(trans)))
            else:
                translated.elements.extend(trans)
        else:
            translated += gate

    return translated


def simplify_tz(gate: TZ) -> Iterator[Gate]:
    """
    Simplify TZ gates to T, S, Z, S_H, T_H gates where possible,
    and drop identities.
    """
    qbs = gate.qubits
    t = gate.params['t'] % 2
    idx = int(t*4)
    if np.isclose(t*4, idx):    # FIXME: Tolerance parameter
        if idx != 0:  # Skip Identity
            gatetype = (I, T, S, Z, S_H, T_H)[idx]
            yield gatetype(*qbs)
    else:
        yield gate


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
    t = gate.params['theta'] / pi
    yield TX(t, q0)


def translate_ry_to_ty(gate: RY) -> Iterator[TY]:
    """Translate RY gate to TY"""
    q0, = gate.qubits
    t = gate.params['theta'] / pi
    yield TY(t, q0)


def translate_rz_to_tz(gate: RZ) -> Iterator[TZ]:
    """Translate RZ gate to TZ"""
    q0, = gate.qubits
    t = gate.params['theta'] / pi
    yield TZ(t, q0)


def translate_phase_to_rz(gate: PHASE) -> Iterator[RZ]:
    """Translate PHASE gate to RZ (ignoreing global phase)"""
    q0, = gate.qubits
    theta = gate.params['theta']
    yield RZ(theta, q0)


def translate_tx_to_rx(gate: TX) -> Iterator[RX]:
    """Translate TX gate to RX"""
    q0, = gate.qubits
    theta = gate.params['t'] * pi
    yield RX(theta, q0)


def translate_ty_to_ry(gate: TY) -> Iterator[RY]:
    """Translate TY gate to RY"""
    q0, = gate.qubits
    theta = gate.params['t'] * pi
    yield RY(theta, q0)


def translate_tz_to_rz(gate: TZ) -> Iterator[RZ]:
    """Translate TZ gate to RZ"""
    q0, = gate.qubits
    theta = gate.params['t'] * pi
    yield RZ(theta, q0)


def translate_ty_to_xzx(gate: TY) -> Iterator[Union[TX, TZ]]:
    """Translate TY gate to TX and TZ gates"""
    q0, = gate.qubits
    t = gate.params['t']
    yield TX(0.5, q0)
    yield TZ(t, q0)
    yield TX(-0.5, q0)


def translate_w_to_zxz(gate: W) -> Iterator[Union[X, TZ]]:
    """Translate TY gate to TX and TZ gates"""
    q0, = gate.qubits
    p = gate.params['p']
    yield TZ(-p, q0)
    yield X(q0)
    yield TZ(p, q0)


def translate_tw_to_zxz(gate: TW) -> Iterator[Union[TX, TZ]]:
    """Translate TY gate to TX and TZ gates"""
    q0, = gate.qubits
    p, t = gate.params.values()
    yield TZ(-p, q0)
    yield TX(t, q0)
    yield TZ(p, q0)


def translate_v_to_tx(gate: V) -> Iterator[TX]:
    """Translate T gate to TZ"""
    q0, = gate.qubits
    yield TX(0.5, q0)


def translate_invv_to_tx(gate: V_H) -> Iterator[TX]:
    """Translate T gate to TZ"""
    q0, = gate.qubits
    yield TX(-0.5, q0)


def translate_th_to_tx(gate: TH) -> Iterator[Union[TX, H, S, T, S_H, T_H]]:
    """Translate powers of the Hadamard gate to TX and TY"""
    q0, = gate.qubits
    t, = gate.params.values()

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
    theta, phi, lam = gate.params.values()
    yield RZ(lam, q0)
    yield RY(theta, q0)
    yield RZ(phi, q0)


def translate_u2_to_zyz(gate: U2) -> Iterator[Union[RZ, RY]]:
    """Translate QASMs U2 gate to RZ and RY"""
    q0, = gate.qubits
    phi, lam = gate.params.values()
    yield RZ(lam, q0)
    yield RY(pi/2, q0)
    yield RZ(phi, q0)


# def translate_u1_to_rz(gate: U1) -> Iterator[RZ]:
#     """Translate QASMs U1 gate to RZ and RY"""
#     q0, = gate.qubits
#     lam, = gate.params.values()
#     yield RZ(lam, q0)


# FIXME: ZYZ gate needed at all?
# TESTME
# def translate_zyz_to_zyz(gate: ZYZ) -> Iterator[Union[TZ, TY]]:
#     q0, = gate.qubits
#     t0, t1, t2 = gate.params.values()
#     yield RZ(t0, q0)
#     yield RY(t1, q0)
#     yield RZ(t2, q0)


def translate_tx_to_hzh(gate: TX) -> Iterator[Union[H, TZ]]:
    """Convert a TX gate to a circuit with Hadamard and TZ gates"""
    q0, = gate.qubits
    t, = gate.params.values()
    yield H(q0)
    yield TZ(t, q0)
    yield H(q0)


def translate_cnot_to_cz(gate: CNOT) -> Iterator[Union[H, CZ]]:
    """Convert CNOT gate to a CZ based circuit."""
    q0, q1 = gate.qubits
    yield H(q1)
    yield CZ(q0, q1)
    yield H(q1)


def translate_cz_to_zz(gate: CZ) -> Iterator[Union[ZZ, S_H]]:
    """Convert CZ gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    yield ZZ(0.5, q0, q1)
    yield S_H(q0)
    yield S_H(q1)


def translate_iswap_to_swap_cz(gate: ISWAP) -> Iterator[Union[SWAP, CZ, S]]:
    """Convert ISWAP gate to a SWAP, CZ based circuit."""
    q0, q1 = gate.qubits
    yield SWAP(q0, q1)
    yield CZ(q0, q1)
    yield S(q0)
    yield S(q1)


def translate_swap_to_cnot(gate: SWAP) -> Iterator[CNOT]:
    """Convert a SWAP gate to a circuit with 3 CNOTs."""
    q0, q1 = gate.qubits
    yield CNOT(q0, q1)
    yield CNOT(q1, q0)
    yield CNOT(q0, q1)


def translate_ctx_to_zz(gate: CTX) -> Iterator[Union[ZZ, TZ, H]]:
    """Convert a controlled X^t gate to a ZZ based circuit.
    ::

        ───●─────     ───────ZZ^-t/2───Z^t/2───────
           │       =          │
        ───X^t───     ───H───ZZ^-t/2───Z^t/2───H───

    """
    t, = gate.params.values()
    q0, q1 = gate.qubits
    yield H(q1)
    yield ZZ(-t/2, q0, q1)
    yield TZ(t/2, q0)
    yield TZ(t/2, q1)
    yield H(q1)


def translate_cphase_to_zz(gate: CPHASE) -> Iterator[Union[ZZ, TZ]]:
    """Convert a CPHASE gate to a ZZ based circuit."""
    t = - gate.params['theta'] / (2 * pi)
    q0, q1 = gate.qubits
    yield ZZ(t, q0, q1)
    yield TZ(-t, q0)
    yield TZ(-t, q1)


def translate_cphase00_to_zz(gate: CPHASE00) -> Iterator[Union[X, ZZ, TZ]]:
    """Convert a CPHASE gate to a ZZ based circuit."""
    t = - gate.params['theta'] / (2 * pi)
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
    t = - gate.params['theta'] / (2 * pi)
    q0, q1 = gate.qubits
    yield X(q0)
    yield ZZ(t, q0, q1)
    yield TZ(-t, q0)
    yield TZ(-t, q1)
    yield X(q0)


def translate_cphase10_to_zz(gate: CPHASE10) -> Iterator[Union[X, ZZ, TZ]]:
    """Convert a CPHASE10 gate to a ZZ based circuit."""
    t = - gate.params['theta'] / (2 * pi)
    q0, q1 = gate.qubits

    yield X(q1)
    yield ZZ(t, q0, q1)
    yield TZ(-t, q0)
    yield TZ(-t, q1)
    yield X(q1)


def translate_can_to_cnot(gate: CAN) -> Iterator[Union[CNOT, Z, TZ, TY]]:
    """Translate canonical gate to 3 CNOTS

    Ref:
        Optimal Quantum Circuits for General Two-Qubit Gates, Vatan & Williams
        (2004) (quant-ph/0308006) Fig. 6
    """
    # Note: sign flip on central TZ, TY, TY because of differing sign
    # conventions for Canonical.
    tx, ty, tz = gate.params.values()
    q0, q1 = gate.qubits
    yield TZ(+0.5, q1)
    yield CNOT(1, 0)
    yield TZ(tz-0.5, q0)
    yield TY(-0.5+tx, q1)
    yield CNOT(0, 1)
    yield TY(-ty+0.5, q1)
    yield CNOT(1, 0)
    yield TZ(-0.5, q0)

    # yield TZ(-0.5, q1)
    # yield CNOT(1, 0)
    # yield TZ(tz-0.5, q0)
    # yield TY(0.5-tx, q1)
    # yield CNOT(0, 1)
    # yield TY(ty-0.5, q1)
    # yield CNOT(1, 0)
    # yield TZ(0.5, q0)


def translate_can_to_xx_yy_zz(gate: CAN) -> Iterator[Union[XX, YY, ZZ]]:
    """Convert a canonical gate to a circuit with XX, YY, and ZZ gates."""
    tx, ty, tz = gate.params.values()
    q0, q1 = gate.qubits

    if isinstance(tx, Symbol) or not np.isclose(tx, 0.0):
        yield XX(tx, q0, q1)
    if isinstance(tx, Symbol) or not np.isclose(ty, 0.0):
        yield YY(ty, q0, q1)
    if isinstance(tx, Symbol) or not np.isclose(tz, 0.0):
        yield ZZ(tz, q0, q1)


def translate_xx_to_zz(gate: XX) -> Iterator[Union[H, ZZ]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield H(q0)
    yield H(q1)
    yield ZZ(t, q0, q1)
    yield H(q0)
    yield H(q1)


def translate_zz_to_xx(gate: ZZ) -> Iterator[Union[H, XX]]:
    """Convert an XX gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield H(q0)
    yield H(q1)
    yield XX(t, q0, q1)
    yield H(q0)
    yield H(q1)


def translate_yy_to_zz(gate: YY) -> Iterator[Union[TX, ZZ]]:
    """Convert a YY gate to a ZZ based circuit."""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield TX(0.5, q0)
    yield TX(0.5, q1)
    yield ZZ(t, q0, q1)
    yield TX(-0.5, q0)
    yield TX(-0.5, q1)


def translate_zz_to_cnot(gate: ZZ) -> Iterator[Union[CNOT, TZ]]:
    """Convert a ZZ gate to a CNOT based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield CNOT(q0, q1)
    yield TZ(t, q1)
    yield CNOT(q0, q1)


def translate_piswap_to_can(gate: PISWAP) -> Iterator[CAN]:
    """Convert PISWAP gate to a caononical gate
    ::

        ───PISWAP(θ)───     ───CAN(-2*θ/π,-2*θ/π,0)───
              │          =      │
        ───PISWAP(θ)───     ───CAN(-2*θ/π,-2*θ/π,0)───
    """
    q0, q1 = gate.qubits
    theta = gate.params['theta']
    t = - 2 * theta/pi
    yield CAN(t, t, 0, q0, q1)


def translate_exch_to_can(gate: EXCH) -> Iterator[CAN]:
    """Convert an exchange gate to a canonical based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    yield CAN(t, t, t, q0, q1)


def translate_cswap_to_ccnot(gate: CSWAP) -> Iterator[Union[CNOT, CCNOT]]:
    """Convert a CSWAP gate to a circuit with a CCNOT and 2 CNOTs"""
    q0, q1, q2 = gate.qubits
    yield CNOT(q2, q1)
    yield CCNOT(q0, q1, q2)
    yield CNOT(q2, q1)


def translate_cswap_to_cnot(gate: CSWAP) -> \
        Iterator[Union[CNOT, TY, T, T_H, TX, S]]:
    """Adjacency respective decomposition of CSWAP to CNOT. 
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
    yield TY(-0.5, q2)
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
    yield TX(0.5, q1)
    yield CNOT(q0, q1)
    yield CNOT(q1, q2)
    yield S(q2)
    yield TX(0.5, q1)
    yield TX(-0.5, q2)


def translate_cswap_inside_to_cnot(gate: CSWAP) -> \
        Iterator[Union[CNOT, H, T, T_H, V, V_H, S, S_H]]:
    """Adjacency respective decomposition of CSWAP to CNOT. 
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


def translate_fsim_to_can(gate: FSIM) -> Iterator[Union[CAN, CZ]]:
    """Convert the Cirq's FSIM  gate to a canonical gate"""
    q0, q1 = gate.qubits
    theta, phi = gate.params.values()

    yield CAN(theta / pi, theta / pi, 0, q0, q1)
    yield CZ(q0, q1) ** (-phi/pi)


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


def translate_cy_to_cnot(gate: CY) -> Iterator[Union[CNOT, S, S_H]]:
    """Translate CY to CNOT (CX)"""
    q0, q1 = gate.qubits
    yield S_H(q1)
    yield CNOT(q0, q1)
    yield S(q1)


# QASM Gates

def translate_cu3_to_cnot(gate: CU3) -> Iterator[Union[CNOT, PHASE, U3]]:
    """Translate QASM's CU3 gate to standard gates"""
    # Kudos: Adapted from qiskit
    # https://github.com/Qiskit/qiskit-terra/blob/master/qiskit/extensions/standard/cu3.py
    q0, q1 = gate.qubits
    theta, phi, lam = gate.params.values()

    yield PHASE((lam+phi)/2, q0)
    yield PHASE((lam-phi)/2, q1)
    yield CNOT(q0, q1)
    yield U3(-theta / 2, 0, -(phi+lam)/2, q1)
    yield CNOT(q0, q1)
    yield U3(theta / 2, phi, 0, q1)


def translate_crz_to_cnot(gate: CRZ) -> Iterator[Union[CNOT, PHASE, U3]]:
    """Translate QASM's CRZ gate to standard gates.

    Ref:
        https://arxiv.org/pdf/1707.03429.pdf
    """
    q0, q1 = gate.qubits
    theta, = gate.params.values()

    yield PHASE(theta/2, q1)
    yield CNOT(q0, q1)
    yield PHASE(-theta/2, q1)
    yield CNOT(q0, q1)


def translate_rzz_to_cnot(gate: RZZ) -> Iterator[Union[CNOT, PHASE, U3]]:
    """Translate QASM's RZZ gate to standard gates"""
    q0, q1 = gate.qubits
    theta, = gate.params.values()
    yield CNOT(q0, q1)
    yield PHASE(theta, q1)
    yield CNOT(q0, q1)


def translate_pswap_to_canonical(gate: PSWAP) -> Iterator[Union[CAN, Y]]:
    """Translate parametric SWAP to a canonical circuit"""

    q0, q1 = gate.qubits
    theta, = gate.params.values()
    t = 0.5 - theta / pi
    yield Y(q0)
    yield CAN(0.5, 0.5, t, q0, q1)
    yield Y(q1)


def translate_iden(gate: IDEN) -> Iterator[I]:
    """Translate multi-qubit identity gate to single qubit identities"""
    for q in gate.qubits:
        yield I(q)

# def translate_rn_zyz(gate: RN) -> Iterator[Union[TZ, TY]]:
#     # FIXME: Numerical decomposition. Not ideal, but I don't know
#     # the analytic decompositon
#     circ = zyz_decomposition(gate)
#     for gate in circ:
#         yield circ


# def translate_barenco_zyz(gate: BARENCO) -> Iterator[Union[TZ, TY]]:
#     # FIXME: Numerical decomposition. Not ideal, but I don't know
#     # the analytic decompositon
#     circ = zyz_decomposition(gate)
#     for gate in circ:
#         yield circ


TRANSLATORS: Dict[str, Callable] = {}
TRANSLATORS = {name: func for name, func in globals().items()
               if name.startswith('translate_')}


# Note: Translators automagically added
__all__ = (
    'translate',
    'simplify_tz',
    'select_translators',
    'TRANSLATORS') + tuple(TRANSLATORS.keys())

# fin
