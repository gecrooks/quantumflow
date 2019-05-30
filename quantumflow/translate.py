
"""
QuantumFlow: Translate gates into equevelnt circuits of more 'elemental' gates.

CSWAP -> CCNOT -> CNOT
CAN -> (XX, YY, ZZ)
(XX, YY, PHASE, PHASE00, PHASE01, PHASE11) -> ZZ
ISWAP -> (SWAP, CNOT)
SWAP -> CNOT
CNOT -> CZ <-> ZZ
"""

import numpy as np

from quantumflow.circuits import Circuit
from quantumflow.stdgates import (H, S, S_H, T, T_H, X, Y, Z, TX, TY, TZ,
                                  RX, RY, RZ, CAN, XX, YY, ZZ, CNOT, CZ,
                                  ISWAP, CPHASE, CPHASE01, CPHASE10, CPHASE00,
                                  SWAP, PISWAP, CCNOT, EXCH, CSWAP)
from quantumflow.circuits import ccnot_circuit


def translate_x_to_tx(gate: X) -> Circuit:
    """Translate X gate to TX"""
    q0, = gate.qubits
    return Circuit([TX(1, q0)])


def translate_y_to_ty(gate: Y) -> Circuit:
    """Translate Y gate to TY"""
    q0, = gate.qubits
    return Circuit([TY(1, q0)])


def translate_z_to_tz(gate: Z) -> Circuit:
    """Translate Z gate to TZ"""
    q0, = gate.qubits
    return Circuit([TZ(1, q0)])


def translate_s_to_tz(gate: S) -> Circuit:
    """Translate S gate to TZ"""
    q0, = gate.qubits
    return Circuit([TZ(0.5, q0)])


def translate_t_to_tz(gate: T) -> Circuit:
    """Translate T gate to TZ"""
    q0, = gate.qubits
    return Circuit([TZ(0.25, q0)])


def translate_invs_to_tz(gate: S_H) -> Circuit:
    q0, = gate.qubits
    return Circuit([TZ(-0.5, q0)])


def translate_invt_to_tz(gate: T_H) -> Circuit:
    """Translate inverse T gate to RZ (a quil standard gate)"""
    q0, = gate.qubits
    return Circuit([TZ(-0.25, q0)])


def translate_rx_to_tx(gate) -> Circuit:
    """Translate RX gate to TX"""
    q0, = gate.qubits
    t = gate.params['theta'] / np.pi
    return Circuit([TX(t, q0)])


def translate_ry_to_ty(gate) -> Circuit:
    """Translate RY gate to TY"""
    q0, = gate.qubits
    t = gate.params['theta'] / np.pi
    return Circuit([TY(t, q0)])


def translate_rz_to_tz(gate) -> Circuit:
    """Translate RZ gate to TZ"""
    q0, = gate.qubits
    t = gate.params['theta'] / np.pi
    return Circuit([TZ(t, q0)])


def translate_tx_to_rx(gate) -> Circuit:
    """Translate TX gate to RX"""
    q0, = gate.qubits
    theta = gate.params['t'] * np.pi
    return Circuit([RX(theta, q0)])


def translate_ty_to_ry(gate) -> Circuit:
    """Translate TY gate to RY"""
    q0, = gate.qubits
    theta = gate.params['t'] * np.pi
    return Circuit([RY(theta, q0)])


def translate_tz_to_rz(gate) -> Circuit:
    """Translate TZ gate to RZ"""
    q0, = gate.qubits
    theta = gate.params['t'] * np.pi
    return Circuit([RZ(theta, q0)])


def translate_ty_to_xzx(gate: TY) -> Circuit:
    """Translate TY gate to TX and TZ gates"""
    q0, = gate.qubits
    t = gate.params['t']
    return Circuit([TX(0.5, q0), TZ(t, q0), TX(-0.5, q0)])


# TESTME
# def translate_ty_to_zxz(gate: TY) -> Circuit:
#     """Translate TY gate to TZ and TX gates"""
#     q0, = gate.qubits
#     t = gate.params['t']
#     return Circuit([TZ(-0.5, q0), TX(t, q0), TZ(0.5, q0)])


def translate_tx_to_zxzxz(gate: TX) -> Circuit:
    """Convert an arbitrary power of a Pauli-X gate to native gates"""
    q0, = gate.qubits
    t = gate.params['t']

    if t == 0.5 or t == -0.5:
        return Circuit([gate])

    circ = Circuit([TZ(0.5, q0),
                    TX(0.5, q0),
                    TZ(t, q0),
                    TX(-0.5, q0),
                    TZ(-0.5, q0)])
    return circ


def translate_hadamard_to_zxz(gate: H) -> Circuit:
    """Convert a Hadamard gate to a circuit with TZ and TX gates"""
    q0, = gate.qubits
    return Circuit([TZ(0.5, q0), TX(0.5, q0), TZ(0.5, q0)])


def translate_cnot_to_cz(gate: CNOT) -> Circuit:
    """Convert CNOT gate to a CZ based circuit"""
    q0, q1 = gate.qubits
    return Circuit([H(q1), CZ(q0, q1), H(q1)])


def translate_cz_to_zz(gate: CZ) -> Circuit:
    """Convert CZ gate to a ZZ based circuit"""
    q0, q1 = gate.qubits
    return Circuit([ZZ(0.5, q0, q1), S_H(q0), S_H(q1)])


def translate_iswap_to_swap_cz(gate: ISWAP) -> Circuit:
    """Convert ISWAP gate to a SWAP, CZ based circuit"""
    q0, q1 = gate.qubits
    return Circuit([SWAP(q0, q1), CZ(q0, q1), S(q0), S(q1)])


def translate_swap_to_cnot(gate: ISWAP) -> Circuit:
    """Convert a SWAP gate to a circuit with 3 CNOTs"""
    q0, q1 = gate.qubits
    return Circuit([CNOT(q0, q1), CNOT(q1, q0), CNOT(q0, q1)])


def translate_cphase_to_zz(gate: CPHASE) -> Circuit:
    """Convert a CPHASE gate to a ZZ based circuit"""
    t = - gate.params['theta'] / (2 * np.pi)
    q0, q1 = gate.qubits
    return Circuit([ZZ(t, q0, q1), TZ(-t, q0), TZ(-t, q1)])


def translate_cphase00_to_zz(gate: CPHASE00) -> Circuit:
    """Convert a CPHASE gate to a ZZ based circuit"""
    t = - gate.params['theta'] / (2 * np.pi)
    q0, q1 = gate.qubits
    circ = Circuit([X(q0),
                    X(q1),
                    ZZ(t, q0, q1),
                    TZ(-t, q0),
                    TZ(-t, q1),
                    X(q0),
                    X(q1)])
    return circ


def translate_cphase01_to_zz(gate: CPHASE01) -> Circuit:
    """Convert a CPHASE01 gate to a ZZ based circuit"""
    t = - gate.params['theta'] / (2 * np.pi)
    q0, q1 = gate.qubits
    circ = Circuit([X(q0),
                    ZZ(t, q0, q1),
                    TZ(-t, q0),
                    TZ(-t, q1),
                    X(q0)])
    return circ


def translate_cphase10_to_zz(gate: CPHASE10) -> Circuit:
    """Convert a CPHASE10 gate to a ZZ based circuit"""
    t = - gate.params['theta'] / (2 * np.pi)
    q0, q1 = gate.qubits
    circ = Circuit([X(q1),
                    ZZ(t, q0, q1),
                    TZ(-t, q0),
                    TZ(-t, q1),
                    X(q1)])
    return circ


def translate_can_to_xx_yy_zz(gate: CAN) -> Circuit:
    """Convert a canonical gate to a circuit with XX, YY, and ZZ gates"""
    tx, ty, tz = gate.params.values()
    q0, q1 = gate.qubits
    circ = Circuit()
    if not np.isclose(tx, 0.0):
        circ += XX(tx, q0, q1)
    if not np.isclose(ty, 0.0):
        circ += YY(ty, q0, q1)
    if not np.isclose(tz, 0.0):
        circ += ZZ(tz, q0, q1)
    return circ


def translate_xx_to_zz(gate: XX) -> Circuit:
    """Covnert an XX gate to a ZZ based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    circ = Circuit([H(q0),
                    H(q1),
                    ZZ(t, q0, q1),
                    H(q0),
                    H(q1)])
    return circ


def translate_yy_to_zz(gate: YY) -> Circuit:
    """Covnert a YY gate to a ZZ based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    circ = Circuit([X(q0)**0.5,
                    X(q1)**0.5,
                    ZZ(t, q0, q1),
                    X(q0)**-0.5,
                    X(q1)**-0.5])
    return circ


def translate_zz_to_cnot(gate: ZZ) -> Circuit:
    """Convert a ZZ gate to a CNOT based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    return Circuit([CNOT(q0, q1), TZ(t, q1), CNOT(q0, q1)])


# FIXME Does not look right
# def translate_piswap_to_can(gate: PISWAP):
#     """Convert PISWAP gate to a caononical gate."""
#     q0, q1 = gate.qubits
#     t = gate.params['t']
#     return Circuit([CAN(t, t, 0, q0, q1)])


def translate_exch_to_can(gate: EXCH):
    """Convert an excahnge gate to a canonical based circuit"""
    q0, q1 = gate.qubits
    t = gate.params['t']
    return Circuit([CAN(t, t, t, q0, q1)])


def translate_cswap_to_ccnot(gate: CSWAP) -> Circuit:
    """Convert a CSWAP gate to a circuit with a CCNOT and 2 CNOTs"""
    q0, q1, q2 = gate.qubits
    circ = Circuit([CNOT(q2, q1),
                    CCNOT(q0, q1, q2),
                    CNOT(q2, q1)])
    return circ


def translate_ccnot_to_cnot(gate: CCNOT) -> Circuit:
    """Standard decomposition of CCNOT (Toffoli) gate into
    six CNOT gates (Plus Hadamard and T gates.) [Nielsen2000]_

    .. [Nielsen2000]
        M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum
        Information, Cambridge University Press (2000).
    """
    return ccnot_circuit(gate.qubits)


# DOCME
TO_QUIL_GATESET = [
    translate_can_to_xx_yy_zz,
    # translate_piswap,  # FIXME
    translate_exch_to_can,
    translate_xx_to_zz,
    translate_yy_to_zz,
    translate_zz_to_cnot
    ]


def translate_circuit(circ: Circuit,
                      translators,
                      recurse: bool = True) -> Circuit:
    """Apply a collection of translatriosn to each gate in circuit.
    If recurse, then apply translations to output of translations until
    until translationally invariant.
    """
    gates = list(reversed(circ.elements))
    translated = Circuit()

    gateclass_translation = {trans.__annotations__['gate']: trans
                             for trans in translators}

    while gates:
        gate = gates.pop()
        if type(gate) in gateclass_translation:
            trans = gateclass_translation[type(gate)](gate)
            if recurse:
                gates.extend(reversed(trans.elements))
            else:
                translated.elements.extend(trans.elements)
        else:
            translated += gate

    return translated
