#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A collection of useful circuit identities"""

from itertools import zip_longest

import numpy as np

from quantumflow import (
    I, H, X, Y, Z, CNOT, CZ, SWAP, ISWAP, XX, YY, ZZ, S, CAN,
    CCNOT, RZ, Circuit, gates_close, RX, CPHASE, TZ, TY,
    CPHASE00, CPHASE10, CPHASE01, PSWAP, translate_ccnot_to_cnot,
    circuit_to_diagram)

import quantumflow as qf

from sympy import Symbol

syms = {
    'alpha':    Symbol('α'),
    'lam':      Symbol('λ'),
    'nx':       Symbol('n_x'),
    'ny':       Symbol('n_y'),
    'nz':       Symbol('n_z'),
    'p':        Symbol('p'),
    'phi':      Symbol('φ'),
    't':        Symbol('t'),
    't0':       Symbol('t_0'),
    't1':       Symbol('t_1'),
    't2':       Symbol('t_2'),
    'theta':    Symbol('θ'),
    'tx':       Symbol('t_x'),
    'ty':       Symbol('t_y'),
    'tz':       Symbol('t_z'),
    }


def identities():
    """ Return a list of circuit identities, each consisting of a name, and
    two equivalent Circuits."""

    circuit_identities = []

    # Pick random parameter
    theta = np.pi * np.random.uniform()

    # Single qubit gate identities

    name = "Hadamard is own inverse"
    circ0 = Circuit([H(0), H(0)])
    circ1 = Circuit([I(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Hadamards convert X to Z"
    circ0 = Circuit([H(0), X(0), H(0)])
    circ1 = Circuit([Z(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Hadamards convert Z to X"
    circ0 = Circuit([H(0), Z(0), H(0)])
    circ1 = Circuit([X(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "S sandwich converts X to Y"
    circ0 = Circuit([S(0).H, X(0), S(0)])
    circ1 = Circuit([Y(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "S sandwich converts Y to X"
    circ0 = Circuit([S(0), Y(0), S(0).H])
    circ1 = Circuit([X(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Hadamards convert RZ to RX"
    circ0 = Circuit([H(0), RZ(theta, 0), H(0)])
    circ1 = Circuit([RX(theta, 0)])
    circuit_identities.append([name, circ0, circ1])

    # ZYZ Decompositions

    name = "Hadamard ZYZ decomposition"
    circ0 = Circuit([H(0)])
    circ1 = Circuit([Z(0), Y(0)**0.5])
    circuit_identities.append([name, circ0, circ1])

    # CNOT identities

    name = "CZ to CNOT"
    circ0 = Circuit([CZ(0, 1)])
    circ1 = Circuit([H(1), CNOT(0, 1), H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "CZ to CNOT (2)"
    circ0 = Circuit([CZ(0, 1)])
    circ1 = Circuit([TY(+0.5, 1), CNOT(0, 1), TY(-0.5, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "SWAP to 3 CNOTs"
    circ0 = Circuit([SWAP(0, 1)])
    circ1 = Circuit([CNOT(0, 1), CNOT(1, 0), CNOT(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "SWAP to 3 CZs"
    circ0 = Circuit([SWAP(0, 1)])
    circ1 = Circuit([CZ(0, 1), H(0), H(1), CZ(1, 0), H(0), H(1),
                     CZ(0, 1), H(0), H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISWAP decomposition to SWAP and CNOT"
    circ0 = Circuit([ISWAP(0, 1)])
    circ1 = Circuit([SWAP(0, 1), H(1), CNOT(0, 1), H(1), S(0), S(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISWAP decomposition to SWAP and CZ"
    # This makes it clear why you can commute RZ's across ISWAP
    circ0 = Circuit([ISWAP(0, 1)])
    circ1 = Circuit([SWAP(0, 1), CZ(0, 1), S(0), S(1)])
    circuit_identities.append([name, circ0, circ1])

    # http://info.phys.unm.edu/~caves/courses/qinfo-f14/lectures/lectures21-23.pdf
    name = "CNOT sandwich with X on control"
    circ0 = Circuit([CNOT(0, 1), X(0), CNOT(0, 1)])
    circ1 = Circuit([X(0), X(1)])
    circuit_identities.append([name, circ0, circ1])

    # http://info.phys.unm.edu/~caves/courses/qinfo-f14/lectures/lectures21-23.pdf
    name = "CNOT sandwich with Z on target"
    circ0 = Circuit([CNOT(0, 1), Z(1), CNOT(0, 1)])
    circ1 = Circuit([Z(0), Z(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "DCNOT (Double-CNOT) to iSWAP"
    circ0 = Circuit([CNOT(0, 1), CNOT(1, 0)])
    circ1 = Circuit([H(0), S(0).H, S(1).H, ISWAP(0, 1), H(1)])
    circuit_identities.append([name, circ0, circ1])

    # Commuting single qubit gates across 2 qubit games

    name = "Commute X on CNOT target"
    circ0 = Circuit([X(1), CNOT(0, 1)])
    circ1 = Circuit([CNOT(0, 1), X(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X on CNOT control"
    circ0 = Circuit([X(0), CNOT(0, 1)])
    circ1 = Circuit([CNOT(0, 1), X(0), X(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute Z on CNOT target"
    circ0 = Circuit([Z(1), CNOT(0, 1)])
    circ1 = Circuit([CNOT(0, 1), Z(0), Z(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute Z on CNOT control"
    circ0 = Circuit([Z(0), CNOT(0, 1)])
    circ1 = Circuit([CNOT(0, 1), Z(0)])
    circuit_identities.append([name, circ0, circ1])

    #  Canonical gate identities

    name = "Canonical gates: CZ to ZZ"
    circ0 = Circuit([CZ(0, 1),
                     S(0),
                     S(1)])
    circ1 = Circuit([ZZ(0.5, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Canonical gates: XX to ZZ"
    circ0 = Circuit([H(0), H(1),
                     XX(0.5, 0, 1),
                     H(0), H(1)])
    circ1 = Circuit([ZZ(0.5, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Canonical gates: CNOT to XX"
    circ0 = Circuit([CNOT(0, 1)])
    circ1 = Circuit([H(0),
                     XX(0.5, 0, 1),
                     H(0), H(1),
                     S(0).H, S(1).H,
                     H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Canonical gates: SWAP to Canonical"
    circ0 = Circuit([SWAP(0, 1)])
    circ1 = Circuit([CAN(0.5, 0.5, 0.5, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISWAP to Canonical"
    circ0 = Circuit([ISWAP(0, 1)])
    circ1 = Circuit([CAN(0.5, 0.5, 1.0, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISWAP to Canonical in Weyl chamber"
    circ0 = Circuit([ISWAP(0, 1)])
    circ1 = Circuit([X(0), CAN(0.5, 0.5, 0.0, 0, 1), X(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "DCNOT to Canonical"
    circ0 = Circuit([CNOT(0, 1), CNOT(1, 0)])
    circ1 = Circuit([H(0), S(0).H, S(1).H, X(0),
                     CAN(0.5, 0.5, 0.0, 0, 1), X(1), H(1)])
    circuit_identities.append([name, circ0, circ1])

    # Multi-qubit circuits

    name = "CNOT controls commute"
    circ0 = Circuit([CNOT(1, 0), CNOT(1, 2)])
    circ1 = Circuit([CNOT(1, 2), CNOT(1, 0)])
    circuit_identities.append([name, circ0, circ1])

    name = "CNOT targets commute"
    circ0 = Circuit([CNOT(0, 1), CNOT(2, 1)])
    circ1 = Circuit([CNOT(2, 1), CNOT(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commutation of CNOT target/control"
    circ0 = Circuit([CNOT(0, 1), CNOT(1, 2)])
    circ1 = Circuit([CNOT(1, 2), CNOT(0, 2), CNOT(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Indirect CNOT and 4 CNOTS with intermediate qubit"
    circ0 = Circuit([CNOT(0, 2), I(1)])
    circ1 = Circuit([CNOT(0, 1),
                     CNOT(1, 2),
                     CNOT(0, 1),
                     CNOT(1, 2)])
    circuit_identities.append([name, circ0, circ1])

    name = "CZs with shared shared qubit commute"
    circ0 = Circuit([CZ(0, 1), CZ(1, 2)])
    circ1 = Circuit([CZ(1, 2), CZ(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    # Parametric circuits

    name = "ZZ to CNOTs"  # 1108.4318
    circ0 = Circuit([ZZ(theta/np.pi, 0, 1)])
    circ1 = Circuit([CNOT(0, 1), RZ(theta, 1), CNOT(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "XX to CNOTs"  # 1108.4318
    circ0 = Circuit([XX(theta/np.pi, 0, 1)])
    circ1 = Circuit([H(0), H(1), CNOT(0, 1), RZ(theta, 1), CNOT(0, 1),
                     H(0), H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "XX to CNOTs (2)"
    circ0 = Circuit([XX(theta/np.pi, 0, 1)])
    circ1 = Circuit([Y(0)**0.5, Y(1)**0.5, CNOT(0, 1), RZ(theta, 1),
                     CNOT(0, 1), Y(0)**-0.5, Y(1)**-0.5])
    circuit_identities.append([name, circ0, circ1])

    name = "YY to CNOTs"
    circ0 = Circuit([YY(theta/np.pi, 0, 1)])
    circ1 = Circuit([X(0)**0.5, X(1)**0.5, CNOT(0, 1), RZ(theta, 1),
                     CNOT(0, 1), X(0)**-0.5, X(1)**-0.5])
    circuit_identities.append([name, circ0, circ1])

    def cphase_to_zz(gate: CPHASE):
        t = - gate.params['theta'] / (2 * np.pi)
        q0, q1 = gate.qubits
        circ = Circuit([ZZ(t, q0, q1), TZ(-t, q0), TZ(-t, q1)])
        return circ

    def cphase00_to_zz(gate: CPHASE00):
        t = - gate.params['theta'] / (2 * np.pi)
        q0, q1 = gate.qubits
        circ = Circuit([X(0), X(1),
                        ZZ(t, q0, q1), TZ(-t, q0), TZ(-t, q1),
                        X(0), X(1)])
        return circ

    def cphase01_to_zz(gate: CPHASE00):
        t = - gate.params['theta'] / (2 * np.pi)
        q0, q1 = gate.qubits
        circ = Circuit([X(0),
                        ZZ(t, q0, q1), TZ(-t, q0), TZ(-t, q1),
                        X(0)])
        return circ

    def cphase10_to_zz(gate: CPHASE00):
        t = - gate.params['theta'] / (2 * np.pi)
        q0, q1 = gate.qubits
        circ = Circuit([X(1),
                        ZZ(t, q0, q1), TZ(-t, q0), TZ(-t, q1),
                        X(1)])
        return circ

    name = "CPHASE to ZZ"
    gate = CPHASE(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "CPHASE00 to ZZ"
    gate = CPHASE00(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase00_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "CPHASE01 to ZZ"
    gate = CPHASE01(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase01_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "CPHASE10 to ZZ"
    gate = CPHASE10(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase10_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "PSWAP to Canonical"
    gate = PSWAP(theta, 0, 1)
    t = 0.5 - theta / np.pi
    circ0 = Circuit([gate])
    circ1 = Circuit()
    circ1 += TY(1, 0)
    circ1 += CAN(0.5, 0.5, t)
    circ1 += TY(1, 1)
    circuit_identities.append([name, circ0, circ1])

    # Three qubit gates

    name = "Toffoli gate CNOT decomposition"
    circ0 = Circuit([CCNOT(0, 1, 2)])
    circ1 = translate_ccnot_to_cnot(CCNOT(0, 1, 2))
    circuit_identities.append([name, circ0, circ1])

    name = "CCZ to CNOTs, respecting adjacency"
    gate = qf.CCZ(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_ccz_to_adjacent_cnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CCNOT to CCZ"
    gate = qf.CCNOT(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_ccnot_to_ccz(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CSWAP to CCNOT"
    gate = qf.CSWAP(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_cswap_to_ccnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CSWAP to CNOT"
    gate = qf.CSWAP(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_cswap_to_cnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CSWAP to CNOT (control between targets)"
    gate = qf.CSWAP(1, 0, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_cswap_inside_to_cnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CH to Clifford+T"
    gate = qf.CH(0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_ch_to_cpt(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CV to Clifford+T"
    gate = qf.CV(0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_cv_to_cpt(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CV_H to Clifford+T"
    gate = qf.CV_H(0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_cvh_to_cpt(gate))
    circuit_identities.append([name, circ0, circ1])

    # from sympy import Symbol, pi
    name = "CTX to ZZ"
    gate = qf.CTX(0.2, 0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_ctx_to_zz(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "PISWAP to Canonical"
    # theta = Symbol('θ')
    theta = 0.2
    gate = qf.PISWAP(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(qf.translate_piswap_to_can(gate))
    circuit_identities.append([name, circ0, circ1])

    return circuit_identities


def _print_circuit_identity(name, circ0, circ1,
                            min_col_width=0,
                            col_sep=5,
                            left_margin=8):
    print("# ", name)

    circ0 = Circuit(circ0)
    circ1 = Circuit(circ1)

    gates0 = circuit_to_diagram(circ0, qubit_labels=False).splitlines()
    gates1 = circuit_to_diagram(circ1, qubit_labels=False).splitlines()

    for gate0, gate1 in zip_longest(gates0, gates1, fillvalue=""):
        line = (' '*col_sep).join([gate0.ljust(min_col_width),
                                   gate1.ljust(min_col_width)])
        line = (' '*left_margin) + line
        line = line.rstrip()
        print(line)

    print()
    print()

    assert gates_close(circ0.asgate(), circ1.asgate())


def _check_circuit_identities(circuit_identities):
    for name, circ0, circ1 in circuit_identities:
        _print_circuit_identity(name, circ0, circ1)


if __name__ == "__main__":
    print()
    print("# Validate and report various circuit identities "
          "(up to global phase)")
    print()
    _check_circuit_identities(identities())
