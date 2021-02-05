#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A collection of useful circuit identities"""

from itertools import zip_longest

import numpy as np
from sympy import Symbol, pi

from quantumflow import (
    CCZ,
    CH,
    CV,
    CV_H,
    CZ,
    XX,
    YY,
    ZZ,
    Can,
    CCNot,
    Circuit,
    CNot,
    CNotPow,
    CPhase,
    CPhase00,
    CPhase01,
    CPhase10,
    CSwap,
    H,
    I,
    ISwap,
    PSwap,
    Rx,
    Rz,
    S,
    Swap,
    V,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
    circuit_to_diagram,
    circuit_translate,
    gates_close,
    translate_can_to_cnot,
    translate_ccnot_to_ccz,
    translate_ccnot_to_cnot,
    translate_ccz_to_adjacent_cnot,
    translate_ch_to_cpt,
    translate_cnotpow_to_zz,
    translate_cswap_inside_to_cnot,
    translate_cswap_to_ccnot,
    translate_cswap_to_cnot,
    translate_cv_to_cpt,
    translate_cvh_to_cpt,
    translate_xy_to_can,
)

syms = {
    "alpha": Symbol("α"),
    "lam": Symbol("λ"),
    "nx": Symbol("n_x"),
    "ny": Symbol("n_y"),
    "nz": Symbol("n_z"),
    "p": Symbol("p"),
    "phi": Symbol("φ"),
    "t": Symbol("t"),
    "t0": Symbol("t_0"),
    "t1": Symbol("t_1"),
    "t2": Symbol("t_2"),
    "theta": Symbol("θ"),
    "tx": Symbol("t_x"),
    "ty": Symbol("t_y"),
    "tz": Symbol("t_z"),
}


def identities():
    """Return a list of circuit identities, each consisting of a name, and
    two equivalent Circuits."""

    circuit_identities = []

    # Pick random parameter
    # theta = np.pi * np.random.uniform()
    # t = np.random.uniform()

    theta = syms["theta"]
    t = syms["t"]

    tx = syms["tx"]
    ty = syms["ty"]
    tz = syms["tz"]

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

    name = "Hadamards convert Rz to Rx"
    circ0 = Circuit([H(0), Rz(theta, 0), H(0)])
    circ1 = Circuit([Rx(theta, 0)])
    circuit_identities.append([name, circ0, circ1])

    # Simplified from Cirq's "_potential_cross_whole_w"
    # "_potential_cross_partial_w" is essentially same identity
    # X and ZPow commute
    name = "X-sandwich inverts ZPow"
    circ0 = Circuit([X(0), ZPow(t, 0), X(0)])
    circ1 = Circuit([ZPow(-t, 0)])
    circuit_identities.append([name, circ0, circ1])

    # Same relation  as previous
    name = "Commute X past ZPow"
    circ0 = Circuit([X(0), ZPow(t, 0)])
    circ1 = Circuit([ZPow(-t, 0), X(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X past YPow"
    circ0 = Circuit([X(0), YPow(t, 0)])
    circ1 = Circuit([YPow(-t, 0), X(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute V past ZPow"
    circ0 = Circuit([V(0), ZPow(t, 0)])
    circ1 = Circuit([YPow(t, 0), V(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute V past YPow"
    circ0 = Circuit([V(0), YPow(t, 0)])
    circ1 = Circuit([ZPow(-t, 0), V(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute V.H past ZPow"
    circ0 = Circuit([V(0).H, ZPow(t, 0)])
    circ1 = Circuit([YPow(-t, 0), V(0).H])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute V.H past YPow"
    circ0 = Circuit([V(0).H, YPow(t, 0)])
    circ1 = Circuit([ZPow(t, 0), V(0).H])
    circuit_identities.append([name, circ0, circ1])

    name = "Couumte S.H past XPow"
    circ0 = Circuit([S(0).H, XPow(t, 0)])
    circ1 = Circuit([YPow(t, 0), S(0).H])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute S past XPow"
    circ0 = Circuit([S(0), XPow(t, 0)])
    circ1 = Circuit([YPow(-t, 0), S(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Y to ZX"
    circ0 = Circuit([Y(0)])
    circ1 = Circuit([Z(0), X(0)])
    circuit_identities.append([name, circ0, circ1])

    # ZYZ Decompositions

    name = "Hadamard ZYZ decomposition"
    circ0 = Circuit([H(0)])
    circ1 = Circuit([Z(0), Y(0) ** 0.5])
    circuit_identities.append([name, circ0, circ1])

    # CNot identities

    name = "CZ to CNot"
    circ0 = Circuit([CZ(0, 1)])
    circ1 = Circuit([H(1), CNot(0, 1), H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "CZ to CNot (2)"
    circ0 = Circuit([CZ(0, 1)])
    circ1 = Circuit([YPow(+0.5, 1), CNot(0, 1), YPow(-0.5, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Swap to 3 CNots"
    circ0 = Circuit([Swap(0, 1)])
    circ1 = Circuit([CNot(0, 1), CNot(1, 0), CNot(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Swap to 3 CZs"
    circ0 = Circuit([Swap(0, 1)])
    circ1 = Circuit([CZ(0, 1), H(0), H(1), CZ(1, 0), H(0), H(1), CZ(0, 1), H(0), H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISwap decomposition to Swap and CNot"
    circ0 = Circuit([ISwap(0, 1)])
    circ1 = Circuit([Swap(0, 1), H(1), CNot(0, 1), H(1), S(0), S(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISwap decomposition to Swap and CZ"
    # This makes it clear why you can commute Rz's across ISwap
    circ0 = Circuit([ISwap(0, 1)])
    circ1 = Circuit([Swap(0, 1), CZ(0, 1), S(0), S(1)])
    circuit_identities.append([name, circ0, circ1])

    # http://info.phys.unm.edu/~caves/courses/qinfo-f14/lectures/lectures21-23.pdf
    name = "CNot sandwich with X on control"
    circ0 = Circuit([CNot(0, 1), X(0), CNot(0, 1)])
    circ1 = Circuit([X(0), X(1)])
    circuit_identities.append([name, circ0, circ1])

    # http://info.phys.unm.edu/~caves/courses/qinfo-f14/lectures/lectures21-23.pdf
    name = "CNot sandwich with Z on target"
    circ0 = Circuit([CNot(0, 1), Z(1), CNot(0, 1)])
    circ1 = Circuit([Z(0), Z(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "DCNot (Double-CNot) to iSwap"
    circ0 = Circuit([CNot(0, 1), CNot(1, 0)])
    circ1 = Circuit([H(0), S(0).H, S(1).H, ISwap(0, 1), H(1)])
    circuit_identities.append([name, circ0, circ1])

    # Commuting single qubit gates across 2 qubit games

    name = "Commute X on CNot target"
    circ0 = Circuit([X(1), CNot(0, 1)])
    circ1 = Circuit([CNot(0, 1), X(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X on CNot control"
    circ0 = Circuit([X(0), CNot(0, 1)])
    circ1 = Circuit([CNot(0, 1), X(0), X(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute Z on CNot target"
    circ0 = Circuit([Z(1), CNot(0, 1)])
    circ1 = Circuit([CNot(0, 1), Z(0), Z(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute Z on CNot control"
    circ0 = Circuit([Z(0), CNot(0, 1)])
    circ1 = Circuit([CNot(0, 1), Z(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X with CZ"
    circ0 = Circuit([X(0), CZ(0, 1)])
    circ1 = Circuit([CZ(0, 1), X(0), Z(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X with XX"
    circ0 = Circuit([X(0), XX(t, 0, 1)])
    circ1 = Circuit(
        [
            XX(t, 0, 1),
            X(0),
        ]
    )
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X with YY"
    circ0 = Circuit([X(0), YY(t, 0, 1)])
    circ1 = Circuit(
        [
            YY(-t, 0, 1),
            X(0),
        ]
    )
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X with ZZ"
    circ0 = Circuit([X(0), ZZ(t, 0, 1)])
    circ1 = Circuit(
        [
            ZZ(-t, 0, 1),
            X(0),
        ]
    )
    circuit_identities.append([name, circ0, circ1])

    name = "Commute X with Canonical"
    circ0 = Circuit([X(0), Can(tx, ty, tz, 0, 1)])
    circ1 = Circuit([Can(tx, -ty, -tz, 0, 1), X(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute Y with Canonical"
    circ0 = Circuit([Y(0), Can(tx, ty, tz, 0, 1)])
    circ1 = Circuit([Can(-tx, ty, -tz, 0, 1), Y(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute Z with Canonical"
    circ0 = Circuit([Z(0), Can(tx, ty, tz, 0, 1)])
    circ1 = Circuit([Can(-tx, -ty, tz, 0, 1), Z(0)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commute Sqrt(X) with Canonical switches ty and tz arguments"
    circ0 = Circuit([V(0), V(1), Can(tx, ty, tz, 0, 1)])
    circ1 = Circuit([Can(tx, tz, ty, 0, 1), V(0), V(1)])
    circuit_identities.append([name, circ0, circ1])

    #  Canonical decompositions

    name = "Canonical gates: CZ to ZZ"
    circ0 = Circuit([CZ(0, 1), S(0), S(1)])
    circ1 = Circuit([ZZ(0.5, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Canonical gates: XX to ZZ"
    circ0 = Circuit([H(0), H(1), XX(0.5, 0, 1), H(0), H(1)])
    circ1 = Circuit([ZZ(0.5, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Canonical gates: CNot to XX"
    circ0 = Circuit([CNot(0, 1)])
    circ1 = Circuit([H(0), XX(0.5, 0, 1), H(0), H(1), S(0).H, S(1).H, H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Canonical gates: Swap to Canonical"
    circ0 = Circuit([Swap(0, 1)])
    circ1 = Circuit([Can(0.5, 0.5, 0.5, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISwap to Canonical"
    circ0 = Circuit([ISwap(0, 1)])
    circ1 = Circuit([Can(-0.5, -0.5, 0.0, 0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISwap to Canonical in Weyl chamber"
    circ0 = Circuit([ISwap(0, 1)])
    circ1 = Circuit([X(0), Can(0.5, 0.5, 0.0, 0, 1), X(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "ISwap conjugate to ISwap"
    circ0 = Circuit([ISwap(0, 1).H])
    circ1 = Circuit([X(0), ISwap(0, 1), X(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "DCNot to Canonical"
    circ0 = Circuit([CNot(0, 1), CNot(1, 0)])
    circ1 = Circuit([H(0), S(0).H, S(1).H, X(0), Can(0.5, 0.5, 0.0, 0, 1), X(1), H(1)])
    circuit_identities.append([name, circ0, circ1])

    # Multi-qubit circuits

    name = "CNot controls commute"
    circ0 = Circuit([CNot(1, 0), CNot(1, 2)])
    circ1 = Circuit([CNot(1, 2), CNot(1, 0)])
    circuit_identities.append([name, circ0, circ1])

    name = "CNot targets commute"
    circ0 = Circuit([CNot(0, 1), CNot(2, 1)])
    circ1 = Circuit([CNot(2, 1), CNot(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Commutation of CNot target/control"
    circ0 = Circuit([CNot(0, 1), CNot(1, 2)])
    circ1 = Circuit([CNot(1, 2), CNot(0, 2), CNot(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Indirect CNot and 4 CNotS with intermediate qubit"
    circ0 = Circuit([CNot(0, 2), I(1)])
    circ1 = Circuit([CNot(0, 1), CNot(1, 2), CNot(0, 1), CNot(1, 2)])
    circuit_identities.append([name, circ0, circ1])

    name = "CZs with shared shared qubit commute"
    circ0 = Circuit([CZ(0, 1), CZ(1, 2)])
    circ1 = Circuit([CZ(1, 2), CZ(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Z's shift ZZ by one unit"
    circ0 = Circuit([ZZ(t, 0, 1)])
    circ1 = Circuit([ZZ(t + 1, 0, 1), Z(0), Z(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "Z's shift ZZ by one unit"
    circ0 = Circuit([XX(t, 0, 1)])
    circ1 = Circuit([XX(t + 1, 0, 1), X(0), X(1)])
    circuit_identities.append([name, circ0, circ1])

    # Parametric circuits

    name = "ZZ to CNots"  # 1108.4318
    circ0 = Circuit([ZZ(theta / pi, 0, 1)])
    circ1 = Circuit([CNot(0, 1), Rz(theta, 1), CNot(0, 1)])
    circuit_identities.append([name, circ0, circ1])

    name = "XX to CNots"  # 1108.4318
    circ0 = Circuit([XX(theta / pi, 0, 1)])
    circ1 = Circuit([H(0), H(1), CNot(0, 1), Rz(theta, 1), CNot(0, 1), H(0), H(1)])
    circuit_identities.append([name, circ0, circ1])

    name = "XX to CNots (2)"
    circ0 = Circuit([XX(theta / pi, 0, 1)])
    circ1 = Circuit(
        [
            Y(0) ** 0.5,
            Y(1) ** 0.5,
            CNot(0, 1),
            Rz(theta, 1),
            CNot(0, 1),
            Y(0) ** -0.5,
            Y(1) ** -0.5,
        ]
    )
    circuit_identities.append([name, circ0, circ1])

    name = "YY to CNots"
    circ0 = Circuit([YY(theta / pi, 0, 1)])
    circ1 = Circuit(
        [
            X(0) ** 0.5,
            X(1) ** 0.5,
            CNot(0, 1),
            Rz(theta, 1),
            CNot(0, 1),
            X(0) ** -0.5,
            X(1) ** -0.5,
        ]
    )
    circuit_identities.append([name, circ0, circ1])

    def cphase_to_zz(gate: CPhase):
        t = -gate.param("theta") / (2 * pi)
        q0, q1 = gate.qubits
        circ = Circuit([ZZ(t, q0, q1), ZPow(-t, q0), ZPow(-t, q1)])
        return circ

    def cphase00_to_zz(gate: CPhase00):
        t = -gate.param("theta") / (2 * pi)
        q0, q1 = gate.qubits
        circ = Circuit(
            [X(0), X(1), ZZ(t, q0, q1), ZPow(-t, q0), ZPow(-t, q1), X(0), X(1)]
        )
        return circ

    def cphase01_to_zz(gate: CPhase00):
        t = -gate.param("theta") / (2 * pi)
        q0, q1 = gate.qubits
        circ = Circuit([X(0), ZZ(t, q0, q1), ZPow(-t, q0), ZPow(-t, q1), X(0)])
        return circ

    def cphase10_to_zz(gate: CPhase00):
        t = -gate.param("theta") / (2 * pi)
        q0, q1 = gate.qubits
        circ = Circuit([X(1), ZZ(t, q0, q1), ZPow(-t, q0), ZPow(-t, q1), X(1)])
        return circ

    name = "CPhase to ZZ"
    gate = CPhase(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "CPhase00 to ZZ"
    gate = CPhase00(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase00_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "CPhase01 to ZZ"
    gate = CPhase01(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase01_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "CPhase10 to ZZ"
    gate = CPhase10(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = cphase10_to_zz(gate)
    circuit_identities.append([name, circ0, circ1])

    name = "PSwap to Canonical"
    gate = PSwap(theta, 0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit()
    circ1 += YPow(1, 0)
    circ1 += Can(0.5, 0.5, 0.5 - theta / pi, 0, 1)
    circ1 += YPow(1, 1)
    circuit_identities.append([name, circ0, circ1])

    # Three qubit gates

    name = "Toffoli gate CNot decomposition"
    circ0 = Circuit([CCNot(0, 1, 2)])
    circ1 = translate_ccnot_to_cnot(CCNot(0, 1, 2))
    circuit_identities.append([name, circ0, circ1])

    name = "CCZ to CNots, respecting adjacency"
    gate = CCZ(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_ccz_to_adjacent_cnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CCNot to CCZ"
    gate = CCNot(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_ccnot_to_ccz(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CSwap to CCNot"
    gate = CSwap(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_cswap_to_ccnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CSwap to CNot"
    gate = CSwap(0, 1, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_cswap_to_cnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CSwap to CNot (control between targets)"
    gate = CSwap(1, 0, 2)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_cswap_inside_to_cnot(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CH to Clifford+T"
    gate = CH(0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_ch_to_cpt(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CV to Clifford+T"
    gate = CV(0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_cv_to_cpt(gate))
    circuit_identities.append([name, circ0, circ1])

    name = "CV_H to Clifford+T"
    gate = CV_H(0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_cvh_to_cpt(gate))
    circuit_identities.append([name, circ0, circ1])

    # from sympy import Symbol, pi
    name = "CNotPow to ZZ"
    gate = CNotPow(t, 0, 1)
    circ0 = Circuit([gate])
    circ1 = Circuit(translate_cnotpow_to_zz(gate))
    circuit_identities.append([name, circ0, circ1])

    # name = "PISwap to XY"
    # gate = PISwap(theta, 0, 1)
    # circ0 = Circuit([gate])
    # circ1 = Circuit(translate_piswap_to_xy(gate))
    # circuit_identities.append([name, circ0, circ1])

    name = "Powers of iSwap to CNot sandwich"
    gate = ISwap(0, 1) ** t
    trans = [translate_xy_to_can, translate_can_to_cnot]
    circ0 = Circuit([gate])
    circ1 = circuit_translate(circ0, trans)
    circuit_identities.append([name, circ0, circ1])

    name = "Commute ZPow past multiple CNots"
    gate = ZPow(t, 1)
    circ0 = Circuit([gate, CNot(0, 1), CNot(1, 2), CNot(0, 1)])
    circ1 = Circuit([CNot(0, 1), CNot(1, 2), CNot(0, 1), gate])
    circuit_identities.append([name, circ0, circ1])

    return circuit_identities


def _print_circuit_identity(
    name, circ0, circ1, min_col_width=0, col_sep=5, left_margin=8
):
    print("# ", name)

    circ0 = Circuit(circ0)
    circ1 = Circuit(circ1)

    gates0 = circuit_to_diagram(circ0, qubit_labels=False).splitlines()
    gates1 = circuit_to_diagram(circ1, qubit_labels=False).splitlines()

    for gate0, gate1 in zip_longest(gates0, gates1, fillvalue=""):
        line = (" " * col_sep).join(
            [gate0.ljust(min_col_width), gate1.ljust(min_col_width)]
        )
        line = (" " * left_margin) + line
        line = line.rstrip()
        print(line)

    print()
    print()

    concrete = {name: np.random.uniform() for name in syms.values()}
    circ0f = circ0.resolve(concrete)
    circ1f = circ1.resolve(concrete)
    assert gates_close(circ0f.asgate(), circ1f.asgate())


def _check_circuit_identities(circuit_identities):
    for name, circ0, circ1 in circuit_identities:
        _print_circuit_identity(name, circ0, circ1)


if __name__ == "__main__":
    print()
    print("# Validate and report various circuit identities " "(up to global phase)")
    print()
    _check_circuit_identities(identities())
