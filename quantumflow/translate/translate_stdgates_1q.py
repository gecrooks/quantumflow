# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# 1-qubit gates

from typing import Iterator, Union

from .. import var
from ..stdgates import (
    S_H,
    T_H,
    U2,
    U3,
    V_H,
    H,
    HPow,
    PhasedX,
    PhasedXPow,
    PhaseShift,
    Rn,
    Rx,
    Ry,
    Rz,
    S,
    SqrtY,
    SqrtY_H,
    T,
    V,
    X,
    XPow,
    Y,
    YPow,
    Z,
    ZPow,
)
from .translations import register_translation

# 1-qubit gates


@register_translation
def translate_x_to_tx(gate: X) -> Iterator[XPow]:
    """Translate X gate to XPow"""
    (q0,) = gate.qubits
    yield XPow(1, q0)


@register_translation
def translate_y_to_ty(gate: Y) -> Iterator[YPow]:
    """Translate Y gate to YPow"""
    (q0,) = gate.qubits
    yield YPow(1, q0)


@register_translation
def translate_z_to_tz(gate: Z) -> Iterator[ZPow]:
    """Translate Z gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(1, q0)


@register_translation
def translate_s_to_tz(gate: S) -> Iterator[ZPow]:
    """Translate S gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(0.5, q0)


@register_translation
def translate_t_to_tz(gate: T) -> Iterator[ZPow]:
    """Translate T gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(0.25, q0)


@register_translation
def translate_invs_to_tz(gate: S_H) -> Iterator[ZPow]:
    """Translate S.H gate to ZPow"""
    (q0,) = gate.qubits
    yield ZPow(-0.5, q0)


@register_translation
def translate_invt_to_tz(gate: T_H) -> Iterator[ZPow]:
    """Translate inverse T gate to Rz (a quil standard gate)"""
    (q0,) = gate.qubits
    yield ZPow(-0.25, q0)


@register_translation
def translate_rx_to_tx(gate: Rx) -> Iterator[XPow]:
    """Translate Rx gate to XPow"""
    (q0,) = gate.qubits
    (theta,) = gate.params
    t = theta / var.PI
    yield XPow(t, q0)


@register_translation
def translate_ry_to_ty(gate: Ry) -> Iterator[YPow]:
    """Translate Ry gate to YPow"""
    (q0,) = gate.qubits
    (theta,) = gate.params
    t = theta / var.PI
    yield YPow(t, q0)


@register_translation
def translate_rz_to_tz(gate: Rz) -> Iterator[ZPow]:
    """Translate Rz gate to ZPow"""
    (q0,) = gate.qubits
    (theta,) = gate.params
    t = theta / var.PI
    yield ZPow(t, q0)


@register_translation
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


@register_translation
def translate_phase_to_rz(gate: PhaseShift) -> Iterator[Rz]:
    """Translate Phase gate to Rz (ignoring global phase)"""
    (q0,) = gate.qubits
    theta = gate.param("theta")
    yield Rz(theta, q0)


@register_translation
def translate_sqrty_to_ty(gate: SqrtY) -> Iterator[YPow]:
    """Translate sqrt-Y gate to YPow"""
    (q0,) = gate.qubits
    yield YPow(0.5, q0)


@register_translation
def translate_sqrty_h_to_ty(gate: SqrtY_H) -> Iterator[YPow]:
    """Translate sqrt-Y gate to YPow"""
    (q0,) = gate.qubits
    yield YPow(-0.5, q0)


@register_translation
def translate_tx_to_rx(gate: XPow) -> Iterator[Rx]:
    """Translate XPow gate to Rx"""
    (q0,) = gate.qubits
    theta = gate.param("t") * var.PI
    yield Rx(theta, q0)


@register_translation
def translate_ty_to_ry(gate: YPow) -> Iterator[Ry]:
    """Translate YPow gate to Ry"""
    (q0,) = gate.qubits
    theta = gate.param("t") * var.PI
    yield Ry(theta, q0)


@register_translation
def translate_tz_to_rz(gate: ZPow) -> Iterator[Rz]:
    """Translate ZPow gate to Rz"""
    (q0,) = gate.qubits
    theta = gate.param("t") * var.PI
    yield Rz(theta, q0)


@register_translation
def translate_ty_to_xzx(gate: YPow) -> Iterator[Union[XPow, ZPow]]:
    """Translate YPow gate to XPow and ZPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield XPow(0.5, q0)
    yield ZPow(t, q0)
    yield XPow(-0.5, q0)


@register_translation
def translate_tx_to_zyz(gate: XPow) -> Iterator[Union[YPow, S, S_H]]:
    """Translate XPow gate to S and YPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield S(q0)
    yield YPow(t, q0)
    yield S_H(q0)


@register_translation
def translate_tz_to_xyx(gate: ZPow) -> Iterator[Union[YPow, V, V_H]]:
    """Translate ZPow gate to V and YPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield V_H(q0)
    yield YPow(t, q0)
    yield V(q0)


@register_translation
def translate_phased_x_to_zxz(gate: PhasedX) -> Iterator[Union[X, ZPow]]:
    """Translate YPow gate to XPow and ZPow gates"""
    (q0,) = gate.qubits
    p = gate.param("p")
    yield ZPow(-p, q0)
    yield X(q0)
    yield ZPow(p, q0)


@register_translation
def translate_phased_tx_to_zxz(gate: PhasedXPow) -> Iterator[Union[XPow, ZPow]]:
    """Translate YPow gate to XPow and ZPow gates"""
    (q0,) = gate.qubits
    p, t = gate.params
    yield ZPow(-p, q0)
    yield XPow(t, q0)
    yield ZPow(p, q0)


@register_translation
def translate_v_to_tx(gate: V) -> Iterator[XPow]:
    """Translate V gate to XPow"""
    (q0,) = gate.qubits
    yield XPow(0.5, q0)


@register_translation
def translate_invv_to_tx(gate: V_H) -> Iterator[XPow]:
    """Translate V_H gate to XPow"""
    (q0,) = gate.qubits
    yield XPow(-0.5, q0)


@register_translation
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


@register_translation
def translate_ty_to_zxz(gate: YPow) -> Iterator[Union[XPow, S, S_H]]:
    """Translate YPow gate to ZPow and XPow gates"""
    (q0,) = gate.qubits
    t = gate.param("t")
    yield S_H(q0)
    yield XPow(t, q0)
    yield S(q0)


@register_translation
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


@register_translation
def translate_hadamard_to_zxz(gate: H) -> Iterator[Union[XPow, ZPow]]:
    """Convert a Hadamard gate to a circuit with ZPow and XPow gates."""
    (q0,) = gate.qubits
    yield ZPow(0.5, q0)
    yield XPow(0.5, q0)
    yield ZPow(0.5, q0)


@register_translation
def translate_u3_to_zyz(gate: U3) -> Iterator[Union[Rz, Ry]]:
    """Translate QASMs U3 gate to Rz and Ry"""
    (q0,) = gate.qubits
    theta, phi, lam = gate.params
    yield Rz(lam, q0)
    yield Ry(theta, q0)
    yield Rz(phi, q0)


@register_translation
def translate_u2_to_zyz(gate: U2) -> Iterator[Union[Rz, Ry]]:
    """Translate QASMs U2 gate to Rz and Ry"""
    (q0,) = gate.qubits
    phi, lam = gate.params
    yield Rz(lam, q0)
    yield Ry(var.PI / 2, q0)
    yield Rz(phi, q0)


@register_translation
def translate_tx_to_hzh(gate: XPow) -> Iterator[Union[H, ZPow]]:
    """Convert a XPow gate to a circuit with Hadamard and ZPow gates"""
    (q0,) = gate.qubits
    (t,) = gate.params
    yield H(q0)
    yield ZPow(t, q0)
    yield H(q0)


# fin
