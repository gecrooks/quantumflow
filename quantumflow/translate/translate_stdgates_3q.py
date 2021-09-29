# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


# 3-qubit gates

from typing import Iterator, Union

from .. import var
from ..stdgates import (
    CCZ,
    CV,
    CV_H,
    S_H,
    T_H,
    V_H,
    Barenco,
    CCiX,
    CCNot,
    CCXPow,
    CISwap,
    CNot,
    CNotPow,
    CSwap,
    Deutsch,
    H,
    Margolus,
    S,
    T,
    V,
)
from .translations import register_translation


@register_translation
def translate_ccix_to_cnot(gate: CCiX) -> Iterator[Union[CNot, H, T, T_H]]:
    """Decompose doubly-controlled iX-gate to 4 CNots.

    ::

        ───●────     ───────────────●────────────────●────────────
           │                        │                │
        ───●────  =  ───────●───────┼────────●───────┼────────────
           │                │       │        │       │
        ───iX───     ───H───X───T───X───T⁺───X───T───X───T⁺───H───



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


@register_translation
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


@register_translation
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


@register_translation
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


@register_translation
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


@register_translation
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


@register_translation
def translate_ccxpow_to_cnotpow(gate: CCXPow) -> Iterator[Union[CNot, CNotPow]]:
    """Decomposition of powers of CCNot gates to powers of CNot gates."""
    q0, q1, q2 = gate.qubits
    (t,) = gate.params
    yield CNotPow(t / 2, q1, q2)
    yield CNot(q0, q1)
    yield CNotPow(-t / 2, q1, q2)
    yield CNot(q0, q1)
    yield CNotPow(t / 2, q0, q2)


@register_translation
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


@register_translation
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


@register_translation
def translate_ciswap_to_ccix(gate: CISwap) -> Iterator[Union[CNot, CCiX]]:
    """Translate a controlled-iswap gate to CCiX"""
    q0, q1, q2 = gate.qubits
    yield CNot(q2, q1)
    yield CCiX(q0, q1, q2)
    yield CNot(q2, q1)


@register_translation
def translate_cswap_to_ccnot(gate: CSwap) -> Iterator[Union[CNot, CCNot]]:
    """Convert a CSwap gate to a circuit with a CCNot and 2 CNots"""
    q0, q1, q2 = gate.qubits
    yield CNot(q2, q1)
    yield CCNot(q0, q1, q2)
    yield CNot(q2, q1)


@register_translation
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


@register_translation
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


@register_translation
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


@register_translation
def translate_margolus_to_cnot(
    gate: Margolus,
) -> Iterator[Union[CNot, V, V_H, T, T_H]]:
    """Decomposition of Margolus gate to 3 CNots.

    (Note that V_H T V = Ry(pi/4) up to phase)
    ::

        ─────────────────────────────────────●───────────────────────────────────
                                             │
        ──────────────────●──────────────────┼─────────────────●─────────────────
                          │                  │                 │
        ───V_H───T⁺───V───X───V_H───T⁺───V───X───V_H───T───V───X───V_H───T───V───
    """
    q0, q1, q2 = gate.qubits

    yield V(q2).H
    yield T(q2).H
    yield V(q2)
    yield CNot(q1, q2)
    yield V(q2).H
    yield T(q2).H
    yield V(q2)
    yield CNot(q0, q2)
    yield V(q2).H
    yield T(q2)
    yield V(q2)
    yield CNot(q1, q2)
    yield V(q2).H
    yield T(q2)
    yield V(q2)


# FIXME
# @register_translation
# def translate_ccnot_to_margolus(gate: Margolus) -> Iterator[Union[CCNot, X, CCZ]]:
#     """Decomposition of a Toffoli gate to a Margolus gate ("Simplified Toffoli")
#     plus a CCZ.

#     ::
#         ───────●───────Margolus_0───
#                │           │
#         ───X───●───X───Margolus_1───
#                │           │
#         ───────●───────Margolus_2───
#     """
#     q0, q1, q2 = gate.qubits

#     yield X(q1)
#     yield CCZ(q0, q1, q2)
#     yield X(q1)
#     yield CCNot(q0, q1, q2)


# fin
