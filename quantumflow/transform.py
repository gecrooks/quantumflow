# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
QuantumFlow: Translate, transform, and compile circuits.
"""

# Note: Beta Prototype

from typing import Callable, Generator, Set, Tuple

from .circuits import Circuit
from .dagcircuit import DAGCircuit
from .info import almost_identity
from .ops import Gate, Operation
from .stdgates import CZ, ZZ, H, XPow, YPow, ZPow
from .translate import (
    circuit_translate,
    translate_ccnot_to_cnot,
    translate_cnot_to_cz,
    translate_cphase_to_zz,
    translate_cswap_to_ccnot,
    translate_hadamard_to_zxz,
    translate_invt_to_tz,
    translate_invv_to_tx,
    translate_t_to_tz,
    translate_tx_to_zxzxz,
    translate_v_to_tx,
    translate_zz_to_cnot,
)


# FIXME: transpile instead of compile?
def compile_circuit(circ: Circuit) -> Circuit:
    """Compile a circuit to standard gate set (CZ, X^0.5, ZPow),
    simplifying circuit where possible.
    """
    # FIXME: Should be automagic translations
    # Convert multi-qubit gates to CZ gates
    trans = [
        translate_cswap_to_ccnot,
        translate_ccnot_to_cnot,
        translate_cphase_to_zz,
        translate_cnot_to_cz,
        translate_t_to_tz,
        translate_invt_to_tz,
        translate_zz_to_cnot,
        translate_v_to_tx,
        translate_invv_to_tx,
    ]
    circ = circuit_translate(circ, trans)

    dagc = DAGCircuit(circ)
    remove_identites(dagc)
    merge_hadamards(dagc)
    convert_HZH(dagc)

    # Standardize 1-qubit gates
    circ = Circuit(dagc)
    circ = circuit_translate(circ, [translate_hadamard_to_zxz])
    circ = circuit_translate(circ, [translate_tx_to_zxzxz], recurse=False)

    # Gather and merge ZPow gates
    dagc = DAGCircuit(circ)
    retrogress_tz(dagc)
    merge_tz(dagc)
    remove_identites(dagc)

    circ = Circuit(dagc)

    return circ


def find_pattern(
    dagc: DAGCircuit,
    gateset1: Set,
    gateset2: Set,
) -> Generator[Tuple[Operation, Operation], None, None]:
    """Find where a gate from gateset1 is followed by a gate from gateset2 in
    a DAGCircuit"""
    for elem2 in dagc:
        if type(elem2) not in gateset2:
            continue

        for q2 in elem2.qubits:
            elem1 = dagc.prev_element(elem2, q2)
            if type(elem1) not in gateset1:
                continue
            yield (elem1, elem2)


def remove_element(dagc: DAGCircuit, elem: Operation) -> None:
    """Remove a node from a DAGCircuit"""

    for qubit in elem.qubits:
        prv = dagc.prev_element(elem, qubit)
        nxt = dagc.next_element(elem, qubit)
        dagc.graph.add_edge(prv, nxt, key=qubit)
    dagc.graph.remove_node(elem)


def remove_identites(dagc: DAGCircuit) -> None:
    """Remove identities from a DAGCircuit"""
    for elem in dagc:
        if isinstance(elem, Gate) and almost_identity(elem):
            remove_element(dagc, elem)


def merge_hadamards(dagc: DAGCircuit) -> None:
    """Merge and remove neighboring Hadamard gates"""
    for elem1, elem2 in find_pattern(dagc, {H}, {H}):
        remove_element(dagc, elem1)
        remove_element(dagc, elem2)


def merge_tx(dagc: DAGCircuit) -> None:
    """Merge neighboring ZPow gates"""
    _merge_turns(dagc, XPow)


def merge_ty(dagc: DAGCircuit) -> None:
    """Merge neighboring ZPow gates"""
    _merge_turns(dagc, YPow)


def merge_tz(dagc: DAGCircuit) -> None:
    """Merge neighboring ZPow gates"""
    _merge_turns(dagc, ZPow)


def _merge_turns(dagc: DAGCircuit, gate_class: Callable) -> None:
    for gate0, gate1 in find_pattern(dagc, {gate_class}, {gate_class}):
        t = gate0.param("t") + gate1.param("t")
        (qubit,) = gate0.qubits
        gate = gate_class(t, qubit)

        prv = dagc.prev_element(gate0)
        nxt = dagc.next_element(gate1)
        dagc.graph.add_edge(prv, gate, key=qubit)
        dagc.graph.add_edge(gate, nxt, key=qubit)

        dagc.graph.remove_node(gate0)
        dagc.graph.remove_node(gate1)


def retrogress_tz(dagc: DAGCircuit) -> None:
    """Commute ZPow gates as far backward in the circuit as possible"""
    G = dagc.graph
    again = True
    while again:
        again = False
        for elem1, elem2 in find_pattern(dagc, {ZZ, CZ}, {ZPow}):
            (q,) = elem2.qubits
            elem0 = dagc.prev_element(elem1, q)
            elem3 = dagc.next_element(elem2, q)

            G.remove_edge(elem0, elem1, q)
            G.remove_edge(elem1, elem2, q)
            G.remove_edge(elem2, elem3, q)

            G.add_edge(elem0, elem2, key=q)
            G.add_edge(elem2, elem1, key=q)
            G.add_edge(elem1, elem3, key=q)
            again = True


# TODO: Rename? merge_hzh
# TODO: larger pattern, simplifying sequences of 1-qubit Clifford gates
def convert_HZH(dagc: DAGCircuit) -> None:
    """Convert a sequence of H-ZPow-H gates to a XPow gate"""
    for elem2, elem3 in find_pattern(dagc, {ZPow}, {H}):
        elem1 = dagc.prev_element(elem2)
        if not isinstance(elem1, H):
            continue

        prv = dagc.prev_element(elem1)
        nxt = dagc.next_element(elem3)

        t = elem2.param("t")
        (q0,) = elem2.qubits
        gate = XPow(t, q0)

        dagc.graph.remove_node(elem1)
        dagc.graph.remove_node(elem2)
        dagc.graph.remove_node(elem3)

        dagc.graph.add_edge(prv, gate, key=q0)
        dagc.graph.add_edge(gate, nxt, key=q0)


# fin
