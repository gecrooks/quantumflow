
"""
QuantumFlow: Translate, transform, and compile circuits.
"""

# FIXME: Rename to transform?


from typing import Generator, Tuple, Set, Callable
from .ops import Operation, Gate
from .circuits import Circuit
from .gates import (TX, TZ, ZZ, CZ, H)
from .dagcircuit import DAGCircuit
from .gates import almost_identity
from .translate import (
    translate_cswap_to_ccnot,
    translate_ccnot_to_cnot,
    translate_cphase_to_zz,
    translate_cnot_to_cz,
    translate_t_to_tz,
    translate_invt_to_tz,
    translate_zz_to_cnot,
    translate_hadamard_to_zxz,
    translate_tx_to_zxzxz,
    translate_v_to_tx,
    translate_invv_to_tx,
    translate)


# FIXME: transpile instead of compile?
def compile_circuit(circ: Circuit) -> Circuit:
    """Compile a circuit to standard gate set (CZ, X^0.5, TZ),
    simplifing circuit where possible.
    """

    # Convert multi-qubit gates to CZ gates
    trans = [translate_cswap_to_ccnot,
             translate_ccnot_to_cnot,
             translate_cphase_to_zz,
             translate_cnot_to_cz,
             translate_t_to_tz,
             translate_invt_to_tz,
             translate_zz_to_cnot,
             translate_v_to_tx,
             translate_invv_to_tx,
             ]
    circ = translate(circ, trans)

    dagc = DAGCircuit(circ)
    remove_identites(dagc)
    merge_hadamards(dagc)
    convert_HZH(dagc)

    # Standardize 1-qubit gates
    circ = Circuit(dagc)
    circ = translate(circ, [translate_hadamard_to_zxz])
    circ = translate(circ, [translate_tx_to_zxzxz], recurse=False)

    # Gather and merge TZ gates
    dagc = DAGCircuit(circ)
    retrogress_tz(dagc)
    merge_tz(dagc)
    remove_identites(dagc)

    circ = Circuit(dagc)

    return circ


def find_pattern(dagc: DAGCircuit,
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
    """Remove identites from a DAGCircuit"""
    for elem in dagc:
        if isinstance(elem, Gate) and almost_identity(elem):
            remove_element(dagc, elem)


def merge_hadamards(dagc: DAGCircuit) -> None:
    """Merge and remove neighbouring Hadamard gates"""
    for elem1, elem2 in find_pattern(dagc, {H}, {H}):
        remove_element(dagc, elem1)
        remove_element(dagc, elem2)


def merge_tz(dagc: DAGCircuit) -> None:
    """Merge neighbouring TZ gates"""
    _merge_turns(dagc, TZ)


# def merge_tx(dagc: DAGCircuit) -> None:
#     """Merge neighbouring TZ gates"""
#     _merge_turns(dagc, TX)


# def merge_ty(dagc: DAGCircuit) -> None:
#     """Merge neighbouring TZ gates"""
#     _merge_turns(dagc, TY)


def _merge_turns(dagc: DAGCircuit, gate_class: Callable) -> None:
    for gate0, gate1 in find_pattern(dagc, {gate_class}, {gate_class}):
        t = gate0.params['t'] + gate1.params['t']
        qubit, = gate0.qubits
        gate = gate_class(t, qubit)

        prv = dagc.prev_element(gate0)
        nxt = dagc.next_element(gate1)
        dagc.graph.add_edge(prv, gate, key=qubit)
        dagc.graph.add_edge(gate, nxt, key=qubit)

        dagc.graph.remove_node(gate0)
        dagc.graph.remove_node(gate1)


def retrogress_tz(dagc: DAGCircuit) -> None:
    """Commute TZ gates as far backward in the circuit as possible"""
    G = dagc.graph
    again = True
    while again:
        again = False
        for elem1, elem2 in find_pattern(dagc, {ZZ, CZ}, {TZ}):
            q, = elem2.qubits
            elem0 = dagc.prev_element(elem1, q)
            elem3 = dagc.next_element(elem2, q)

            G.remove_edge(elem0, elem1, q)
            G.remove_edge(elem1, elem2, q)
            G.remove_edge(elem2, elem3, q)

            G.add_edge(elem0, elem2, key=q)
            G.add_edge(elem2, elem1, key=q)
            G.add_edge(elem1, elem3, key=q)
            again = True


def convert_HZH(dagc: DAGCircuit) -> None:
    """Convert a sequence of H-TZ-H gates to a TX gate"""
    for elem2, elem3 in find_pattern(dagc, {TZ}, {H}):
        elem1 = dagc.prev_element(elem2)
        if not isinstance(elem1, H):
            continue

        prv = dagc.prev_element(elem1)
        nxt = dagc.next_element(elem3)

        t = elem2.params['t']
        q0, = elem2.qubits
        gate = TX(t, q0)

        dagc.graph.remove_node(elem1)
        dagc.graph.remove_node(elem2)
        dagc.graph.remove_node(elem3)

        dagc.graph.add_edge(prv, gate, key=q0)
        dagc.graph.add_edge(gate, nxt, key=q0)
