# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


from itertools import chain
from typing import List, Set, Type

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.translate import translation_source_gate
from quantumflow.visualization import kwarg_to_symbol


@pytest.mark.parametrize("trans", qf.TRANSLATORS.values())  # type: ignore
def test_translators(trans: Type[qf.StdGate]) -> None:
    gatet = translation_source_gate(trans)

    args = [np.random.uniform(-4, 4) for _ in gatet.cv_args]
    qbs = range(10, 10 + gatet.cv_qubit_nb)  # Check that qubits are preserved
    gate = gatet(*chain(args, qbs))

    circ1 = qf.Circuit(trans(gate))  # type: ignore
    print(type(circ1[0]))
    circ1[0].tensor
    circ1.asgate()

    circ1.asgate().tensor

    assert qf.gates_close(gate, circ1.asgate())

    # # FIXME. Many translations currently do not respect phase
    # print('Checking gates phase close...')
    # assert qf.gates_phase_close(gate, circ1.asgate())


concrete = {n: np.random.uniform(-4, 4) for n in kwarg_to_symbol.values()}


@pytest.mark.parametrize("trans", qf.TRANSLATORS.values())  # type: ignore
def test_translators_symbolic(trans: Type[qf.StdGate]) -> None:
    """Check that translations can handle symbolic arguments"""
    gatet = translation_source_gate(trans)
    args = [kwarg_to_symbol[a] for a in gatet.cv_args]
    qbs = range(gatet.cv_qubit_nb)
    gate = gatet(*chain(args, qbs))

    qubits = "abcdefg"[0 : gate.qubit_nb]  # Check that qubits are preserved
    gate = gate.on(*qubits)

    circ0 = qf.Circuit([gate])
    circ1 = qf.Circuit(trans(gate))  # type: ignore

    circ0f = circ0.resolve(concrete)
    circ1f = circ1.resolve(concrete)
    assert qf.gates_close(circ0f.asgate(), circ1f.asgate())


def test_translate() -> None:
    circ0 = qf.Circuit([qf.CSwap(0, 1, 2)])

    translators = [
        qf.translate_cswap_to_ccnot,
        qf.translate_ccnot_to_cnot,
        qf.translate_cnot_to_cz,
    ]
    circ1 = qf.circuit_translate(circ0, translators)
    assert circ1.size() == 33

    circ1 = qf.circuit_translate(circ0, translators, recurse=False)

    qf.gates_close(circ0.asgate(), circ1.asgate())

    circ2 = qf.circuit_translate(circ0)
    qf.gates_close(circ0.asgate(), circ2.asgate())


def test_circuit_translate_targets() -> None:
    circ0 = qf.Circuit([qf.CSwap(0, 1, 2)])
    targets = [qf.Can, qf.XPow, qf.ZPow, qf.I]
    circ1 = qf.circuit_translate(circ0, targets=targets)  # type: ignore
    qf.gates_close(circ0.asgate(), circ1.asgate())


def test_can_to_cnot() -> None:
    gate = qf.Can(0.3, 0.23, 0.22, 0, 1)
    circ = qf.Circuit(qf.translate_can_to_cnot(gate))  # type: ignore
    assert qf.gates_close(gate, circ.asgate())

    gate = qf.Can(0.3, 0.23, 0.0, 0, 1)
    circ = qf.Circuit(qf.translate_can_to_cnot(gate))  # type: ignore
    print(qf.canonical_decomposition(circ.asgate()))
    assert qf.gates_close(gate, circ.asgate())


terminal_2q_gate = (
    qf.Can,
    qf.XX,
    qf.YY,
    qf.ZZ,
    qf.CNot,
    qf.CZ,
)


@pytest.mark.parametrize("term_gate", terminal_2q_gate)
def test_decompose_to_terminal_2q_gate(term_gate: Type[qf.StdGate]) -> None:
    # 1 qubit terminal gates
    # We include identity and global phase because translators can't
    # delete gates
    gates: Set[Type[qf.StdGate]] = {qf.XPow, qf.ZPow, qf.I, qf.Ph}

    gates.add(term_gate)
    trans = qf.select_translators(gates, qf.TRANSLATORS.values())  # type: ignore
    for t in trans:
        gatet = translation_source_gate(t)
        gates.add(gatet)  # type: ignore

    missing = set(qf.StdGate.cv_stdgates.values()) - gates
    if len(missing) != 0:
        print("Missing gates:", missing)
        assert len(missing) == 0


def test_circuit_translate_exception() -> None:
    circ0 = qf.Circuit([qf.CSwap(0, 1, 2)])
    with pytest.raises(ValueError):
        gates: List[Type[qf.StdGate]] = [qf.XPow, qf.ZPow, qf.I, qf.Ph]
        translators = [
            qf.translate_cswap_to_ccnot,
            qf.translate_ccnot_to_cnot,
            qf.translate_cnot_to_cz,
        ]
        qf.circuit_translate(circ0, translators, gates)


# fin
