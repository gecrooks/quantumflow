

import numpy as np

import quantumflow as qf
from quantumflow.translate import translation_source_gate
from quantumflow.visualization import kwarg_to_symbol
import pytest


@pytest.mark.parametrize("trans", qf.TRANSLATORS.values())
def test_translators(trans):
    gatet = translation_source_gate(trans)

    args = [np.random.uniform(-4, 4) for _ in gatet.args()]
    gate = gatet(*args)
    qubits = range(10, 10+gate.qubit_nb)    # Check that qubits are preserved
    gate = gate.on(*qubits)

    circ1 = qf.Circuit(trans(gate))

    assert qf.gates_close(gate, circ1.asgate())

    # # FIXME. Many translations currently do not respect phase
    # print('Checking gates phase close...')
    # assert qf.gates_phase_close(gate, circ1.asgate())


concrete = {n: np.random.uniform(-4, 4) for n in kwarg_to_symbol.values()}
@pytest.mark.parametrize("trans", qf.TRANSLATORS.values())
def test_translators_symbolic(trans):
    """Check that translations can handle symbolic arguments"""
    gatet = translation_source_gate(trans)
    args = [kwarg_to_symbol[a] for a in gatet.args()]
    gate = gatet(*args)

    qubits = 'abcdefg'[0: gate.qubit_nb]    # Check that qubits are preserved
    gate = gate.on(*qubits)

    circ0 = qf.Circuit([gate])
    circ1 = qf.Circuit(trans(gate))

    circ0f = circ0.resolve(concrete)
    circ1f = circ1.resolve(concrete)
    assert qf.gates_close(circ0f.asgate(), circ1f.asgate())


def test_translate():
    circ0 = qf.Circuit([qf.CSWAP(0, 1, 2)])

    translators = [qf.translate_cswap_to_ccnot,
                   qf.translate_ccnot_to_cnot,
                   qf.translate_cnot_to_cz]
    circ1 = qf.circuit_translate(circ0, translators)
    assert circ1.size() == 33

    circ1 = qf.circuit_translate(circ0, translators, recurse=False)

    qf.gates_close(circ0.asgate(), circ1.asgate())

    circ2 = qf.circuit_translate(circ0)
    qf.gates_close(circ0.asgate(), circ2.asgate())


def test_circuit_translate_targets():
    circ0 = qf.Circuit([qf.CSWAP(0, 1, 2)])
    targets = {qf.CAN, qf.TX, qf.TZ, qf.I}
    circ1 = qf.circuit_translate(circ0, targets=targets)
    qf.gates_close(circ0.asgate(), circ1.asgate())


def test_can_to_cnot():
    gate = qf.CAN(0.3, 0.23, 0.22)
    circ = qf.Circuit(qf.translate_can_to_cnot(gate))
    assert qf.gates_close(gate, circ.asgate())

    gate = qf.CAN(0.3, 0.23, 0)
    circ = qf.Circuit(qf.translate_can_to_cnot(gate))
    print(qf.canonical_decomposition(circ.asgate()))
    assert qf.gates_close(gate, circ.asgate())


terminal_2q_gate = (
    qf.CAN,
    qf.XX,
    qf.YY,
    qf.ZZ,
    qf.CNOT,
    qf.CZ,
    # qf.ISWAP,  # FIXME
    qf.SqrtISwap,
    qf.SqrtISwap_H,
    qf.SqrtSwap,
    # qf.SqrtISwap_H, # FIXME
    )


@pytest.mark.parametrize("term_gate", terminal_2q_gate)
def test_decompose_to_terminal_2q_gate(term_gate):
    # 1 qubit terminal gates
    # We include identity and global phase because translators can't
    # delete gates
    gates = {qf.TX, qf.TZ, qf.I, qf.Ph}

    gates.add(term_gate)
    trans = qf.select_translators(gates, qf.TRANSLATORS.values())
    for t in trans:
        gatet = translation_source_gate(t)
        gates.add(gatet)

    missing = qf.STD_GATESET - gates
    print("Missing gates:", missing)
    assert len(missing) == 0


def test_circuit_translate_exception():
    circ0 = qf.Circuit([qf.CSWAP(0, 1, 2)])
    with pytest.raises(ValueError):
        gates = {qf.TX, qf.TZ, qf.I, qf.Ph}
        translators = [qf.translate_cswap_to_ccnot,
                       qf.translate_ccnot_to_cnot,
                       qf.translate_cnot_to_cz]
        qf.circuit_translate(circ0, translators, gates)
