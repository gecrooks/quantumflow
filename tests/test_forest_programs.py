
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np

import pytest
pytest.importorskip("pyquil")      # noqa: 402

import quantumflow as qf
from quantumflow import forest

from quantumflow.forest.programs import TARGETS, PC


def test_empty_program():
    prog = forest.Program()
    ket = prog.run()
    assert ket.qubits == ()
    assert ket.qubit_nb == 0


def test_nop():
    prog = forest.Program()
    prog += forest.Nop()
    prog.run()

    for inst in prog:
        assert inst is not None

    assert forest.Nop().qubits == ()
    assert forest.Nop().qubit_nb == 0


def test_nop_evolve():
    prog = forest.Program()
    prog += forest.Nop()
    prog.evolve()


def test_compile_label():
    prog = forest.Program()
    prog += forest.Label('Here')
    prog += forest.Nop()
    prog += forest.Label('There')

    ket = prog.run()

    assert ket.memory[TARGETS] == {'Here': 0, 'There': 2}


def test_jump():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 0)
    prog += forest.Jump('There')
    prog += forest.Not(ro[0])
    prog += forest.Label('There')
    prog += forest.Not(ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 1

    prog += forest.JumpWhen('There', ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 0

    prog += forest.Not(ro[0])
    prog += forest.JumpUnless('There', ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 1


def test_wait():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 0)
    prog += forest.Wait()
    prog += forest.Not(ro[0])
    prog += forest.Wait()
    prog += forest.Not(ro[0])
    prog += forest.Wait()
    prog += forest.Not(ro[0])

    ket = prog.run()
    assert ket.memory[ro[0]] == 1


def test_include():
    prog = forest.Program()
    prog += forest.Move(('a', 0), 0)
    instr = forest.Include('somefile.quil', forest.Program())
    assert instr.quil() == 'INCLUDE "somefile.quil"'


def test_halt():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 0)
    prog += forest.Halt()
    prog += forest.Not(ro[0])

    ket = prog.run()
    assert ket.memory[PC] == -1
    assert ket.memory[ro[0]] == 0


def test_reset():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 1)
    prog += forest.Call('X', params=[], qubits=[0])
    prog += forest.Reset()
    prog += forest.Measure(0, ro[1])
    ket = prog.run()

    assert ket.qubits == (0,)
    assert ket.memory[ro[0]] == 1
    assert ket.memory[ro[1]] == 0


def test_reset_one():
    prog = forest.Program()
    prog += forest.Call('X', params=[], qubits=[0])
    prog += forest.Call('X', params=[], qubits=[1])
    prog += forest.Reset(0)
    prog += forest.Measure(0, ('b', 0))
    prog += forest.Measure(1, ('b', 1))
    ket = prog.run()

    assert ket.memory[('b', 0)] == 0
    assert ket.memory[('b', 1)] == 1


def test_xgate():
    prog = forest.Program()
    prog += forest.Call('X', params=[], qubits=[0])
    ket = prog.run()

    assert ket.qubits == (0,)
    # assert prog.cbits == []

    qf.print_state(ket)


def test_call():
    prog = forest.Program()
    prog += forest.Call('BELL', params=[], qubits=[])

    assert str(prog) == 'BELL\n'


# FIXME: ref-qvm has do_until option? From pyquil?
def test_measure_until():

    prog = forest.Program()
    prog += forest.Move(('c', 2), 1)
    prog += forest.Label('redo')
    prog += forest.Call('X', [], [0])
    prog += forest.Call('H', [], [0])
    prog += forest.Measure(0, ('c', 2))
    prog += forest.JumpUnless('redo', ('c', 2))

    ket = prog.run()
    assert ket.memory[('c', 2)] == 1


def test_belltest():
    prog = forest.Program()
    prog += forest.Call('H', [], [0])
    prog += forest.Call('CNOT', [], [0, 1])
    ket = prog.run()

    assert qf.states_close(ket, qf.ghz_state(2))
    # qf.print_state(prog.state)


def test_occupation_basis():
    prog = forest.Program()
    prog += forest.Call('X', [], [0])
    prog += forest.Call('X', [], [1])
    prog += forest.Call('I', [], [2])
    prog += forest.Call('I', [], [3])

    ket = prog.run()

    assert ket.qubits == (0, 1, 2, 3)
    probs = qf.asarray(ket.probabilities())
    assert probs[1, 1, 0, 0] == 1.0
    assert probs[1, 1, 0, 1] == 0.0

# TODO: TEST EXP CIRCUIT from test_qvm


def test_qaoa_circuit():
    wf_true = [0.00167784 + 1.00210180e-05*1j, 0.50000000 - 4.99997185e-01*1j,
               0.50000000 - 4.99997185e-01*1j, 0.00167784 + 1.00210180e-05*1j]
    prog = forest.Program()
    prog += forest.Call('RY', [np.pi/2], [0])
    prog += forest.Call('RX', [np.pi], [0])
    prog += forest.Call('RY', [np.pi/2], [1])
    prog += forest.Call('RX', [np.pi], [1])
    prog += forest.Call('CNOT', [], [0, 1])
    prog += forest.Call('RX', [-np.pi/2], [1])
    prog += forest.Call('RY', [4.71572463191], [1])
    prog += forest.Call('RX', [np.pi/2], [1])
    prog += forest.Call('CNOT', [], [0, 1])
    prog += forest.Call('RX', [-2*2.74973750579], [0])
    prog += forest.Call('RX', [-2*2.74973750579], [1])

    test_state = prog.run()
    true_state = qf.State(wf_true)
    assert qf.states_close(test_state, true_state)


QUILPROG = """RY(pi/2) 0
RX(pi) 0
RY(pi/2) 1
RX(pi) 1
CNOT 0 1
RX(-pi/2) 1
RY(4.71572463191) 1
RX(pi/2) 1
CNOT 0 1
RX(-5.49947501158) 0
RX(-5.49947501158) 1
"""


# HADAMARD = """
# DEFGATE HADAMARD:
#     1/sqrt(2), 1/sqrt(2)
#     1/sqrt(2), -1/sqrt(2)
# HADAMARD 0
# """


# def test_defgate():
#     prog = qf.forest.quil_to_program(HADAMARD)
#     ket = prog.run()
#     qf.print_state(ket)
#     assert qf.states_close(ket, qf.ghz_state(1))


# CP = """
# DEFGATE CP(%theta):
#     1, 0, 0, 0
#     0, 1, 0, 0
#     0, 0, 1, 0
#     0, 0, 0, cis(pi+%theta)

# X 0
# X 1
# CP(1.0) 0 1
# """


# def test_defgate_param():
#     prog = qf.forest.quil_to_program(CP)
#     # ket0 = prog.compile()
#     # qf.print_state(ket0)
#     ket1 = prog.run()
#     qf.print_state(ket1)

#     ket = qf.zero_state(2)
#     ket = qf.X(0).run(ket)
#     ket = qf.X(1).run(ket)
#     ket = qf.CPHASE(1.0, 0, 1).run(ket)
#     qf.print_state(ket)

#     assert qf.states_close(ket1, ket)


CIRC0 = """DEFCIRCUIT CIRCX:
    X 0

CIRCX 0
"""


def test_defcircuit():
    prog = forest.Program()

    circ = forest.DefCircuit('CIRCX', {})
    circ += forest.Call('X', params=[], qubits=[0])

    prog += circ
    prog += forest.Call('CIRCX', params=[], qubits=[0])

    assert str(prog) == CIRC0

    # prog.compile()
    assert prog.qubits == [0]

    # FIXME: Not implemented
    # prog.run()
    # qf.print_state(prog.state)


CIRC1 = """DEFCIRCUIT CIRCX(this):
    NOP

"""


def test_defcircuit_param():
    prog = forest.Program()
    circ = forest.DefCircuit('CIRCX', {'this': None})
    circ += forest.Nop()
    prog += circ
    assert str(prog) == CIRC1


def test_exceptions():
    prog = forest.Program()
    prog += forest.Call('NOT_A_GATE', [], [0])
    with pytest.raises(RuntimeError):
        prog.run()


def test_neg():
    c = forest.Register('c')
    assert str(forest.Neg(c[10])) == 'NEG c[10]'


def test_logics():
    c = forest.Register('c')

    circ = qf.Circuit([forest.Move(c[0], 0),
                       forest.Move(c[1], 1),
                       forest.And(c[0], c[1])])
    # assert len(circ) == 3     # FIXME
    # assert circ.cbits == [c[0], c[1]] # FIXME

    ket = circ.run()
    assert ket.memory == {c[0]: 0, c[1]: 1}

    circ += forest.Not(c[1])
    circ += forest.And(c[0], c[1])
    ket = circ.run(ket)
    assert ket.memory == {c[0]: 0, c[1]: 0}

    circ = forest.Circuit()
    circ += forest.Move(c[0], 0)
    circ += forest.Move(c[1], 1)
    circ += forest.Ior(c[0], c[1])
    ket = circ.run()
    assert ket.memory == {c[0]: 1, c[1]: 1}

    circ = forest.Circuit()
    circ += forest.Move(c[0], 1)
    circ += forest.Move(c[1], 1)
    circ += forest.Xor(c[0], c[1])
    ket = circ.run()
    assert ket.memory == {c[0]: 0, c[1]: 1}

    circ += forest.Exchange(c[0], c[1])
    ket = circ.run(ket)
    assert ket.memory == {c[0]: 1, c[1]: 0}

    circ += forest.Move(c[0], c[1])
    ket = circ.run(ket)
    assert ket.memory == {c[0]: 0, c[1]: 0}


def test_add():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 1)
    prog += forest.Move(ro[1], 2)
    prog += forest.Add(ro[0], ro[1])
    prog += forest.Add(ro[0], 4)
    ket = prog.run()
    assert ket.memory[ro[0]] == 7


def test_density_add():
    ro = forest.Register()
    circ = forest.Circuit()
    circ += forest.Move(ro[0], 1)
    circ += forest.Move(ro[1], 2)
    circ += forest.Add(ro[0], ro[1])
    circ += forest.Add(ro[0], 4)
    rho = circ.evolve()
    assert rho.memory[ro[0]] == 7


def test_mul():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 1)
    prog += forest.Move(ro[1], 2)
    prog += forest.Mul(ro[0], ro[1])
    prog += forest.Mul(ro[0], 4)
    ket = prog.run()
    assert ket.memory[ro[0]] == 8


def test_div():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 4)
    prog += forest.Move(ro[1], 1)
    prog += forest.Div(ro[0], ro[1])
    prog += forest.Div(ro[0], 2)
    ket = prog.run()
    assert ket.memory[ro[0]] == 2


def test_sub():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 1)
    prog += forest.Move(ro[1], 2)
    prog += forest.Sub(ro[0], ro[1])
    prog += forest.Sub(ro[0], 4)
    prog += forest.Neg(ro[0])
    ket = prog.run()
    assert ket.memory[ro[0]] == 5


def test_comparisions():
    ro = forest.Register()
    prog = forest.Program()
    prog += forest.Move(ro[0], 1)
    prog += forest.Move(ro[1], 2)
    prog += forest.EQ(('eq', 0), ro[0], ro[1])
    prog += forest.GT(('gt', 0), ro[0], ro[1])
    prog += forest.GE(('ge', 0), ro[0], ro[1])
    prog += forest.LT(('lt', 0), ro[0], ro[1])
    prog += forest.LE(('le', 0), ro[0], ro[1])
    ket = prog.run()
    assert ket.memory[('eq', 0)] == 0
    assert ket.memory[('gt', 0)] == 0
    assert ket.memory[('ge', 0)] == 0
    assert ket.memory[('lt', 0)] == 1
    assert ket.memory[('le', 0)] == 1
