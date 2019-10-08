
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import math
import cmath

import pytest
pytest.importorskip("pyquil")      # noqa: 402

from quantumflow import xforest as forest
from quantumflow.utils import cis

from .. import ALMOST_ZERO


def _test(quil_string, *instructions):
    prog0 = forest.quil_to_program(quil_string)
    prog1 = forest.Program(instructions)

    assert prog0.quil() == prog1.quil()


def test_empty():
    _test("")


def test_simple():
    simple = "HALT", "WAIT", "NOP", "RESET"

    for quil in simple:
        prog = forest.quil_to_program(quil)
        assert len(prog) == 1
        assert str(prog[0]) == quil


def test_reset_qubit():
    _test('RESET 1', forest.Reset(1))


def test_math():
    def get_arg(prog):
        return list(prog[0].params.values())[0]

    arg = get_arg(forest.quil_to_program("RX(1) 0"))
    assert arg == 1

    arg = get_arg(forest.quil_to_program("RX(2.9) 0"))
    assert arg == 2.9

    arg = get_arg(forest.quil_to_program("RX(-2.9) 0"))
    assert arg == -2.9

    arg = get_arg(forest.quil_to_program("RX(+2.9) 0"))
    assert arg == +2.9

    arg = get_arg(forest.quil_to_program("RX(+2.9/4.2) 0"))
    assert arg == +2.9/4.2

    arg = get_arg(forest.quil_to_program("RX(+2.9*4.2) 0"))
    assert arg == +2.9*4.2

    arg = get_arg(forest.quil_to_program("RX(2.9+4.2) 0"))
    assert arg == 2.9+4.2

    arg = get_arg(forest.quil_to_program("RX(2.9-4.2) 0"))
    assert arg == 2.9-4.2

    arg = get_arg(forest.quil_to_program("RX(pi) 0"))
    assert arg == math.pi

    arg = get_arg(forest.quil_to_program("RX(2.0*pi) 0"))
    assert arg == 2.0*math.pi

    arg = get_arg(forest.quil_to_program("RX(SIN(1.0)) 0"))
    assert arg == math.sin(1.0)

    arg = get_arg(forest.quil_to_program("RX(SIN(1.0)) 0"))
    assert arg == math.sin(1.0)

    arg = get_arg(forest.quil_to_program("RX(EXP(3.3)) 0"))
    assert arg == math.exp(3.3)
    print(arg, type(arg))

    arg = get_arg(forest.quil_to_program("RX(COS(2.3)) 0"))
    print(arg, type(arg))
    assert math.cos(2.3) - arg == ALMOST_ZERO

    arg = get_arg(forest.quil_to_program("RX( SQRT( 42  )   ) 0"))
    assert math.sqrt(42) - arg == ALMOST_ZERO

    arg = get_arg(forest.quil_to_program("RX(CIS(2)) 0"))
    assert cmath.isclose(cis(2), arg)

    arg = get_arg(forest.quil_to_program("RX(2.3 i) 0"))
    assert arg == 2.3j

    arg = get_arg(forest.quil_to_program("RX(EXP(2)) 0"))
    assert math.exp(2) - arg == ALMOST_ZERO

    arg = get_arg(forest.quil_to_program("RX(2^2) 0"))
    assert 4 - arg == ALMOST_ZERO


def test_classical():
    # ro = forest.Register()
    # quil = "TRUE ro[1]"
    # cmd = forest.quil_to_program(quil)[0]
    # assert isinstance(cmd, qf.Move)
    # assert cmd.target == ro[1]
    # assert str(cmd) == 'MOVE ro[1] 1'

    b = forest.Register('b')
    # quil = "FALSE b[2]"
    # cmd = forest.quil_to_program(quil)[0]
    # assert isinstance(cmd, qf.Move)
    # assert cmd.target == b[2]
    # assert str(cmd) == 'MOVE b[2] 0'

    quil = "NOT b[3]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Not)
    assert cmd.target == b[3]
    assert str(cmd) == quil

    quil = "NEG b[3]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Neg)
    assert cmd.target == b[3]
    assert str(cmd) == quil

    quil = "AND b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.And)
    assert str(cmd) == quil

    quil = "IOR b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Ior)
    assert str(cmd) == quil

    quil = "XOR b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Xor)
    assert str(cmd) == quil

    a = forest.Register('a')
    # _test("TRUE b[0]", qf.Move(b[0], 1))
    # _test("FALSE b[0]", qf.Move(b[0], 0))
    _test("NOT b[0]", forest.Not(b[0]))
    _test("AND b[0] a[1]", forest.And(b[0], a[1]))
    _test("XOR b[0] b[1]", forest.Xor(b[0], b[1]))


def test_classical_moves():
    quil = "MOVE b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Move)
    assert str(cmd) == quil

    quil = "EXCHANGE b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Exchange)
    assert str(cmd) == quil

    # quil = "CONVERT b[0] b[1]"                # FIXME
    # cmd = forest.quil_to_program(quil)[0]
    # assert isinstance(cmd, qf.Convert)
    # assert str(cmd) == quil

    quil = "LOAD b[0] this b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Load)
    assert str(cmd) == quil
    # assert len(cmd.cbits) == 2

    quil = "STORE that b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Store)
    assert str(cmd) == quil
    # assert len(cmd.cbits) == 2

    # quil = "STORE that b[0] 200"               # FIXME
    # cmd = forest.quil_to_program(quil)[0]
    # assert isinstance(cmd, forest.Store)
    # assert str(cmd) == quil
    # # assert len(cmd.cbits) == 1

    b = forest.Register('b')
    _test("MOVE b[0] b[1]", forest.Move(b[0], b[1]))
    _test("EXCHANGE b[0] b[1]", forest.Exchange(b[0], b[1]))


def test_classical_math():
    quil = "ADD b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Add)
    assert str(cmd) == quil

    quil = "MUL b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Mul)
    assert str(cmd) == quil

    quil = "SUB b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Sub)
    assert str(cmd) == quil

    quil = "DIV b[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Div)
    assert str(cmd) == quil

    quil = "ADD b[0] 4"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Add)
    assert str(cmd) == quil

    quil = "MUL b[0] 2"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Mul)
    assert str(cmd) == quil

    quil = "SUB b[0] 3"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Sub)
    assert str(cmd) == quil

    quil = "DIV b[0] 2"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Div)
    assert str(cmd) == quil


def test_comparisons():
    quil = "EQ a[1] c[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.EQ)
    assert str(cmd) == quil

    quil = "GT a[1] c[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.GT)
    assert str(cmd) == quil

    quil = "GE a[1] c[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.GE)
    assert str(cmd) == quil

    quil = "LT a[1] c[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.LT)
    assert str(cmd) == quil

    quil = "LE a[1] c[0] b[1]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.LE)
    assert str(cmd) == quil


def test_delare():
    quil = "DECLARE ro BITS"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Declare)
    assert str(cmd) == quil

    quil = "DECLARE ro BITS [10]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Declare)
    assert str(cmd) == quil

    quil = "DECLARE ro BITS [10] SHARING rs"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Declare)
    assert str(cmd) == quil

    # FIXME
    # quil = "DECLARE ro BITS [10] SHARING rs OFFSET 2 this"
    # cmd = forest.quil_to_program(quil)[0]
    # assert isinstance(cmd, qf.Declare)
    # assert str(cmd) == quil

    # FIXME
    # quil = "DECLARE ro BITS [10] SHARING rs OFFSET 16 REAL OFFSET 32 OCTET"
    # cmd = forest.quil_to_program(quil)[0]
    # assert isinstance(cmd, qf.Declare)
    # assert str(cmd) == quil


def test_label():
    quil = "LABEL @some_target"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Label)
    assert cmd.target == "some_target"
    assert str(cmd) == quil


def test_jump():
    quil = "JUMP @some_target"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Jump)
    assert cmd.target == "some_target"
    assert str(cmd) == quil

    quil = "JUMP-UNLESS @some_target ro[2]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.JumpUnless)
    # assert cmd.cbits == [('ro', 2)]
    assert str(cmd) == quil

    quil = "JUMP-WHEN @some_target b[3]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.JumpWhen)
    # assert cmd.cbits == [('b', 3)]
    assert str(cmd) == quil


def test_jumps2():
    _test("LABEL @test_1", forest.Label("test_1"))
    _test("JUMP @test_1", forest.Jump("test_1"))
    cb = forest.Register('cb')
    _test("JUMP-WHEN @test_1 cb[0]", forest.JumpWhen("test_1", cb[0]))
    _test("JUMP-UNLESS @test_1 cb[1]", forest.JumpUnless("test_1", cb[1]))


def test_measure():
    quil = "MEASURE 1"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Measure)
    assert str(cmd) == quil

    quil = "MEASURE 3 reg0[2]"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Measure)
    assert str(cmd) == quil

    prog = forest.quil_to_program(quil)
    print(len(prog))
    print(prog)

    _test("MEASURE 0", forest.Measure(0))
    _test("MEASURE 0 b[1]", forest.Measure(0, forest.Register('b')[1]))

# Not yet supported in pyquil
# def test_measure_ident():
#    _test("MEASURE a", forest.Measure('a'))
#    _test("MEASURE a b", forest.Measure('a', 'b'))


def test_pragma():
    quil = "PRAGMA somename"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Pragma)
    assert str(cmd) == quil

    quil = "PRAGMA somename arg0"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Pragma)
    assert str(cmd) == quil

    quil = "PRAGMA somename arg0 arg1"
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Pragma)
    assert len(cmd.args) == 2
    assert str(cmd) == quil

    quil = 'PRAGMA somename "some string"'
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Pragma)
    assert not cmd.args
    assert cmd.freeform == "some string"
    assert str(cmd) == quil

    quil = 'PRAGMA somename arg2 arg3 arg7 "some string"'
    cmd = forest.quil_to_program(quil)[0]
    assert isinstance(cmd, forest.Pragma)
    assert len(cmd.args) == 3
    assert cmd.freeform == "some string"
    assert str(cmd) == quil

    _test('PRAGMA gate_time H "10 ns"',
          forest.Pragma('gate_time', ['H'], '10 ns'))
    _test('PRAGMA qubit 0', forest.Pragma('qubit', [0]))
    _test('PRAGMA NO-NOISE', forest.Pragma('NO-NOISE'))


def test_gate_1qubit():
    gates = 'I', 'H', 'X', 'Y', 'Z', 'S', 'T'

    for g in gates:
        quil = f'{g} 42'
        cmd = forest.quil_to_program(quil)[0]
        assert cmd.name == g
        assert cmd.qubits == (42,)
        assert not cmd.params


def test_gate_2qubit():
    gates = 'SWAP', 'CZ', 'CNOT', 'ISWAP'

    for g in gates:
        quil = f'{g} 2 42'
        cmd = forest.quil_to_program(quil)[0]
        assert cmd.name == g
        assert cmd.qubits == (2, 42)
        assert not cmd.params


def test_gate_3qubit():
    gates = 'CCNOT', 'CSWAP'

    for g in gates:
        quil = f'{g} 2 42 5'
        cmd = forest.quil_to_program(quil)[0]
        assert cmd.name == g
        assert cmd.qubits == (2, 42, 5)
        assert not cmd.params


def test_gate_1qubit_param():
    gates = 'RX', 'RY', 'RZ'
    for g in gates:
        quil = f'{g}(2.8) 42'
        cmd = forest.quil_to_program(quil)[0]
        assert cmd.name == g
        assert cmd.qubits == (42,)
        assert cmd.params == {'theta': 2.8}


def test_gate_2qubit_param():
    gates = 'CPHASE00', 'CPHASE01', 'CPHASE10', 'CPHASE', 'PSWAP'

    for g in gates:
        quil = f'{g}(0.5) 2 42'
        cmd = forest.quil_to_program(quil)[0]
        assert cmd.name == g
        assert cmd.qubits == (2, 42)
        assert cmd.params == {'theta': 0.5}


def test_parameters():
    _test("RX(123) 0", forest.Call('RX', [123], qubits=[0]))
    _test("CPHASE00(0) 0 1", forest.Call('CPHASE00', [0], [0, 1]))
    _test("A(8,9) 0", forest.Call("A", [8, 9], [0]))
    _test("A(8, 9) 0", forest.Call("A", [8, 9], [0]))


# FIXME
# def test_variable():
#     quil = 'RX(%theta) 1'
#     prog = forest.quil_to_program(quil)
#     assert prog[0].quil() == quil

#     quil = 'RX(2*%theta) 2'
#     prog = forest.quil_to_program(quil)
#     print(quil)
#     assert prog[0].quil() == quil

#     quil = 'RX(%a*(%b + 2)/(3*%c)) 3'
#     prog = forest.quil_to_program(quil)
#     print(quil)
#     assert prog[0].quil() == quil

#     quil = 'RX(cos(%t)) 4'
#     prog = forest.quil_to_program(quil)
#     print(quil)
#     assert prog[0].quil() == quil


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


# def test_prog():
#     prog = forest.quil_to_program(QUILPROG)
#     assert len(prog) == 11
#     assert str(prog) == QUILPROG


# HADAMARD = """DEFGATE HADAMARD:
#     1/sqrt(2), 1/sqrt(2)
#     1/sqrt(2), -1/sqrt(2)
# """


# I2 = """DEFGATE I2:
#     1, 0, 0, 0
#     0, 1, 0, 0
#     0, 0, 1, 0
#     0, 0, 0, 1
# """


# def test_defgate():
#     prog = forest.quil_to_program(HADAMARD)
#     cmd = prog[0]
#     assert cmd.name == 'HADAMARD'
#     assert cmd.name == 'DEFGATE'
#     # assert cmd.matrix[0][0] == 1/math.sqrt(2) #FIXME
#     print(cmd)

#     prog = forest.quil_to_program(I2)
#     cmd = prog[0]
#     assert cmd.name == 'I2'
#     assert cmd.name == 'DEFGATE'
#     assert cmd.matrix[2][2] == 1
#     assert str(cmd) == I2


# PHASEGATE = """DEFGATE PHASEGATE(%theta):
#     1, 0
#     0, cis(%theta)
# """


# def test_defgate_param():
#     prog = forest.quil_to_program(PHASEGATE)
#     cmd = prog[0]
#     print(cmd)
#     assert str(cmd) == PHASEGATE


# def test_defgate2():
#     sqrt_x = qf.DefGate("SQRT-X", np.array([[0.5 + 0.5j, 0.5 - 0.5j],
#                                             [0.5 - 0.5j, 0.5 + 0.5j]]))
#     defgates = """DEFGATE SQRT-X:
#     0.5+0.5i, 0.5-0.5i
#     0.5-0.5i, 0.5+0.5i
# """

#     _test(defgates, sqrt_x)

# FIXME
# hadamard = qf.DefGate("HADAMARD",
#                        np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
#                                  [1 / np.sqrt(2), -1 / np.sqrt(2)]]))

# DEFGATE HADAMARD:
#     sqrt(2)/2, sqrt(2)/2
#     sqrt(2)/2, -sqrt(2)/2
#     """.strip()


# def test_defgate_with_variables():
#     # Note that technically the RX gate includes -i instead of just i but
#     # this messes a bit with the test since
#     # it's not smart enough to figure out that -1*i == -i
#     theta = qf.quil_parameter('theta')

#     rx = [[sympy.cos(theta / 2), 1j * sympy.sin(theta / 2)],
#           [1j * sympy.sin(theta / 2), sympy.cos(theta / 2)]]

#     defgate = 'DEFGATE RX(%theta):\n' \
#               '    cos(%theta/2), i*sin(%theta/2)\n' \
#               '    i*sin(%theta/2), cos(%theta/2)\n'

#     _test(defgate, qf.DefGate('RX', rx, [theta]))


# OTHERGATE = """DEFGATE OTHER(%theta):
#     sqrt(%theta), 0
#     0, exp(%theta)
# """


# def test_defgate_other():
#     # Catch Exp, sqrt to string
#     prog = forest.quil_to_program(OTHERGATE)
#     cmd = prog[0]
#     # print(cmd)
#     assert str(cmd) == OTHERGATE


# FOOBAR = """
# DEFCIRCUIT FOO:
#     LABEL @FOO_A
#     JUMP @GLOBAL # valid, global label
#     JUMP @FOO_A # valid, local to FOO
# #    JUMP @BAR_A # invalid

# DEFCIRCUIT BAR:
#     LABEL @BAR_A
#     JUMP @FOO_A # invalid
# #LABEL @GLOBAL
# ##FOO       # FIXME: DOES NOT PARSE
# ##BAR
# #JUMP @FOO_A # invalid
# #JUMP @BAR_A # invalid
# """

# BELL = """DEFCIRCUIT BELL Qm Qn:
#     H Qm
#     CNOT Qm Qn
# """

# Not yes supported in pyquil
# def test_circuit():
#    prog = forest.quil_to_program(BELL)
#    assert str(prog[0]) == BELL
#
#    prog = forest.quil_to_program(FOOBAR)
#    print(">><<")
#    print(FOOBAR)
#    print(">><<")
#    print(prog)
#    print(">><<")

# TEST INSTUCTIONS

QUILPROG2 = """RY(pi/2) 0
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


def test_instr_qubits():
    prog = forest.quil_to_program(QUILPROG)
    assert prog.qubits == [0, 1]

    # prog = forest.quil_to_program("RX(3) 1 2 7 3")
    # assert prog.qubits == [1, 2, 3, 7]

# FIXME
# def test_extra_spaces():
# forest.quil_to_program("TRUE     [0]") # FIXME: FAILS
# forest.quil_to_program("TRUE     [0]\n   # Tabbed comment")# FIXME: FAILS

# # Not yes supported in pyquil
# forest.quil_to_program("DEFCIRCUIT FOO:\n    TRUE [0]")
# forest.quil_to_program("DEFCIRCUIT FOO:\n    #Comment \n    TRUE [0]\n")
# # Blank line
# forest.quil_to_program("DEFCIRCUIT FOO:\n             \n    TRUE [0]\n")


def test_comments():
    forest.quil_to_program('#Comment')
    forest.quil_to_program('#Comment\n#more comments')
    forest.quil_to_program('#Comment\n  #more comments')
# forest.quil_to_program('#Comment\n    #more comments')#FIXME: Fails
# forest.quil_to_program('TRUE     [0]      #more comments') #FIXME: Fails
# forest.quil_to_program("TRUE     [0]\n    # Tabbed comment") #FIXME: Fails

# # DEFCIRCUITS not yet supported in pyquil
# forest.quil_to_program("DEFCIRCUIT FOO:\n    #Comment \n    TRUE [0]\n")

# Not yes supported in pyquil
# def test_call():
#    forest.quil_to_program('TEST 1 [2]')
