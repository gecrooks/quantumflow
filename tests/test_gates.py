
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.gates
"""

# TODO: Refactor to match split of gates in gates subpackage.

import io
import numpy as np
from numpy import pi

import pytest
from sympy import Symbol

import quantumflow as qf
import quantumflow.backend as bk

from . import ALMOST_ONE


from . import REPS


def test_repr():
    g = qf.H()
    assert str(g) == 'H 0'

    g = qf.RX(3.12)
    assert str(g) == 'RX(3.12) 0'

    g = qf.identity_gate(2)
    assert str(g) == 'IDEN 0 1'

    g = qf.random_gate(4)
    assert str(g) == 'RAND4 0 1 2 3'

    g = qf.H(0)
    assert str(g) == 'H 0'

    g = qf.CNOT(0, 1)
    assert str(g) == 'CNOT 0 1'

    g = qf.Gate(qf.CNOT().tensor)
    assert str(g).startswith('<quantumflow.ops.Gate')


def test_repr2():
    g = qf.ZYZ(3.0, 1.0, 1.0, 0)
    assert str(g) == 'ZYZ(3, 1, 1) 0'


def test_identity_gate():
    N = 5
    eye = qf.asarray(qf.identity_gate(N).asoperator())
    assert np.allclose(eye, np.eye(2**N))


def test_bits():

    ket = qf.X(0).run(qf.zero_state(4))
    assert ket.vec.asarray()[1, 0, 0, 0] == ALMOST_ONE
    ket = qf.X(1).run(qf.zero_state(4))
    assert ket.vec.asarray()[0, 1, 0, 0] == ALMOST_ONE
    ket = qf.X(2).run(qf.zero_state(4))
    assert ket.vec.asarray()[0, 0, 1, 0] == ALMOST_ONE
    ket = qf.X(3).run(qf.zero_state(4))
    assert ket.vec.asarray()[0, 0, 0, 1] == ALMOST_ONE

    ket = qf.zero_state(8)
    ket = qf.X(2).run(ket)
    ket = qf.X(4).run(ket)
    ket = qf.X(6).run(ket)

    res = ket.vec.asarray()

    assert res[0, 0, 1, 0, 1, 0, 1, 0]


def test_gate_bits():
    for n in range(1, 6):
        assert qf.identity_gate(n).qubit_nb == n


def test_gates_close():
    assert not qf.gates_close(qf.I(), qf.identity_gate(2))
    assert qf.gates_close(qf.I(), qf.I())
    assert qf.gates_close(qf.P0(), qf.P0())
    assert not qf.gates_close(qf.P0(), qf.P1())

    assert qf.gates_close(qf.CNOT(0, 1), qf.CNOT(0, 1))
    assert not qf.gates_close(qf.CNOT(1, 0), qf.CNOT(0, 1))


def test_normalize():
    ket = qf.random_state(2)
    assert qf.asarray(ket.norm()) == ALMOST_ONE
    ket = qf.P0(0).run(ket)
    assert qf.asarray(ket.norm()) != ALMOST_ONE
    ket = ket.normalize()
    assert qf.asarray(ket.norm()) == ALMOST_ONE


def test_gate_inverse():
    inv = qf.S().H

    eye = qf.S() @ inv
    assert qf.gates_close(eye, qf.I())

    inv = qf.ISWAP().H
    eye = qf.ISWAP() @ inv

    assert qf.gates_close(eye, qf.identity_gate(2))


def test_projectors():
    ket = qf.zero_state(1)
    assert qf.asarray(qf.P0(0).run(ket).norm()) == 1.0

    ket = qf.H(0).run(ket)

    measure0 = qf.P0(0).run(ket)
    assert qf.asarray(measure0.norm()) * 2 == ALMOST_ONE

    measure1 = qf.P1(0).run(ket)
    assert qf.asarray(measure1.norm()) * 2 == ALMOST_ONE


def test_not_unitary():
    assert not qf.almost_unitary(qf.P0())
    assert not qf.almost_unitary(qf.P1())


def test_almost_hermitian():
    assert qf.almost_hermitian(qf.X())
    assert not qf.almost_hermitian(qf.ISWAP())


def test_almost_identity():
    assert not qf.almost_identity(qf.X())
    assert qf.almost_identity(qf.identity_gate(4))


def test_random_gate():
    for n in range(1, 3):
        for _ in range(REPS):
            assert qf.almost_unitary(qf.random_gate(n))


def test_expectation():
    ket = qf.zero_state(4)
    M = np.zeros(shape=([2]*4))
    M[0, 0, 0, 0] = 42
    M[1, 0, 0, 0] = 1
    M[0, 1, 0, 0] = 2
    M[0, 0, 1, 0] = 3
    M[0, 0, 0, 1] = 4
    M = bk.astensor(M)

    avg = ket.expectation(M)
    assert qf.asarray(avg) == 42

    ket = qf.w_state(4)
    assert qf.asarray(ket.expectation(M)) == 2.5


def test_join_gates():
    gate = qf.join_gates(qf.H(0), qf.X(1))
    ket = qf.zero_state(2)
    ket = gate.run(ket)
    ket = qf.H(0).run(ket)
    ket = qf.X(1).run(ket)

    assert qf.states_close(ket, qf.zero_state(2))


# TODO: Move to test_states
def test_join_states():
    q0 = qf.zero_state([0])
    q1 = qf.zero_state([1, 2])
    q2 = qf.join_states(q0, q1)
    assert q2.qubit_nb == 3
    assert q2.vec.asarray()[0, 0, 0] == ALMOST_ONE

    q3 = qf.zero_state([3, 4, 5])
    q3 = qf.X(4).run(q3)

    print(q0)
    print(q1)
    print(q3)

    q4 = qf.join_states(qf.join_states(q0, q1), q3)
    assert q4.qubit_nb == 6
    assert q4.vec.asarray()[0, 0, 0, 0, 1, 0] == ALMOST_ONE


def test_control_gate():
    assert qf.gates_close(qf.control_gate(1, qf.X(0)), qf.CNOT(1, 0))

    assert qf.gates_close(qf.control_gate(0, qf.CNOT(1, 2)),
                          qf.CCNOT(0, 1, 2))


def test_conditional_gate():
    controlled_gate = qf.conditional_gate(0, qf.X(1), qf.Y(1))

    state = qf.zero_state(2)
    state = controlled_gate.run(state)
    assert state.vec.asarray()[0, 1] == ALMOST_ONE

    state = qf.X(0).run(state)
    state = controlled_gate.run(state)
    assert 1.0j*state.vec.asarray()[1, 0] == ALMOST_ONE


def test_print_gate():
    stream = io.StringIO()

    qf.print_gate(qf.CNOT(), file=stream)
    s = stream.getvalue()
    ref = ("00 -> 00 : (1+0j)\n" +
           "01 -> 01 : (1+0j)\n" +
           "10 -> 11 : (1+0j)\n" +
           "11 -> 10 : (1+0j)\n")

    assert s == ref


def test_inverse_random():
    K = 4
    for _ in range(REPS):
        gate = qf.random_gate(K)
        inv = gate.H
        gate = inv @ gate
        assert qf.gates_close(qf.identity_gate(4), gate)


def test_hermitian():
    gate_names = ['I', 'X', 'Y', 'Z', 'H', 'SWAP']
    for name in gate_names:
        print(name)
        gate = qf.NAMED_GATES[name]()
        conj = gate.H

        assert qf.gates_close(gate, conj)


def test_gatemul():
    # three cnots same as one swap
    gate0 = qf.identity_gate([0, 1])

    gate1 = qf.CNOT(1, 0)
    gate2 = qf.CNOT(0, 1)
    gate3 = qf.CNOT(1, 0)

    gate = gate0
    gate = gate1 @ gate
    gate = gate2 @ gate
    gate = gate3 @ gate
    assert qf.gates_close(gate, qf.SWAP())

    # Again, but with labels
    gate0 = qf.identity_gate(['a', 'b'])

    gate1 = qf.CNOT('b', 'a')
    gate2 = qf.CNOT('a', 'b')
    gate3 = qf.CNOT('b', 'a')

    gate = gate0
    gate = gate1 @ gate
    gate = gate2 @ gate
    gate = gate3 @ gate
    assert qf.gates_close(gate, qf.SWAP('a', 'b'))

    gate4 = qf.X('a')
    gate = gate4 @ gate

    with pytest.raises(NotImplementedError):
        gate = gate4 @ 3


def test_gate_permute():
    gate0 = qf.CNOT(0, 1)
    gate1 = qf.CNOT(1, 0)

    assert not qf.gates_close(gate0, gate1)

    gate2 = gate1.permute([0, 1])
    assert gate2.qubits == (0, 1)
    assert qf.gates_close(gate1, gate2)


def test_gates_evolve():
    rho0 = qf.zero_state(3).asdensity()
    qf.H(0).evolve(rho0)


def test_su():
    su = qf.SWAP(0, 1).su()
    assert np.linalg.det(qf.asarray(su.asoperator())) == ALMOST_ONE


# TODO: Move elsewhere
def test_reset():
    reset = qf.Reset(0, 1, 2)

    with pytest.raises(TypeError):
        reset.evolve(qf.random_density([0, 1, 3]))

    with pytest.raises(TypeError):
        reset.asgate()

    with pytest.raises(TypeError):
        reset.aschannel()

    # assert reset.H is reset


def test_interchangeable():
    assert qf.SWAP().interchangeable
    assert not qf.CNOT().interchangeable


def test_symbolic_parameters():
    theta = Symbol('θ')

    gate0 = qf.RZ(theta, 1)
    assert str(gate0) == 'RZ(θ) 1'

    gate1 = gate0 ** 4
    assert str(gate1) == 'RZ(4*θ) 1'

    circ = qf.Circuit([gate0, gate1])
    diag = qf.circuit_to_diagram(circ)
    assert diag == '1: ───Rz(θ)───Rz(4*θ)───'

    gate2 = gate0.resolve({'θ': 2})
    assert gate2.params['theta'] == 2.0


def test_exceptions():

    tensor = qf.CNOT(1, 0).tensor

    with pytest.raises(ValueError):
        qf.Gate(tensor, qubits=[0])

    with pytest.raises(ValueError):
        qf.Gate(tensor, qubits=[0, 1, 2])

    with pytest.raises(ValueError):
        qf.Gate()


def test_relabel():
    gate0 = qf.CNOT(1, 0)
    gate1 = gate0.relabel(['B', 'A'])
    assert gate1.qubits == ('B', 'A')

    gate2 = gate1.relabel({'A': 'a', 'B': 'b', 'C': 'c'})
    assert gate2.qubits == ('b', 'a')

    with pytest.raises(ValueError):
        gate1 = gate0.relabel(['B', 'A', 'C'])


def test_specialize_1q():
    assert isinstance(qf.TZ(0.0).specialize(), qf.I)
    assert isinstance(qf.TZ(0.25).specialize(), qf.T)
    assert isinstance(qf.TZ(0.5).specialize(), qf.S)
    assert isinstance(qf.TZ(1.0).specialize(), qf.Z)
    assert isinstance(qf.TZ(1.5).specialize(), qf.S_H)
    assert isinstance(qf.TZ(1.75).specialize(), qf.T_H)
    assert isinstance(qf.TZ(1.99999999999).specialize(), qf.I)

    special_values = (0.0, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 0.17365,
                      pi, pi/2, pi/4)
    special1p1q = (qf.RX, qf.RY, qf.RZ, qf.TX, qf.TY, qf.TZ, qf.PHASE, qf.TH,
                   qf.W)

    for gatetype in special1p1q:
        for value in special_values:
            gate0 = gatetype(value, 'q0')
            gate1 = gate0.specialize()
            assert gate0.qubits == gate1.qubits
            assert qf.gates_close(gate0, gate1)

    assert isinstance(qf.TW(0.1234, 0.0).specialize(), qf.I)
    assert isinstance(qf.TW(0.0, 1.0).specialize(), qf.X)
    assert isinstance(qf.TW(0.0213, 1.0).specialize(), qf.TW)


def test_specialize_IDEN():
    assert isinstance(qf.IDEN(0).specialize(), qf.I)
    assert isinstance(qf.IDEN(0, 1).specialize(), qf.IDEN)


def test_specialize_2q():
    assert isinstance(qf.CAN(0.0, 0.0, 0.0).specialize(), qf.IDEN)
    assert isinstance(qf.CAN(0.213, 0.0, 0.0).specialize(), qf.XX)
    assert isinstance(qf.CAN(0.0, 0.213, 0.0).specialize(), qf.YY)
    assert isinstance(qf.CAN(0.0, 0.0, 0.213).specialize(), qf.ZZ)
    assert isinstance(qf.CAN(0.213, 0.213, 0.213).specialize(), qf.EXCH)
    assert isinstance(qf.CAN(0.5, 0.32, 0.213).specialize(), qf.CAN)

    assert isinstance(qf.CTX(0.2).specialize(), qf.CTX)
    assert isinstance(qf.CTX(0.0).specialize(), qf.IDEN)
    assert isinstance(qf.CTX(1.0).specialize(), qf.CNOT)

    assert isinstance(qf.XX(0.0).specialize(), qf.IDEN)
    assert isinstance(qf.YY(0.0).specialize(), qf.IDEN)
    assert isinstance(qf.ZZ(0.0).specialize(), qf.IDEN)
    assert isinstance(qf.EXCH(0.0).specialize(), qf.IDEN)


# fin
