# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import io
from typing import Type

import numpy as np
import pytest

import quantumflow as qf

from .config_test import REPS
from .stdgates.stdgates_test import _randomize_gate


def test_bits() -> None:

    ket = qf.X(0).run(qf.zero_state(4))
    assert ket.tensor[1, 0, 0, 0] == 1
    ket = qf.X(1).run(qf.zero_state(4))
    assert ket.tensor[0, 1, 0, 0] == 1
    ket = qf.X(2).run(qf.zero_state(4))
    assert ket.tensor[0, 0, 1, 0] == 1
    ket = qf.X(3).run(qf.zero_state(4))
    assert ket.tensor[0, 0, 0, 1] == 1

    ket = qf.zero_state(8)
    ket = qf.X(2).run(ket)
    ket = qf.X(4).run(ket)
    ket = qf.X(6).run(ket)

    res = ket.tensor
    assert res[0, 0, 1, 0, 1, 0, 1, 0]


def test_CZ() -> None:
    ket = qf.zero_state(2)
    ket = qf.CZ(0, 1).run(ket)
    assert ket.tensor[0, 0] == 1.0

    ket = qf.X(0).run(ket)
    ket = qf.X(1).run(ket)
    ket = qf.CZ(0, 1).run(ket)
    assert -ket.tensor[1, 1] == 1.0


def test_qaoa_circuit() -> None:
    # Kudos: Adapted from reference QVM
    wf_true = np.array(
        [
            0.00167784 + 1.00210180e-05 * 1j,
            0.50000000 - 4.99997185e-01 * 1j,
            0.50000000 - 4.99997185e-01 * 1j,
            0.00167784 + 1.00210180e-05 * 1j,
        ]
    )
    ket_true = qf.State(wf_true.reshape((2, 2)))

    ket = qf.zero_state(2)
    ket = qf.Ry(np.pi / 2, 0).run(ket)
    ket = qf.Rx(np.pi, 0).run(ket)
    ket = qf.Ry(np.pi / 2, 1).run(ket)
    ket = qf.Rx(np.pi, 1).run(ket)
    ket = qf.CNot(0, 1).run(ket)
    ket = qf.Rx(-np.pi / 2, 1).run(ket)
    ket = qf.Ry(4.71572463191, 1).run(ket)
    ket = qf.Rx(np.pi / 2, 1).run(ket)
    ket = qf.CNot(0, 1).run(ket)
    ket = qf.Rx(-2 * 2.74973750579, 0).run(ket)
    ket = qf.Rx(-2 * 2.74973750579, 1).run(ket)

    assert qf.states_close(ket, ket_true)


def test_qubit_qaoa_circuit() -> None:
    # Adapted from reference QVM
    wf_true = np.array(
        [
            0.00167784 + 1.00210180e-05 * 1j,
            0.50000000 - 4.99997185e-01 * 1j,
            0.50000000 - 4.99997185e-01 * 1j,
            0.00167784 + 1.00210180e-05 * 1j,
        ]
    )
    ket_true = qf.State(wf_true.reshape((2, 2)))

    ket = qf.zero_state(2)
    ket = qf.Ry(np.pi / 2, 0).run(ket)
    ket = qf.Rx(np.pi, 0).run(ket)
    ket = qf.Ry(np.pi / 2, 1).run(ket)
    ket = qf.Rx(np.pi, 1).run(ket)
    ket = qf.CNot(0, 1).run(ket)
    ket = qf.Rx(-np.pi / 2, 1).run(ket)
    ket = qf.Ry(4.71572463191, 1).run(ket)
    ket = qf.Rx(np.pi / 2, 1).run(ket)
    ket = qf.CNot(0, 1).run(ket)
    ket = qf.Rx(-2 * 2.74973750579, 0).run(ket)
    ket = qf.Rx(-2 * 2.74973750579, 1).run(ket)

    assert qf.states_close(ket, ket_true)


@pytest.mark.parametrize("gatet", qf.StdGate.cv_stdgates.values())
def test_gate_run(gatet: Type[qf.StdGate]) -> None:
    gate0 = _randomize_gate(gatet)

    gate1 = qf.Unitary(gate0.tensor, gate0.qubits)
    ket = qf.random_state(gate0.qubits)

    ket0 = gate0.run(ket)
    ket1 = gate1.run(ket)
    assert qf.states_close(ket0, ket1)


@pytest.mark.parametrize("gatet", qf.StdGate.cv_stdgates.values())
def test_gate_evolve(gatet: Type[qf.StdGate]) -> None:
    gate0 = _randomize_gate(gatet)

    gate1 = qf.Unitary(gate0.tensor, gate0.qubits)
    rho = qf.random_density(gate0.qubits)

    rho0 = gate0.evolve(rho)
    rho1 = gate1.evolve(rho)
    assert qf.densities_close(rho0, rho1)


# Test PROJECTORS....
def test_projectors() -> None:
    ket = qf.zero_state(1)
    assert qf.P0(0).run(ket).norm() == 1.0

    ket = qf.H(0).run(ket)

    measure0 = qf.P0(0).run(ket)
    assert np.isclose(measure0.norm(), 0.5)

    measure1 = qf.P1(0).run(ket)
    assert np.isclose(measure1.norm(), 0.5)


def test_not_unitary() -> None:
    assert not qf.almost_unitary(qf.P0())
    assert not qf.almost_unitary(qf.P1())


# FIXME, add conditional gate
def test_conditional_gate() -> None:
    controlled_gate = qf.conditional_gate(0, qf.X(1), qf.Y(1))

    state = qf.zero_state(2)
    state = controlled_gate.run(state)
    assert state.tensor[0, 1] == 1.0

    state = qf.X(0).run(state)
    state = controlled_gate.run(state)
    assert 1.0j * state.tensor[1, 0] == 1.0


def test_print_gate() -> None:
    stream = io.StringIO()

    qf.print_gate(qf.CNot(0, 1), file=stream)
    s = stream.getvalue()
    ref = (
        "00 -> 00 : (1+0j)\n"
        + "01 -> 01 : (1+0j)\n"
        + "10 -> 11 : (1+0j)\n"
        + "11 -> 10 : (1+0j)\n"
    )

    assert s == ref


def test_inverse_random() -> None:
    K = 4
    for _ in range(REPS):
        gate = qf.RandomGate(range(K))
        inv = gate.H
        gate1 = inv @ gate
        # TODO: almost_identity
        assert qf.gates_close(qf.IdentityGate([0, 1, 2, 3]), gate1)


# fin
