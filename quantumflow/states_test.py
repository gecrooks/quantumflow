# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow States
"""

import io

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.utils import FrozenDict

from .config_test import REPS


def test_zeros() -> None:
    ket = qf.zero_state(4)
    assert ket.tensor[0, 0, 0, 0] == 1
    assert ket.tensor[0, 0, 1, 0] == 0


def test_w_state() -> None:
    ket = qf.w_state(4)
    assert ket.tensor[0, 0, 0, 0] == 0
    assert ket.tensor[0, 0, 1, 0] * 2.0 == 1


def test_ghz_state() -> None:
    ket = qf.ghz_state(4)
    assert ket.tensor[0, 0, 0, 1] == 0
    assert np.isclose(2.0 * ket.tensor[0, 0, 0, 0] ** 2.0, 1.0)


def test_random_state() -> None:
    state = qf.random_state(4)
    assert state.tensor.shape == (2,) * 4


def test_state_bits() -> None:
    for n in range(1, 6):
        assert qf.zero_state(n).qubit_nb == n


def test_state_labels() -> None:
    # Quil labeling convention
    N = 4
    qubits = range(N - 1, -1, -1)
    ket = qf.zero_state(qubits)
    ket = qf.X(0).run(ket)
    ket = qf.X(1).run(ket)
    assert ket.tensor[0, 0, 1, 1] == 1

    ket = ket.on(0, 1, 3, 4)
    assert ket.tensor[0, 0, 1, 1] == 1

    ket = ket.permute([4, 3, 0, 1])

    ket = ket.permute()
    assert ket.qubits == (0, 1, 3, 4)

    ket = ket.rewire({0: 0, 1: 1, 3: 3, 4: 5})
    assert ket.qubits == (0, 1, 3, 5)


def test_probability() -> None:
    state = qf.w_state(3)
    qf.print_state(state)
    prob = state.probabilities()

    qf.print_probabilities(state)
    assert np.isclose(prob.sum(), 1)


def test_states_close() -> None:
    ket0 = qf.w_state(4)
    ket1 = qf.w_state(3)
    ket2 = qf.w_state(4)
    ket3 = qf.ghz_state(3)

    assert qf.states_close(ket0, ket2)
    assert not qf.states_close(ket1, ket3)
    assert qf.states_close(ket2, ket2)


def test_str() -> None:
    ket = qf.random_state(10)
    s = str(ket)
    assert s[-3:] == "..."


def test_print_state() -> None:
    f = io.StringIO()
    state = qf.w_state(5)
    qf.print_state(state, file=f)


def test_print_probabilities() -> None:
    f = io.StringIO()
    state = qf.w_state(5)
    qf.print_probabilities(state, file=f)


def test_measure() -> None:
    ket = qf.zero_state(2)
    res = ket.measure()
    assert np.allclose(res, [0, 0])

    ket = qf.X(0).run(ket)
    res = ket.measure()
    assert np.allclose(res, [1, 0])

    ket = qf.H(0).run(ket)
    ket = qf.CNot(0, 1).run(ket)
    for _ in range(REPS):
        res = ket.measure()
        assert res[0] == res[1]  # Both qubits measured in same state


def test_sample() -> None:
    ket = qf.zero_state(2)
    ket = qf.H(0).run(ket)
    ket = qf.CNot(0, 1).run(ket)

    samples = ket.sample(10)
    assert samples.sum() == 10


def test_expectation() -> None:
    ket = qf.zero_state(1)
    ket = qf.H(0).run(ket)

    m = ket.expectation([0.4, 0.6])
    assert np.isclose(m, 0.5)

    ket.expectation([0.4, 0.6], 10)


def test_random_density() -> None:
    rho = qf.random_density(4)
    assert list(rho.tensor.shape) == [2] * 8

    rho1 = qf.random_density(4, rank=2, ensemble="Hilbert–Schmidt")
    assert list(rho1.tensor.shape) == [2] * 8

    rho2 = qf.random_density(4, rank=2, ensemble="Bures")
    assert list(rho2.tensor.shape) == [2] * 8

    with pytest.raises(ValueError):
        qf.random_density(4, rank=2, ensemble="not_an_ensemble")


def test_density() -> None:
    ket = qf.random_state(3)
    matrix = np.outer(ket.tensor, np.conj(ket.tensor))
    qf.Density(matrix)
    rho = qf.Density(matrix, [0, 1, 2])

    with pytest.raises(ValueError):
        qf.Density(matrix, [0, 1, 2, 3])

    assert rho.asdensity() is rho

    rho = rho.on(10, 11, 12).permute([12, 11, 10])
    assert rho.qubits == (12, 11, 10)

    rho.permute()


def test_state_to_density() -> None:
    density = qf.ghz_state(4).asdensity()
    assert density.tensor.shape == (2,) * 8

    prob = density.probabilities()
    assert np.isclose(prob[0, 0, 0, 0], 0.5)
    assert np.isclose(prob[0, 1, 0, 0], 0)
    assert np.isclose(prob[1, 1, 1, 1], 0.5)

    ket = qf.random_state(3)
    density = ket.asdensity()
    ket_prob = ket.probabilities()
    density_prob = density.probabilities()

    for index, prob in np.ndenumerate(ket_prob):
        assert np.isclose(prob, density_prob[index])


def test_density_trace() -> None:
    rho = qf.random_density(3)
    assert np.isclose(rho.trace(), 1)

    rho = qf.Density(np.eye(8))
    assert np.isclose(rho.trace(), 8)

    rho = rho.normalize()
    assert np.isclose(rho.trace(), 1)


def test_mixed_density() -> None:
    rho = qf.mixed_density(4)
    assert rho.trace() == 1


def test_join_densities() -> None:
    rho0 = qf.zero_state([0]).asdensity()
    rho1 = qf.zero_state([1]).asdensity()
    rho01 = qf.join_densities(rho0, rho1)
    assert rho01.qubits == (0, 1)


def test_memory() -> None:
    ket0 = qf.zero_state(1)
    assert ket0.memory == {}

    ro = ["ro[0]", "ro[1]"]
    ket1 = ket0.store({ro[1]: 1})
    assert ket0.memory == {}
    assert ket1.memory == {ro[1]: 1}

    ket2 = qf.H(0).run(ket1)
    assert ket2.memory == ket1.memory

    ket3 = ket2.store({ro[1]: 0, ro[0]: 0})
    assert ket3.memory == {ro[0]: 0, ro[1]: 0}

    N = 4
    wf = np.zeros(shape=[2] * N)
    wf[(0,) * N] = 1
    ket = qf.State(wf, list(range(N)), {"a": 2})
    assert ket.memory == {"a": 2}
    assert isinstance(ket.memory, FrozenDict)


def test_density_memory() -> None:
    rho0 = qf.zero_state(1).asdensity()
    assert rho0.memory == {}

    ro = ["ro[0]", "ro[1]"]
    rho1 = rho0.store({ro[1]: 1})
    assert rho0.memory == {}
    assert rho1.memory == {ro[1]: 1}

    rho2 = qf.H(0).aschannel().evolve(rho1)
    assert rho1.memory == rho2.memory

    rho3 = qf.X(0).evolve(rho1)
    assert rho3.memory == rho2.memory


def test_join_states() -> None:
    q0 = qf.zero_state([0])
    q1 = qf.zero_state([1, 2])
    q2 = qf.join_states(q0, q1)
    assert q2.qubit_nb == 3
    assert q2.tensor[0, 0, 0] == 1

    q3 = qf.zero_state([3, 4, 5])
    q3 = qf.X(4).run(q3)

    q4 = qf.join_states(qf.join_states(q0, q1), q3)
    assert q4.qubit_nb == 6
    assert q4.tensor[0, 0, 0, 0, 1, 0] == 1


def test_normalize() -> None:
    ket = qf.random_state(2)
    assert np.isclose(ket.norm(), 1.0)

    ket = qf.Project0(0).run(ket)
    assert ket.norm() < 1.0
    ket = ket.normalize()
    assert np.isclose(ket.norm(), 1.0)


def test_expectation_again() -> None:
    ket = qf.zero_state(4)
    M = np.zeros(shape=([2] * 4))
    M[0, 0, 0, 0] = 42
    M[1, 0, 0, 0] = 1
    M[0, 1, 0, 0] = 2
    M[0, 0, 1, 0] = 3
    M[0, 0, 0, 1] = 4

    avg = ket.expectation(M)
    assert avg == 42

    ket = qf.w_state(4)
    assert ket.expectation(M) == 2.5


def test_error() -> None:
    ket = qf.zero_state(4)
    with pytest.raises(ValueError):
        qf.State(ket.tensor, [0, 1, 2])


@pytest.mark.parametrize("basis", range(4))
@pytest.mark.parametrize("amplitude", [1.0000000000000002, 0.99999998, np.sqrt(1 + 5e-8)])
def test_sample_deterministic_roundoff_preserves_axes_and_state(
    basis, amplitude, monkeypatch
):
    tensor = np.zeros((2, 2), dtype=complex)
    tensor.flat[basis] = -amplitude
    ket = qf.State(tensor, ["left", "right"], {"token": "unchanged"})
    before = ket.tensor.copy()
    probabilities = ket.probabilities().copy()
    rng = np.random.default_rng(42)
    monkeypatch.setattr(np.random, "multinomial", rng.multinomial)

    samples = ket.sample(10**9)

    expected = np.zeros((2, 2), dtype=np.int64)
    expected.flat[basis] = 10**9
    np.testing.assert_array_equal(samples, expected)
    np.testing.assert_array_equal(ket.tensor, before)
    np.testing.assert_array_equal(ket.probabilities(), probabilities)
    assert ket.qubits == ("left", "right")
    assert ket.memory["token"] == "unchanged"


def test_sample_grover_deterministic_overshoot():
    circuit = qf.Circuit(
        [
            qf.H(0),
            qf.H(1),
            qf.CZ(0, 1),
            qf.H(0),
            qf.H(1),
            qf.X(0),
            qf.X(1),
            qf.CZ(0, 1),
            qf.X(0),
            qf.X(1),
            qf.H(0),
            qf.H(1),
        ]
    )
    ket = circuit.run()
    samples = ket.sample(1000)
    np.testing.assert_array_equal(samples, [[0, 0], [0, 1000]])


@pytest.mark.parametrize(
    "tensor",
    [
        [0, 0],
        [2, 0],
        [0.5, 0],
        [np.nan, 0],
        [np.inf, 0],
        [1e154, 1e154],
        [np.sqrt(1 + 2e-7), 0],
        [np.sqrt(1 - 2e-7), 0],
    ],
)
def test_sample_refuses_invalid_state_before_rng(tensor, monkeypatch):
    def unexpected_rng(*args, **kwargs):
        pytest.fail("invalid state reached the random sampler")

    monkeypatch.setattr(np.random, "multinomial", unexpected_rng)
    with pytest.raises(ValueError, match="finite.*normalized"):
        qf.State(tensor).sample(10)


# fin
