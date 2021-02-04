# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.channels
"""
# Kudos: Tests adapted from density branch of reference-qvm by Nick Rubin

from functools import reduce
from operator import add

import numpy as np
import pytest

import quantumflow as qf


def test_transpose_map() -> None:
    # The transpose map is a superoperator that transposes a 1-qubit
    # density matrix. Not physical.
    # quant-ph/0202124

    q0 = 0
    ops = [
        qf.Unitary(np.asarray([[1, 0], [0, 0]]), [q0]),
        qf.Unitary(np.asarray([[0, 0], [0, 1]]), [q0]),
        qf.Unitary(np.asarray([[0, 1], [1, 0]]) / np.sqrt(2), [q0]),
        qf.Unitary(np.asarray([[0, 1], [-1, 0]]) / np.sqrt(2), [q0]),
    ]

    kraus = qf.Kraus(ops, weights=(1, 1, 1, -1))
    rho0 = qf.random_density(1)
    rho1 = kraus.evolve(rho0)

    op0 = rho0.asoperator()
    op1 = rho1.asoperator()
    assert np.allclose(op0.T, op1)

    # The Choi matrix should be same as Swap operator
    choi = kraus.aschannel().choi()
    assert np.allclose(choi, qf.Swap(0, 2).asoperator())


def test_random_density() -> None:
    rho = qf.random_density(4)
    assert rho.tensor.shape == (2,) * 8


def test_density() -> None:
    ket = qf.random_state(3)
    matrix = qf.tensors.outer(ket.tensor, np.conj(ket.tensor), rank=1)
    qf.Density(matrix)
    qf.Density(matrix, [0, 1, 2])

    with pytest.raises(ValueError):
        qf.Density(matrix, [0, 1, 2, 3])


def test_state_to_density() -> None:
    density = qf.ghz_state(4).asdensity()
    assert list(density.tensor.shape) == [2] * 8

    prob = density.probabilities()
    assert np.isclose(prob[0, 0, 0, 0] - 0.5, 0.0)
    assert np.isclose(prob[0, 1, 0, 0], 0.0)
    assert np.isclose(prob[1, 1, 1, 1] - 0.5, 0.0)

    ket = qf.random_state(3)
    density = ket.asdensity()
    ket_prob = ket.probabilities()
    density_prob = density.probabilities()

    for index, prob in np.ndenumerate(ket_prob):
        assert np.isclose(prob - density_prob[index], 0.0)


def test_purity() -> None:
    density = qf.ghz_state(4).asdensity()
    assert np.isclose(qf.purity(density), 1.0)

    for _ in range(10):
        rho = qf.random_density(4)
        purity = np.real(qf.purity(rho))
        assert purity < 1.0
        assert purity >= 0.0


def test_stdkraus_creation() -> None:
    qf.Damping(0.1, 0)
    qf.Depolarizing(0.1, 0)
    qf.Dephasing(0.1, 0)


def test_stdchannels_creation() -> None:
    qf.Damping(0.1, 0).aschannel()
    qf.Depolarizing(0.1, 0).aschannel()
    qf.Dephasing(0.1, 0).aschannel()


def test_identity() -> None:
    chan = qf.IdentityGate([1]).aschannel()
    rho = qf.random_density(2)
    after = chan.evolve(rho)
    assert qf.densities_close(rho, after)

    assert chan.name == "Channel"


def test_channel_chi() -> None:
    chan = qf.IdentityGate([0, 1, 2]).aschannel()
    chi = chan.chi()
    assert chi.shape == (64, 64)


def test_channle_choi() -> None:
    chan0 = qf.Damping(0.1, 0).aschannel()
    choi = chan0.choi()
    chan1 = qf.Channel.from_choi(choi, [0])
    assert qf.channels_close(chan0, chan1)


def test_sample_coin() -> None:
    chan = qf.H(0).aschannel()
    rho = qf.zero_state(1).asdensity()
    rho = chan.evolve(rho)
    prob = rho.probabilities()
    assert np.allclose(prob, [[0.5, 0.5]])


def test_sample_bell() -> None:
    rho = qf.zero_state(2).asdensity()
    chan = qf.H(0).aschannel()
    rho = chan.evolve(rho)
    chan = qf.CNot(0, 1).aschannel()
    rho = chan.evolve(rho)
    prob = rho.probabilities()

    assert np.allclose(prob, [[0.5, 0], [0, 0.5]])


def test_biased_coin() -> None:
    # sample from a 75% head and 25% tails coin
    rho = qf.zero_state(1).asdensity()
    chan = qf.Rx(np.pi / 3, 0).aschannel()
    rho = chan.evolve(rho)
    prob = rho.probabilities()
    assert np.allclose(prob, [0.75, 0.25])


def test_measurement() -> None:
    rho = qf.zero_state(2).asdensity()
    chan = qf.H(0).aschannel()
    rho = chan.evolve(rho)
    rho = qf.Kraus([qf.P0(0), qf.P1(0)]).aschannel().evolve(rho)
    K = qf.Kraus([qf.P0(1), qf.P1(1)])
    _ = K.aschannel()

    rho = qf.Kraus([qf.P0(1), qf.P1(1)]).aschannel().evolve(rho)
    prob = rho.probabilities()
    assert np.allclose(prob, [[0.5, 0], [0.5, 0]])
    assert np.isclose(prob[0, 0] * 2, 1.0)
    assert np.isclose(prob[1, 0] * 2, 1.0)


def test_qaoa() -> None:
    ket_true = [
        0.00167784 + 1.00210180e-05 * 1j,
        0.5 - 4.99997185e-01 * 1j,
        0.5 - 4.99997185e-01 * 1j,
        0.00167784 + 1.00210180e-05 * 1j,
    ]
    rho_true = qf.State(ket_true).asdensity()

    rho = qf.zero_state(2).asdensity()
    rho = qf.Ry(np.pi / 2, 0).aschannel().evolve(rho)
    rho = qf.Rx(np.pi, 0).aschannel().evolve(rho)
    rho = qf.Ry(np.pi / 2, 1).aschannel().evolve(rho)
    rho = qf.Rx(np.pi, 1).aschannel().evolve(rho)
    rho = qf.CNot(0, 1).aschannel().evolve(rho)
    rho = qf.Rx(-np.pi / 2, 1).aschannel().evolve(rho)
    rho = qf.Ry(4.71572463191, 1).aschannel().evolve(rho)
    rho = qf.Rx(np.pi / 2, 1).aschannel().evolve(rho)
    rho = qf.CNot(0, 1).aschannel().evolve(rho)
    rho = qf.Rx(-2 * 2.74973750579, 0).aschannel().evolve(rho)
    rho = qf.Rx(-2 * 2.74973750579, 1).aschannel().evolve(rho)
    assert qf.densities_close(rho, rho_true)


def test_amplitude_damping() -> None:
    rho = qf.zero_state(1).asdensity()
    p = 1.0 - np.exp(-(50) / 15000)
    chan = qf.Damping(p, 0).aschannel()
    rho1 = chan.evolve(rho)
    assert qf.densities_close(rho, rho1)

    rho2 = qf.X(0).aschannel().evolve(rho1)
    rho3 = chan.evolve(rho2)

    expected = qf.Density(
        [[0.00332778 + 0.0j, 0.00000000 + 0.0j], [0.00000000 + 0.0j, 0.99667222 + 0.0j]]
    )
    assert qf.densities_close(expected, rho3)


def test_depolarizing() -> None:
    p = 1.0 - np.exp(-1 / 20)

    chan = qf.Depolarizing(p, 0).aschannel()
    rho0 = qf.Density([[0.5, 0], [0, 0.5]])
    assert qf.densities_close(rho0, chan.evolve(rho0))

    rho0 = qf.random_density(1)
    rho1 = chan.evolve(rho0)
    pr0 = np.real(qf.purity(rho0))
    pr1 = np.real(qf.purity(rho1))
    assert pr0 > pr1

    # Test data extracted from refereneqvm
    rho2 = qf.Density([[0.43328691, 0.48979689], [0.48979689, 0.56671309]])
    rho_test = qf.Density(
        [[0.43762509 + 0.0j, 0.45794666 + 0.0j], [0.45794666 + 0.0j, 0.56237491 + 0.0j]]
    )
    assert qf.densities_close(chan.evolve(rho2), rho_test)

    ket0 = qf.random_state(1)
    qf.Depolarizing(p, 0).run(ket0)

    rho1b = qf.Depolarizing(p, 0).evolve(rho0)
    assert qf.densities_close(rho1, rho1b)


def test_kruas_qubits() -> None:
    rho = qf.Kraus([qf.P0(0), qf.P1(1)])
    assert rho.qubits == (0, 1)
    assert rho.qubit_nb == 2


def test_kraus_evolve() -> None:
    rho = qf.zero_state(1).asdensity()
    p = 1 - np.exp(-50 / 15000)
    kraus = qf.Damping(p, 0)
    rho1 = kraus.evolve(rho)
    assert qf.densities_close(rho, rho1)

    rho2 = qf.X(0).aschannel().evolve(rho1)
    rho3 = kraus.evolve(rho2)

    expected = qf.Density(
        [[0.00332778 + 0.0j, 0.00000000 + 0.0j], [0.00000000 + 0.0j, 0.99667222 + 0.0j]]
    )

    assert qf.densities_close(expected, rho3)


def test_kraus_run() -> None:
    ket0 = qf.zero_state(["a"])
    ket0 = qf.X("a").run(ket0)
    p = 1.0 - np.exp(-2000 / 15000)

    kraus = qf.Damping(p, "a")

    reps = 1000
    results = [kraus.run(ket0).asdensity().asoperator() for _ in range(reps)]
    matrix = reduce(add, results) / reps
    rho_kraus = qf.Density(matrix, ["a"])

    rho0 = ket0.asdensity()
    chan = kraus.aschannel()
    rho_chan = chan.evolve(rho0)

    # If this fails occasionally consider increasing tolerance
    # Can't be very tolerant due to stochastic dynamics

    assert qf.densities_close(rho_chan, rho_kraus, atol=0.05)


def test_channel_adjoint() -> None:
    kraus0 = qf.Damping(0.1, 0)
    chan0 = kraus0.aschannel()
    chan1 = chan0.H.H
    assert qf.channels_close(chan0, chan1)

    chan2 = kraus0.H.aschannel()
    assert qf.channels_close(chan2, chan0.H)

    # 2 qubit hermitian channel
    chan3 = qf.CZ(0, 1).aschannel()
    chan4 = chan3.H
    assert qf.channels_close(chan3, chan4)


def test_kraus_qubits() -> None:
    kraus = qf.Kraus([qf.X(1), qf.Y(0)])
    assert kraus.qubits == (0, 1)

    kraus = qf.Kraus([qf.X("a"), qf.Y("b")])
    assert len(kraus.qubits) == 2
    assert kraus.qubit_nb == 2


def test_chan_qubits() -> None:
    chan = qf.Kraus([qf.X(1), qf.Y(0)]).aschannel()
    assert chan.qubits == (0, 1)
    assert chan.qubit_nb == 2


def test_chan_permute() -> None:
    chan0 = qf.CNot(0, 1).aschannel()
    chan1 = qf.CNot(1, 0).aschannel()

    assert not qf.channels_close(chan0, chan1)

    chan2 = chan1.permute([0, 1])
    assert chan2.qubits == (0, 1)
    assert qf.channels_close(chan1, chan2)

    chan3 = chan1.on(0, 1)
    print(chan0.qubits, chan3.qubits)
    assert qf.channels_close(chan0, chan3)


def test_channel_errors() -> None:
    chan = qf.CNot(0, 1).aschannel()
    with pytest.raises(TypeError):
        chan.run(qf.zero_state(2))

    with pytest.raises(TypeError):
        chan.asgate()

    assert chan.aschannel() is chan

    with pytest.raises(NotImplementedError):
        chan @ 123  # type: ignore

    with pytest.raises(ValueError):
        qf.Channel(chan.tensor, qubits=[0, 1, 2])


def test_kraus_errors() -> None:
    kraus = qf.Kraus([qf.X(1), qf.Y(0)])

    with pytest.raises(TypeError):
        kraus.asgate()


def test_kraus_complete() -> None:
    kraus = qf.Kraus([qf.X(1)])
    assert qf.kraus_iscomplete(kraus)

    kraus = qf.Damping(0.1, 0)
    assert qf.kraus_iscomplete(kraus)

    assert qf.kraus_iscomplete(qf.Damping(0.1, 0))
    assert qf.kraus_iscomplete(qf.Dephasing(0.1, 0))
    assert qf.kraus_iscomplete(qf.Depolarizing(0.1, 0))


def test_askraus() -> None:
    def _roundtrip(kraus: qf.Kraus) -> None:
        assert qf.kraus_iscomplete(kraus)

        chan0 = kraus.aschannel()
        kraus1 = qf.channel_to_kraus(chan0)
        assert qf.kraus_iscomplete(kraus1)

        chan1 = kraus1.aschannel()
        assert qf.channels_close(chan0, chan1)

    p = 1 - np.exp(-50 / 15000)
    _roundtrip(qf.Kraus([qf.X(1)]))
    _roundtrip(qf.Damping(p, 0))
    _roundtrip(qf.Depolarizing(0.9, 1))


def test_channel_trace() -> None:
    chan = qf.I(0).aschannel()
    assert np.isclose(chan.trace(), 4)


def test_channel_join() -> None:
    chan0 = qf.H(0).aschannel()
    chan1 = qf.X(1).aschannel()
    chan01 = qf.join_channels(chan0, chan1)
    assert chan01.qubits == (0, 1)


def test_create_channel() -> None:
    gate = qf.Rx(0.2, 0)
    _ = qf.Rx(0.2, 0).aschannel()

    params = tuple(gate.params)
    qubits = gate.qubits
    tensor = gate.aschannel().tensor
    name = "ChanRx"

    _ = qf.Channel(tensor, qubits, params, name)


def test_random_channel() -> None:
    chan = qf.random_channel([0, 1, 2], rank=4)
    rho0 = qf.random_density([0, 1, 2], rank=2)
    purity0 = qf.purity(rho0)
    rho1 = chan.evolve(rho0)
    purity1 = qf.purity(rho1)

    assert purity1 < purity0

    chan = qf.random_channel([0, 1, 2], rank=4, unital=True)
    assert qf.almost_unital(chan)


# fin
