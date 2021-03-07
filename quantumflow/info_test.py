# Copyright 2020-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import random

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.config import ATOL

from .config_test import REPS


def test_fubini_study_angle() -> None:

    for _ in range(REPS):
        theta = random.uniform(-np.pi, +np.pi)

        ang = qf.fubini_study_angle(qf.I(0).tensor, qf.Rx(theta, 0).su().tensor)
        assert np.isclose(2 * ang / abs(theta), 1.0)

        ang = qf.fubini_study_angle(qf.I(0).tensor, qf.Ry(theta, 0).tensor)
        assert np.isclose(2 * ang / abs(theta), 1.0)

        ang = qf.fubini_study_angle(qf.I(0).tensor, qf.Rz(theta, 0).tensor)
        assert np.isclose(2 * ang / abs(theta), 1.0)

        ang = qf.fubini_study_angle(qf.Swap(0, 1).tensor, qf.PSwap(theta, 0, 1).tensor)

        assert np.isclose(2 * ang / abs(theta), 1.0)

        ang = qf.fubini_study_angle(qf.I(0).tensor, qf.PhaseShift(theta, 0).tensor)
        assert np.isclose(2 * ang / abs(theta), 1.0)

        assert qf.fubini_study_close(qf.Rz(theta, 0).tensor, qf.Rz(theta, 0).tensor)

    for n in range(1, 6):
        eye = qf.IdentityGate(list(range(n)))
        assert np.isclose(qf.fubini_study_angle(eye.tensor, eye.tensor), 0.0)

    with pytest.raises(ValueError):
        qf.fubini_study_angle(qf.RandomGate([1]).tensor, qf.RandomGate([0, 1]).tensor)


def test_fubini_study_angle_states() -> None:
    # The state angle is half angle in Bloch sphere
    angle1 = 0.1324
    ket1 = qf.zero_state(1)
    ket2 = qf.Rx(angle1, 0).run(ket1)
    angle2 = qf.fubini_study_angle(ket1.tensor, ket2.tensor)
    assert np.isclose(angle1, angle2 * 2)


def test_state_angle() -> None:
    ket0 = qf.random_state(1)
    ket1 = qf.random_state(1)
    qf.state_angle(ket0, ket1)

    assert not qf.states_close(ket0, ket1)
    assert qf.states_close(ket0, ket0)


def test_density_angle() -> None:
    rho0 = qf.random_density(1)
    rho1 = qf.random_density(1)
    qf.density_angle(rho0, rho1)

    assert not qf.densities_close(rho0, rho1)
    assert qf.densities_close(rho0, rho0)


def test_gate_angle() -> None:
    gate0 = qf.RandomGate([1])
    gate1 = qf.RandomGate([1])
    qf.gate_angle(gate0, gate1)

    assert not qf.gates_close(gate0, gate1)
    assert qf.gates_close(gate0, gate0)


def test_gates_commute() -> None:
    assert qf.gates_commute(qf.X(0), qf.X(0))
    assert not qf.gates_commute(qf.X(0), qf.T(0))
    assert qf.gates_commute(qf.X(0), qf.T(1))
    assert qf.gates_commute(qf.S(0), qf.T(0))
    assert qf.gates_commute(qf.S(0), qf.T(0))
    assert qf.gates_commute(qf.XX(0.1, 0, 1), qf.X(0))
    assert not qf.gates_commute(qf.ZZ(0.1, 0, 1), qf.X(0))


def test_channel_angle() -> None:
    chan0 = qf.X(0).aschannel()
    chan1 = qf.Y(0).aschannel()
    qf.channel_angle(chan0, chan1)

    assert not qf.channels_close(chan0, chan1)
    assert qf.channels_close(chan0, chan0)


def test_fidelity() -> None:
    rho0 = qf.random_density(4)
    rho1 = qf.random_density(4)

    fid = qf.fidelity(rho0, rho1)
    assert 0.0 <= fid <= 1.0

    rho2 = qf.random_density([3, 2, 1, 0])
    fid = qf.fidelity(rho0, rho2)
    assert 0.0 <= fid <= 1.0

    fid = qf.fidelity(rho0, rho0)
    assert np.isclose(fid, 1.0)

    ket0 = qf.random_state(3)
    ket1 = qf.random_state(3)
    fid0 = qf.state_fidelity(ket0, ket1)

    rho0 = ket0.asdensity()
    rho1 = ket1.asdensity()
    fid1 = qf.fidelity(rho0, rho1)

    assert np.isclose(fid1, fid0)

    fid2 = np.cos(qf.fubini_study_angle(ket0.tensor, ket1.tensor)) ** 2
    assert np.isclose(fid2, fid0)


def test_purity() -> None:
    density = qf.ghz_state(4).asdensity()
    assert np.isclose(qf.purity(density), 1.0)

    for _ in range(10):
        density = qf.random_density(4)
        purity = np.real(qf.purity(density))
        assert purity < 1.0
        assert purity >= 0.0

    rho = qf.Density(np.diag([0.9, 0.1]))
    assert np.isclose(qf.purity(rho), 0.82)  # Kudos: Josh Combs


@pytest.mark.parametrize("repeat", range(10))
def test_bures_distance(repeat: int) -> None:
    rho = qf.random_density(4)
    # Note ATOL. Sometimes does not give accurate answer
    assert np.isclose(qf.bures_distance(rho, rho), 0.0, atol=ATOL * 100)

    rho1 = qf.random_density(4)
    qf.bures_distance(rho, rho1)

    # TODO: Check distance of known special case


def test_bures_angle() -> None:
    rho = qf.random_density(4)
    assert np.isclose(qf.bures_angle(rho, rho), 0.0, atol=ATOL * 10)

    rho1 = qf.random_density(4)
    qf.bures_angle(rho, rho1)

    ket0 = qf.random_state(4)
    ket1 = qf.random_state(4)
    rho0 = ket0.asdensity()
    rho1 = ket1.asdensity()

    ang0 = qf.fubini_study_angle(ket0.tensor, ket1.tensor)
    ang1 = qf.bures_angle(rho0, rho1)

    assert np.isclose(ang0, ang1)


def test_entropy() -> None:
    N = 4
    rho0 = qf.mixed_density(N)
    ent = qf.entropy(rho0, base=2)
    assert np.isclose(ent, N)

    # Entropy invariant to unitary evolution
    chan = qf.RandomGate(range(N)).aschannel()
    rho1 = chan.evolve(rho0)
    ent = qf.entropy(rho1, base=2)
    assert np.isclose(ent, N)


def test_mutual_info() -> None:
    rho0 = qf.mixed_density(4)
    info0 = qf.mutual_info(rho0, qubits0=[0, 1], qubits1=[2, 3])

    # Information invariant to local unitary evolution
    chan = qf.RandomGate(range(2)).aschannel()
    rho1 = chan.evolve(rho0)
    info1 = qf.mutual_info(rho1, qubits0=[0, 1], qubits1=[2, 3])

    assert np.isclose(info0, info1)

    info2 = qf.mutual_info(rho1, qubits0=[0, 1])
    assert np.isclose(info0, info2)


def test_trace_distance() -> None:
    rho = qf.random_density(4)
    assert np.isclose(qf.trace_distance(rho, rho), 0.0)

    rho1 = qf.random_density(4)
    qf.trace_distance(rho, rho1)

    # TODO: Check distance of known special case


def test_average_gate_fidelity() -> None:
    kraus = qf.Damping(0.9, q0=0)
    qf.average_gate_fidelity(kraus)

    qf.average_gate_fidelity(kraus, qf.X(0))
    # TODO: Test actually get correct answer!


def test_circuits_close() -> None:
    circ0 = qf.Circuit([qf.H(0)])
    circ1 = qf.Circuit([qf.H(2)])
    assert not qf.circuits_close(circ0, circ1)

    circ2 = qf.Circuit([qf.X(0)])
    assert not qf.circuits_close(circ0, circ2)

    circ3 = qf.Circuit([qf.H(0)])
    assert qf.circuits_close(circ0, circ3)


def test_gates_phase_close() -> None:
    gate0 = qf.ZPow(0.5, q0=0)
    gate1 = qf.Circuit(qf.translate_tz_to_rz(gate0)).asgate()  # type: ignore
    assert qf.gates_close(gate0, gate1)
    assert not qf.gates_phase_close(gate0, gate1)
    assert qf.gates_phase_close(gate0, gate0)

    gate2 = qf.XPow(0.5, q0=0)
    assert not qf.gates_phase_close(gate0, gate2)


# fin
