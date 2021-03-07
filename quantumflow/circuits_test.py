# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Unit tests for quantumflow.circuits"""


import networkx as nx
import numpy as np
import pytest

import quantumflow as qf
from quantumflow.utils import bitlist_to_int, int_to_bitlist


def true_ket() -> qf.State:
    # Adapted from referenceQVM
    wf_true = np.array(
        [
            0.00167784 + 1.00210180e-05 * 1j,
            0.50000000 - 4.99997185e-01 * 1j,
            0.50000000 - 4.99997185e-01 * 1j,
            0.00167784 + 1.00210180e-05 * 1j,
        ]
    )
    return qf.State(wf_true.reshape((2, 2)))


def test_str() -> None:
    circ = qf.zyz_circuit(0.1, 2.2, 0.5, 0)
    str(circ)
    # TODO Expand


def test_name() -> None:
    assert qf.Circuit().name == "Circuit"


def test_qaoa_circuit() -> None:
    circ = qf.Circuit()
    circ += qf.Ry(np.pi / 2, 0)
    circ += qf.Rx(np.pi, 0)
    circ += qf.Ry(np.pi / 2, 1)
    circ += qf.Rx(np.pi, 1)
    circ += qf.CNot(0, 1)
    circ += qf.Rx(-np.pi / 2, 1)
    circ += qf.Ry(4.71572463191, 1)
    circ += qf.Rx(np.pi / 2, 1)
    circ += qf.CNot(0, 1)
    circ += qf.Rx(-2 * 2.74973750579, 0)
    circ += qf.Rx(-2 * 2.74973750579, 1)

    ket = qf.zero_state(2)
    ket = circ.run(ket)

    assert qf.states_close(ket, true_ket())


def test_qaoa_circuit_turns() -> None:
    circ = qf.Circuit()
    circ += qf.YPow(1 / 2, 0)
    circ += qf.XPow(1, 0)
    circ += qf.YPow(1 / 2, 1)
    circ += qf.XPow(1, 1)
    circ += qf.CNot(0, 1)
    circ += qf.XPow(-1 / 2, 1)
    circ += qf.YPow(4.71572463191 / np.pi, 1)
    circ += qf.XPow(1 / 2, 1)
    circ += qf.CNot(0, 1)
    circ += qf.XPow(-2 * 2.74973750579 / np.pi, 0)
    circ += qf.XPow(-2 * 2.74973750579 / np.pi, 1)

    ket = qf.zero_state(2)
    ket = circ.run(ket)

    assert qf.states_close(ket, true_ket())


def test_circuit_wires() -> None:
    circ = qf.Circuit()
    circ += qf.YPow(1 / 2, 0)
    circ += qf.XPow(1, 10)
    circ += qf.YPow(1 / 2, 1)
    circ += qf.XPow(1, 1)
    circ += qf.CNot(0, 4)

    bits = circ.qubits
    assert bits == (0, 1, 4, 10)


def test_inverse() -> None:
    # Random circuit
    circ = qf.Circuit()
    circ += qf.YPow(1 / 2, 0)
    circ += qf.H(0)
    circ += qf.YPow(1 / 2, 1)
    circ += qf.XPow(1.23123, 1)
    circ += qf.CNot(0, 1)
    circ += qf.XPow(-1 / 2, 1)
    circ += qf.YPow(4.71572463191 / np.pi, 1)
    circ += qf.CNot(0, 1)
    circ += qf.XPow(-2 * 2.74973750579 / np.pi, 0)
    circ += qf.XPow(-2 * 2.74973750579 / np.pi, 1)

    circ_inv = circ.H

    ket = circ.run()
    # qf.print_state(ket)

    ket = circ_inv.run(ket)
    # qf.print_state(ket)

    # print(ket.qubits)
    # print(true_ket().qubits)
    assert qf.states_close(ket, qf.zero_state(2))

    ket = qf.zero_state(2)
    circ += circ_inv
    ket = circ.run(ket)
    assert qf.states_close(ket, qf.zero_state(2))


def test_implicit_state() -> None:
    circ = qf.Circuit()
    circ += qf.YPow(1 / 2, 0)
    circ += qf.H(0)
    circ += qf.YPow(1 / 2, 1)

    ket = circ.run()  # Implicit state
    assert len(ket.qubits) == 2

    with pytest.raises(TypeError):
        # Should fail because qubits aren't sortable, so no standard ordering
        circ += qf.YPow(1 / 2, "namedqubit")
        circ.qubits


def test_elements() -> None:
    circ = qf.Circuit()
    circ1 = qf.Circuit()
    circ2 = qf.Circuit()
    circ1 += qf.Ry(np.pi / 2, 0)
    circ1 += qf.Rx(np.pi, 0)
    circ1 += qf.Ry(np.pi / 2, 1)
    circ1 += qf.Rx(np.pi, 1)
    circ1 += qf.CNot(0, 1)
    circ2 += qf.Rx(-np.pi / 2, 1)
    circ2 += qf.Ry(4.71572463191, 1)
    circ2 += qf.Rx(np.pi / 2, 1)
    circ2 += qf.CNot(0, 1)
    circ2 += qf.Rx(-2 * 2.74973750579, 0)
    circ2 += qf.Rx(-2 * 2.74973750579, 1)
    circ += circ1
    circ += circ2

    assert len(circ) == 11
    assert circ.size() == 11
    assert circ[4].name == "CNot"

    circ_13 = circ[1:3]
    assert len(circ_13) == 2
    assert isinstance(circ_13, qf.Circuit)


def test_create() -> None:
    gen = [qf.H(i) for i in range(8)]

    circ1 = qf.Circuit(list(gen))
    circ1.run(qf.zero_state(8))

    circ2 = qf.Circuit(gen)
    circ2.run(qf.zero_state(8))

    circ3 = qf.Circuit(qf.H(i) for i in range(8))
    circ3.run(qf.zero_state(8))


def test_add() -> None:
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    circ += qf.H(1)

    assert len(list(circ)) == 3

    circ = circ + circ
    assert len(circ) == 6

    circ += circ
    assert len(circ) == 12


def test_ccnot_circuit_evolve() -> None:
    rho0 = qf.random_state(3).asdensity()
    gate = qf.CCNot(0, 1, 2)
    circ = qf.Circuit(qf.translate_ccnot_to_cnot(gate))
    rho1 = gate.evolve(rho0)
    rho2 = circ.evolve(rho0)
    assert qf.densities_close(rho1, rho2)


def test_circuit_aschannel() -> None:
    rho0 = qf.random_state(3).asdensity()
    rho1 = qf.CCNot(0, 1, 2).evolve(rho0)
    gate = qf.CCNot(0, 1, 2)
    circ = qf.Circuit(qf.translate_ccnot_to_cnot(gate))
    chan = circ.aschannel()
    rho2 = chan.evolve(rho0)

    assert qf.densities_close(rho1, rho2)


def test_control_circuit() -> None:
    ccnot = qf.control_circuit([0, 1], qf.X(2))
    ket0 = qf.random_state(3)
    ket1 = qf.CCNot(0, 1, 2).run(ket0)
    ket2 = ccnot.run(ket0)
    assert qf.states_close(ket1, ket2)


def test_phase_estimation_circuit_1() -> None:
    N = 8
    phase = 1 / 4
    gate = qf.Rz(-4 * np.pi * phase, N)
    circ = qf.phase_estimation_circuit(gate, range(N))
    res = circ.run().measure()[0:N]
    est_phase = bitlist_to_int(res) / 2 ** N
    assert np.isclose(phase, est_phase)


def test_phase_estimation_circuit_2() -> None:
    N = 8
    phase = 12 / 256
    gate = qf.Rz(-4 * np.pi * phase, N)
    circ = qf.phase_estimation_circuit(gate, range(N))
    res = circ.run().measure()[0:N]
    est_phase = bitlist_to_int(res) / 2 ** N
    assert np.isclose(phase, est_phase)


def test_phase_estimation_circuit_3() -> None:
    N = 8
    phase = 12 / 256
    gate = qf.ZZ(-4 * phase, N, N + 1)
    circ = qf.phase_estimation_circuit(gate, range(N))
    res = circ.run().measure()[0:N]
    est_phase = bitlist_to_int(res) / 2 ** N
    assert np.isclose(phase, est_phase)

    with pytest.raises(ValueError):
        # Gate and output qubits overlap
        _ = qf.phase_estimation_circuit(gate, range(N + 1))


def test_addition_circuit() -> None:
    # Two bit addition circuit
    circ = qf.addition_circuit([0, 1], [2, 3], [4, 5])

    for c0 in range(0, 2):
        for a0 in range(0, 4):
            for a1 in range(0, 4):
                expected = a0 + a1 + c0
                b0 = int_to_bitlist(a0, 2)
                b1 = int_to_bitlist(a1, 2)
                bc = [c0]
                bits = tuple(b0 + b1 + bc + [0])

                state = np.zeros(shape=[2] * 6)
                state[bits] = 1
                ket = qf.State(state)

                ket = circ.run(ket)
                bits2 = ket.measure()
                res = bits2[[5, 2, 3]]  # type: ignore
                res = bitlist_to_int(res)

                assert res == expected

    # Three bit addition circuit
    circ = qf.addition_circuit([0, 1, 2], [3, 4, 5], [6, 7])
    for c0 in range(0, 2):
        for a0 in range(0, 8):
            for a1 in range(0, 8):
                expected = a0 + a1 + c0
                b0 = int_to_bitlist(a0, 3)
                b1 = int_to_bitlist(a1, 3)
                bc = [c0]
                bits = tuple(b0 + b1 + bc + [0])

                state = np.zeros(shape=[2] * 8)
                state[bits] = 1
                ket = qf.State(state)

                ket = circ.run(ket)
                bits2 = ket.measure()
                res = bits2[[7, 3, 4, 5]]  # type: ignore
                res = bitlist_to_int(res)

                # print(c0, a0, a1, expected, res)
                assert res == expected

    with pytest.raises(ValueError):
        qf.addition_circuit([0, 1, 2], [3, 4, 5, 6], [7, 8])

    with pytest.raises(ValueError):
        qf.addition_circuit([0, 1, 2], [3, 4, 5], [6, 7, 8])


def test_ghz_circuit() -> None:
    N = 12
    qubits = list(range(N))
    circ = qf.ghz_circuit(qubits)
    circ.run()


def test_map_gate() -> None:
    circ = qf.map_gate(qf.X(0), [[0], [1], [2]])
    assert circ[1].qubits[0] == 1

    circ = qf.map_gate(qf.CNot(0, 1), [[0, 1], [1, 2]])
    assert circ[1].qubits == (1, 2)


def test_count_operations() -> None:
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    circ += qf.H(1)
    op_count = qf.count_operations(circ)
    assert op_count == {qf.H: 3}

    circ = qf.addition_circuit([0, 1, 2], [3, 4, 5], [6, 7])
    op_count = qf.count_operations(circ)
    assert op_count == {qf.CNot: 13, qf.CCNot: 6}


def test_graph_circuit() -> None:
    graph = nx.grid_graph([2, 3])
    layers = 8

    params = qf.graph_circuit_params(graph, layers)
    _ = qf.graph_circuit(graph, layers, params)


def test_graph_state_circuit() -> None:
    graph = nx.grid_graph([3, 3])
    _ = qf.graph_state_circuit(graph)


def test_circuit_flat() -> None:
    circ0 = qf.Circuit([qf.X(0), qf.X(1)])
    circ1 = qf.Circuit([qf.Y(0), qf.Y(1)])
    circ2 = qf.Circuit([circ1, qf.Z(0), qf.Z(1)])
    circ = qf.Circuit([circ0, circ2])

    flat = qf.Circuit(circ.flat())
    assert len(flat) == 6
    assert flat[2].name == "Y"


def test_circuit_params() -> None:
    circ = qf.Circuit()
    circ += qf.X(0) ** 0.3
    circ += qf.Swap(1, 2)

    assert len(circ.params) == 1
    assert circ.params == (0.3,)

    with pytest.raises(ValueError):
        _ = circ.param("theta")
