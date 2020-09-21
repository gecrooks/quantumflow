# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import networkx as nx
import numpy as np
import pytest

import quantumflow as qf


def test_gradients() -> None:
    # This test only checks that code runs, not that we get correct answers
    # graph = nx.grid_graph([2, 3])
    graph = nx.grid_graph([2, 1])
    layers = 2
    params = qf.graph_circuit_params(graph, layers)
    circ = qf.graph_circuit(graph, layers, params)
    circ += qf.H((1, 1))  # add a non-parameterized gate. Should be ignored

    qubits = circ.qubits
    ket0 = qf.zero_state(qubits)
    ket1 = qf.random_state(qubits)

    grads0 = qf.state_fidelity_gradients(ket0, ket1, circ)
    # print(grads0)

    _ = qf.state_angle_gradients(ket0, ket1, circ)
    # print(grads1)

    proj = qf.Projection([ket1])
    grads2 = qf.expectation_gradients(ket0, circ, hermitian=proj)
    # print(grads2)

    # Check that qf.expectation_gradients() gives same answers for
    # fidelity as f.state_fidelity_gradients()
    for g0, g1 in zip(grads0, grads2):
        assert np.isclose(g0, g1)
        print(g0, g1)


def test_gradients_func() -> None:
    graph = nx.grid_graph([2, 1])
    layers = 2
    params = qf.graph_circuit_params(graph, layers)
    circ = qf.graph_circuit(graph, layers, params)
    circ += qf.H((1, 1))  # add a non-parameterized gate. Should be ignored
    qubits = circ.qubits
    ket0 = qf.zero_state(qubits)
    ket1 = qf.random_state(qubits)

    grads1 = qf.state_angle_gradients(ket0, ket1, circ)

    proj = qf.Projection([ket1])
    grads3 = qf.expectation_gradients(
        ket0,
        circ,
        hermitian=proj,
        dfunc=lambda fid: -1 / (2 * np.sqrt((1 - fid) * fid)),
    )
    # print(grads3)

    for g0, g1 in zip(grads1, grads3):
        assert np.isclose(g0, g1)
        print(g0, g1)


def test_gradient_errors() -> None:
    circ = qf.Circuit()
    circ += qf.CPhase(0.2, 0, 1)  # Not (currently) differentiable
    qubits = circ.qubits
    ket0 = qf.zero_state(qubits)
    ket1 = qf.random_state(qubits)

    with pytest.raises(ValueError):
        qf.state_fidelity_gradients(ket0, ket1, circ)

    with pytest.raises(ValueError):
        qf.parameter_shift_circuits(circ, 0)

    with pytest.raises(ValueError):
        qf.expectation_gradients(ket0, circ, qf.IdentityGate([0, 1]))


def test_parameter_shift_circuits() -> None:
    """Checks that gradients calculated with middle out algorithm
    match gradients calculated from parameter shift rule.
    """
    graph = nx.grid_graph([2, 2])
    layers = 2
    params = qf.graph_circuit_params(graph, layers)
    circ = qf.graph_circuit(graph, layers, params)
    N = circ.qubit_nb
    qubits = circ.qubits

    ket0 = qf.zero_state(qubits)
    ket1 = qf.random_state(qubits)
    grads = qf.state_fidelity_gradients(ket0, ket1, circ)

    for n in range(N):
        r, circ0, circ1 = qf.parameter_shift_circuits(circ, n)
        fid0 = qf.state_fidelity(circ0.run(ket0), ket1)
        fid1 = qf.state_fidelity(circ1.run(ket0), ket1)
        grad = r * (fid1 - fid0)
    assert np.isclose(grad, grads[n])


# fin
