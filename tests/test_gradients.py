
import pytest

import networkx as nx

import quantumflow as qf

from . import ALMOST_ZERO


def test_graph_circuit():
    graph = nx.grid_graph([2, 3])
    layers = 8

    params = qf.graph_circuit_params(graph, layers)

    print(params)

    circ = qf.graph_circuit(graph, layers, params)

    print(circ)


def test_gradients():
    # This test only checks that code runs, not that we get correct answers
    graph = nx.grid_graph([2, 3])
    layers = 2
    params = qf.graph_circuit_params(graph, layers)
    circ = qf.graph_circuit(graph, layers, params)
    circ += qf.H((1, 1))  # add a non-parameteried gate. Should be ignored

    qubits = circ.qubits
    ket0 = qf.zero_state(qubits)
    ket1 = qf.random_state(qubits)

    grads = qf.state_fidelity_gradients(ket0, ket1, circ)
    print(grads)

    grads = qf.state_angle_gradients(ket0, ket1, circ)
    print(grads)


def test_gradient_errors():
    circ = qf.Circuit()
    circ += qf.CPHASE(0.2, 0, 1)  # Not (currently) differentiable
    qubits = circ.qubits
    ket0 = qf.zero_state(qubits)
    ket1 = qf.random_state(qubits)

    with pytest.raises(ValueError):
        qf.state_fidelity_gradients(ket0, ket1, circ)

    with pytest.raises(ValueError):
        qf.parameter_shift_circuits(circ, 0)


def test_fit_state():
    graph = nx.grid_graph([2, 3], periodic=True)
    layers = 4
    qubits = graph.nodes()
    target_ket = qf.w_state(qubits)

    circ = qf.gradients.fit_state(graph, layers, target_ket, train_steps=2000)
    ket1 = circ.run(qf.zero_state(qubits))

    assert qf.state_angle(target_ket, ket1) < 0.05


def test_parameter_shift_circuits():
    """Checks that gradients calculated with middle out algorithm
    match gradients calcuated from paramter shift rule.
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
        grad = r*(fid1-fid0)
    assert ALMOST_ZERO == grad-grads[n]
