
import pytest

import networkx as nx
import numpy as np

import quantumflow as qf

from . import ALMOST_ZERO, tensorflow2_only, skip_torch


@skip_torch  # FIXME
def test_gradients():
    # This test only checks that code runs, not that we get correct answers
    # graph = nx.grid_graph([2, 3])
    graph = nx.grid_graph([2, 1])
    layers = 2
    params = qf.graph_circuit_params(graph, layers)
    circ = qf.graph_circuit(graph, layers, params)
    circ += qf.H((1, 1))  # add a non-parameteried gate. Should be ignored

    qubits = circ.qubits
    ket0 = qf.zero_state(qubits)
    ket1 = qf.random_state(qubits)

    grads0 = qf.state_fidelity_gradients(ket0, ket1, circ)
    # print(grads0)

    grads1 = qf.state_angle_gradients(ket0, ket1, circ)
    # print(grads1)

    proj = qf.Projection([ket1])
    grads2 = qf.expectation_gradients(ket0, circ, hermitian=proj)
    # print(grads2)

    # Check that qf.expectation_gradients() gives same answers for
    # fidelity as f.state_fidelity_gradients()
    for g0, g1 in zip(grads0, grads2):
        assert qf.asarray(g0 - g1) == ALMOST_ZERO
        print(g0, g1)

    proj = qf.Projection([ket1])
    grads3 = qf.expectation_gradients(ket0, circ, hermitian=proj,
                                      dfunc=lambda fid:
                                      - 1 / (2 * np.sqrt((1-fid) * fid)))
    # print(grads3)

    for g0, g1 in zip(grads1, grads3):
        assert qf.asarray(g0 - g1) == ALMOST_ZERO
        print(g0, g1)


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

    with pytest.raises(ValueError):
        qf.expectation_gradients(ket0, circ, qf.I(0, 1))


@skip_torch  # FIXME
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


@tensorflow2_only
def test_fidelity_gradients_tf():
    # Compare gradients cacluated using middle_out versus tensorflow
    import tensorflow as tf
    N = 2
    graph = nx.grid_graph([N, 1])
    steps = 1
    params = qf.graph_circuit_params(graph, steps)
    circ = qf.graph_circuit(graph, steps, params)

    target_ket = qf.w_state(circ.qubits)
    ket0 = qf.random_state(circ.qubits)

    grads = qf.state_fidelity_gradients(ket0, target_ket, circ)
    print(grads)

    tf_t = tf.Variable(params)
    with tf.GradientTape() as t:
        t.watch(tf_t)
        tf_circ = qf.graph_circuit(graph, steps, tf_t)
        fid = qf.state_fidelity(tf_circ.run(ket0), target_ket)
    tf_grads = qf.asarray(t.gradient(fid, tf_t))

    for qf_g, tf_g in zip(grads, tf_grads):
        assert qf_g-tf_g == ALMOST_ZERO

    gang = qf.state_angle_gradients(ket0, target_ket, circ)
    with tf.GradientTape() as t:
        t.watch(tf_t)
        tf_circ = qf.graph_circuit(graph, steps, tf_t)
        fid = qf.state_angle(tf_circ.run(ket0), target_ket)
    tf_grads = qf.asarray(t.gradient(fid, tf_t))

    for qf_g, tf_g in zip(gang, tf_grads):
        assert qf_g-tf_g == ALMOST_ZERO
