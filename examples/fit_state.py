#!/usr/bin/env python

"""
QuantumFlow Examples:
    Train a circuit to geenerate a state using gradient descent
"""

import networkx as nx
import quantumflow as qf


def fit_state(graph: nx.graph,
              steps: int,
              target_ket: qf.State,
              train_steps: int = 200,
              learning_rate: float = 0.005) -> qf.Circuit:
    """Use Stocastic Gradient Descent to train a circuit to
    generate a particular target state. Gradients are
    calcuated using middle out algorithm."""

    ket0 = qf.zero_state(target_ket.qubits)
    ket1 = target_ket
    params = qf.graph_circuit_params(graph, steps)
    opt = qf.Adam(learning_rate)

    for step in range(train_steps):
        circ = qf.graph_circuit(graph, steps, params)

        ang = qf.state_angle(circ.run(ket0), ket1)
        print(step, ang)
        grads = qf. state_angle_gradients(ket0, ket1, circ)
        params = opt.get_update(params, grads)

        if ang < 0.05:
            break

    return circ


def example_fit_state():
    graph = nx.grid_graph([2, 2])
    layers = 4
    qubits = graph.nodes()
    target_ket = qf.w_state(qubits)

    circ = fit_state(graph, layers, target_ket, train_steps=2000)
    ket1 = circ.run(qf.zero_state(qubits))

    assert qf.state_angle(target_ket, ket1) < 0.05

    return(circ)


if __name__ == "__main__":
    def main():
        """CLI"""
        print(fit_state.__doc__)
        print('Fitting to 6 qubit W state...')
        circ = example_fit_state()
        print(circ)

    main()
