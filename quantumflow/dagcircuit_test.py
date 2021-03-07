# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.dagcircuit
"""


import numpy as np
import pytest

import quantumflow as qf
from quantumflow.dagcircuit import In, Out


# TODO Refactor in test_circuit
def _test_circ() -> qf.Circuit:
    # Adapted from referenceQVM
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
    return circ


def _true_ket() -> qf.State:
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


def test_init() -> None:
    dag = qf.DAGCircuit([])
    assert dag.size() == 0


def test_inverse() -> None:
    dag = qf.DAGCircuit(_test_circ())
    inv_dag = dag.H

    ket0 = qf.random_state(2)
    ket1 = dag.run(ket0)
    ket2 = inv_dag.run(ket1)

    assert qf.states_close(ket0, ket2)


def test_ascircuit() -> None:
    circ0 = qf.ghz_circuit(range(5))
    dag = qf.DAGCircuit(circ0)
    circ1 = qf.Circuit(dag)

    assert tuple(circ1.qubits) == (0, 1, 2, 3, 4)
    assert dag.qubits == circ0.qubits
    assert dag.qubit_nb == 5


def test_str() -> None:
    circ0 = qf.ghz_circuit(range(5))
    dag = qf.DAGCircuit(circ0)
    string = str(dag)
    assert (
        string
        == """DAGCircuit
    H 0
    CNot 0 1
    CNot 1 2
    CNot 2 3
    CNot 3 4\
"""
    )


def test_asgate() -> None:
    circ0 = qf.zyz_circuit(0.1, 2.2, 0.5, 0)
    gate0 = circ0.asgate()
    dag0 = qf.DAGCircuit(circ0)
    gate1 = dag0.asgate()
    assert qf.gates_close(gate0, gate1)


def test_evolve() -> None:
    rho0 = qf.random_state(3).asdensity()
    rho1 = qf.CCNot(0, 1, 2).evolve(rho0)

    dag = qf.DAGCircuit(qf.translate_ccnot_to_cnot(qf.CCNot(0, 1, 2)))
    rho2 = dag.evolve(rho0)

    assert qf.densities_close(rho1, rho2)


def test_aschannel() -> None:
    rho0 = qf.random_state(3).asdensity()
    rho1 = qf.CCNot(0, 1, 2).evolve(rho0)

    dag = qf.DAGCircuit(qf.translate_ccnot_to_cnot(qf.CCNot(0, 1, 2)))
    chan = dag.aschannel()
    rho2 = chan.evolve(rho0)

    assert qf.densities_close(rho1, rho2)


def test_depth() -> None:
    circ = qf.Circuit(qf.QFTGate([0, 1, 2, 3]).decompose())
    dag = qf.DAGCircuit(circ)
    assert dag.depth() == 8

    circ = qf.ghz_circuit(range(5))
    dag = qf.DAGCircuit(circ)
    assert dag.depth() == 5

    assert dag.depth(local=False) == 4


def test_moments() -> None:
    circ0 = qf.ghz_circuit(range(5))
    dag = qf.DAGCircuit(circ0)
    circ = dag.moments()
    assert circ.size() == dag.depth()

    circ1 = qf.Circuit(
        [
            qf.Z(0),
            qf.Z(1),
            qf.Z(2),
            qf.CNot(0, 1),
            qf.Measure(0, 0),
            qf.Measure(1, 1),
            qf.Measure(2, 2),
        ]
    )
    moments = qf.DAGCircuit(circ1).moments()
    print()
    print(moments)

    assert len(moments) == 3
    assert len(moments[0]) == 3  # type: ignore
    assert len(moments[1]) == 1  # type: ignore
    assert len(moments[2]) == 3  # type: ignore

    with pytest.warns(DeprecationWarning):
        _ = dag.layers()


def test_components() -> None:
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    dag = qf.DAGCircuit(circ)
    assert dag.component_nb() == 2

    circ += qf.CNot(0, 1)
    dag = qf.DAGCircuit(circ)
    assert dag.component_nb() == 1

    circ0 = qf.ghz_circuit([0, 2, 4, 6, 8])
    circ1 = qf.ghz_circuit([1, 3, 5, 7, 9])

    circ = circ0 + circ1
    dag = qf.DAGCircuit(circ)
    comps = dag.components()
    assert dag.component_nb() == 2
    assert len(comps) == 2

    circ0 = qf.Circuit(qf.QFTGate([0, 2, 4, 6]).decompose())
    circ1 = qf.ghz_circuit([1, 3, 5, 7])
    circ += circ0
    circ += circ1
    circ += qf.H(10)
    dag = qf.DAGCircuit(circ)
    comps = dag.components()
    assert dag.component_nb() == 3
    assert len(comps) == 3


def test_next_prev() -> None:
    circ = qf.ghz_circuit([0, 2, 4, 6, 8])
    elem = circ[3]
    dag = qf.DAGCircuit(circ)

    assert dag.next_element(elem, elem.qubits[1]) == circ[4]
    assert dag.prev_element(elem, elem.qubits[0]) == circ[2]

    assert dag.next_element(elem, elem.qubits[0]) == Out(4)
    assert dag.prev_element(elem, elem.qubits[1]) == In(6)


def test_on() -> None:
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    dag = qf.DAGCircuit(circ)

    dag = dag.on(2, 3)
    assert dag.qubits == (2, 3)

    dag = dag.rewire({2: 4, 3: 6})
    assert dag.qubits == (4, 6)

    with pytest.raises(ValueError):
        dag.on(2, 3, 5)


# fin
