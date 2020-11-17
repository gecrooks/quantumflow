# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.xqiskit
"""

import pytest

pytest.importorskip("qiskit")

from qiskit import (  # noqa: E402
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
)

import quantumflow as qf  # noqa: E402
from quantumflow.xqiskit import QiskitSimulator  # noqa: E402
from quantumflow.xqiskit import circuit_to_qiskit  # noqa: E402
from quantumflow.xqiskit import qiskit_to_circuit  # noqa: E402
from quantumflow.xqiskit import translate_gates_to_qiskit  # noqa: E402


def test_qiskit_to_circuit() -> None:

    q = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(q, c)

    qc.ccx(q[0], q[1], q[2])
    qc.ch(q[0], q[1])
    qc.crz(0.1, q[0], q[1])
    qc.cswap(q[0], q[1], q[2])

    # The QuantumCircuit.cu1 method is deprecated as of 0.16.0.
    # You should use the QuantumCircuit.cp method instead, which acts identically.

    # The QuantumCircuit.cu3 method is deprecated as of 0.16.0.
    # You should use the QuantumCircuit.cu method instead,
    # where cu3(ϴ,φ,λ) = cu(ϴ,φ,λ,0).

    # The QuantumCircuit.u1 method is deprecated as of 0.16.0.
    # You should use the QuantumCircuit.p method instead, which acts identically.

    # The QuantumCircuit.u2 method is deprecated as of 0.16.0.
    # You can use the general 1-qubit gate QuantumCircuit.u instead:
    # u2(φ,λ) = u(π/2, φ, λ).
    # Alternatively, you can decompose it in terms of QuantumCircuit.p
    # and QuantumCircuit.sx
    # u2(φ,λ) = p(π/2+φ) sx p(π/2+λ) (1 pulse on hardware).

    # The QuantumCircuit.u3 method is deprecated as of 0.16.0.
    # You should use QuantumCircuit.u instead, which acts identically.
    # Alternatively, you can decompose u3 in terms of QuantumCircuit.p
    # and QuantumCircuit.sx
    # u3(ϴ,φ,λ) = p(φ+π) sx p(ϴ+π) sx p(λ) (2 pulses on hardware).

    qc.cu1(0.1, q[0], q[1])
    qc.cu3(0.1, 0.2, 0.3, q[0], q[1])
    qc.cx(q[0], q[1])
    qc.cy(q[0], q[1])
    qc.cz(q[0], q[1])
    qc.h(q[0])
    qc.i(q[1])
    qc.i(q[2])
    qc.rx(0.0, q[0])
    qc.ry(0.1, q[1])
    qc.rz(0.2, q[2])
    qc.rzz(0.1, q[0], q[1])
    qc.s(q[2])
    qc.sdg(q[2])
    qc.swap(q[0], q[1])
    qc.t(q[1])
    qc.tdg(q[1])
    qc.u1(0.2, q[2])
    qc.u2(0.1, 0.2, q[2])
    qc.u3(0.1, 0.2, 0.3, q[2])
    qc.x(q[0])
    qc.y(q[0])
    qc.z(q[0])

    circ = qiskit_to_circuit(qc)
    # print(circ)

    assert (
        str(circ)
        == """Circuit
    CCNot 0 1 2
    CH 0 1
    CRZ(1/10) 0 1
    CSwap 0 1 2
    CPhase(1/10) 0 1
    CU3(1/10, 1/5, 3/10) 0 1
    CNot 0 1
    CY 0 1
    CZ 0 1
    H 0
    I 1
    I 2
    Rx(0) 0
    Ry(1/10) 1
    Rz(1/5) 2
    RZZ(1/10) 0 1
    S 2
    S_H 2
    Swap 0 1
    T 1
    T_H 1
    PhaseShift(1/5) 2
    U2(1/10, 1/5) 2
    U3(1/10, 1/5, 3/10) 2
    X 0
    Y 0
    Z 0"""
    )

    circuit_to_qiskit(circ)


def test_qiskit_if() -> None:
    q = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(q, c)

    qc.x(q[2]).c_if(c, 1)

    circ = qiskit_to_circuit(qc)
    op = circ[0]
    assert isinstance(op, qf.If)
    assert op.element.name == "X"
    assert op.key == c
    assert op.value == 1


def test_circuit_to_qiskit() -> None:
    circ = qf.Circuit()
    circ += qf.X(0)
    circ += qf.Y(1)
    circ += qf.Z(2)
    circ += qf.Can(0.1, 0.2, 0.2, 0, 1)

    circ1 = translate_gates_to_qiskit(circ)
    print()
    print(qf.circuit_to_diagram(circ1))

    qc = circuit_to_qiskit(circ, translate=True)
    print(qc)

    assert len(circ1) == len(qc)


def test_qiskitsimulator() -> None:
    circ = qf.Circuit()
    circ += qf.H(1)
    circ += qf.X(0)
    circ += qf.H(2)
    circ += qf.Y(3)
    circ += qf.Z(2)
    circ += qf.Can(0.1, 0.2, 0.2, 0, 1)

    sim = QiskitSimulator(circ)
    assert qf.states_close(circ.run(), sim.run())

    ket0 = qf.random_state([0, 1, 2, 3])
    assert qf.states_close(circ.run(ket0), sim.run(ket0))


# fin
