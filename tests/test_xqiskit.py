"""
Unit tests for quantumflow.xqiskit
"""

import pytest
pytest.importorskip("qiskit")      # noqa: 402


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import quantumflow as qf
from quantumflow.xqiskit import (
    qiskit_to_circuit,
    circuit_to_qiskit,
    translate_gates_to_qiskit,
    QiskitSimulator)


def test_qiskit_to_circuit():

    q = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(q, c)

    qc.ccx(q[0], q[1], q[2])
    qc.ch(q[0], q[1])
    qc.crz(0.1, q[0], q[1])
    qc.cswap(q[0], q[1], q[2])
    qc.cu1(0.1, q[0], q[1])
    qc.cu3(0.1, 0.2, 0.3, q[0], q[1])
    qc.cx(q[0], q[1])
    qc.cy(q[0], q[1])
    qc.cz(q[0], q[1])
    qc.h(q[0])
    qc.iden(q[1])
    qc.iden(q[2])
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

    assert str(circ) == """\
CCNOT 0 1 2
CH 0 1
CRZ(1/10) 0 1
CSWAP 0 1 2
CPHASE(1/10) 0 1
CU3(1/10, 1/5, 3/10) 0 1
CNOT 0 1
CY 0 1
CZ 0 1
H 0
I 1
I 2
RX(0) 0
RY(1/10) 1
RZ(1/5) 2
RZZ(1/10) 0 1
S 2
S_H 2
SWAP 0 1
T 1
T_H 1
PHASE(1/5) 2
U2(1/10, 1/5) 2
U3(1/10, 1/5, 3/10) 2
X 0
Y 0
Z 0"""

    circuit_to_qiskit(circ)


def test_qiskit_if():
    q = QuantumRegister(5)
    c = ClassicalRegister(5)
    qc = QuantumCircuit(q, c)

    qc.x(q[2]).c_if(c, 1)

    circ = qiskit_to_circuit(qc)
    op = circ[0]
    assert isinstance(op, qf.If)
    assert op.element.name == 'X'
    assert op.key == c
    assert op.value == 1


def test_circuit_to_qiskit():
    circ = qf.Circuit()
    circ += qf.X(0)
    circ += qf.Y(1)
    circ += qf.Z(2)
    circ += qf.CAN(0.1, 0.2, 0.2, 0, 1)

    circ1 = translate_gates_to_qiskit(circ)
    print(qf.circuit_diagram(circ1))

    qc = circuit_to_qiskit(circ, translate=True)
    print(qc)

    assert len(circ1) == len(qc)


def test_qiskitsimulator():
    circ = qf.Circuit()
    circ += qf.H(1)
    circ += qf.X(0)
    circ += qf.H(2)
    circ += qf.Y(3)
    circ += qf.Z(2)
    circ += qf.CAN(0.1, 0.2, 0.2, 0, 1)

    sim = QiskitSimulator(circ)
    assert qf.states_close(circ.run(), sim.run())

    ket0 = qf.random_state([0, 1, 2, 3])
    assert qf.states_close(circ.run(ket0), sim.run(ket0))
