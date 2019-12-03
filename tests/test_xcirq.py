

import pytest
pytest.importorskip("cirq")      # noqa: 402


from numpy import pi

import cirq as cq
import quantumflow as qf
from quantumflow.xcirq import (from_cirq_qubit, to_cirq_qubit,
                               cirq_to_circuit, circuit_to_cirq,
                               CirqSimulator)


def test_from_cirq_qubit():
    assert from_cirq_qubit(cq.GridQubit(2, 3)) == (2, 3)
    assert from_cirq_qubit(cq.NamedQubit("A Name")) == "A Name"
    assert from_cirq_qubit(cq.LineQubit(101)) == 101


def test_to_cirq_qubit():
    assert cq.GridQubit(2, 3) == to_cirq_qubit((2, 3))
    assert cq.NamedQubit("A Name") == to_cirq_qubit("A Name")
    assert cq.LineQubit(101) == to_cirq_qubit(101)


def test_cirq_to_circuit():
    q0 = cq.LineQubit(0)
    q1 = cq.LineQubit(1)
    q2 = cq.LineQubit(2)

    gate = cirq_to_circuit(cq.Circuit(cq.X(q0)))[0]
    assert isinstance(gate, qf.X)
    assert gate.qubits == (0,)

    gate = cirq_to_circuit(cq.Circuit(cq.X(q1)**0.4))[0]
    assert isinstance(gate, qf.TX)
    assert gate.qubits == (1,)

    gate = cirq_to_circuit(cq.Circuit(cq.CZ(q1, q0)))[0]
    assert isinstance(gate, qf.CZ)
    assert gate.qubits == (1, 0)

    gate = cirq_to_circuit(cq.Circuit(cq.CZ(q1, q0) ** 0.3))[0]
    assert isinstance(gate, qf.CZPow)
    assert gate.qubits == (1, 0)
    assert gate.params['t'] == 0.3

    gate = cirq_to_circuit(cq.Circuit(cq.CNOT(q0, q1)))[0]
    assert isinstance(gate, qf.CNOT)
    assert gate.qubits == (0, 1)

    gate = cirq_to_circuit(cq.Circuit(cq.CNOT(q0, q1) ** 0.25))[0]
    assert isinstance(gate, qf.CNotPow)
    assert gate.qubits == (0, 1)
    assert gate.params['t'] == 0.25

    gate = cirq_to_circuit(cq.Circuit(cq.SWAP(q0, q1)))[0]
    assert isinstance(gate, qf.SWAP)

    gate = cirq_to_circuit(cq.Circuit(cq.ISWAP(q0, q1)))[0]
    assert isinstance(gate, qf.ISWAP)

    gate = cirq_to_circuit(cq.Circuit(cq.CSWAP(q0, q1, q2)))[0]
    assert isinstance(gate, qf.CSWAP)

    gate = cirq_to_circuit(cq.Circuit(cq.CCX(q0, q1, q2)))[0]
    assert isinstance(gate, qf.CCNOT)

    gate = cirq_to_circuit(cq.Circuit(cq.CCZ(q0, q1, q2)))[0]
    assert isinstance(gate, qf.CCZ)

    gate = cirq_to_circuit(cq.Circuit(cq.I(q0)))[0]
    assert isinstance(gate, qf.I)

    gate = cirq_to_circuit(cq.Circuit(cq.XX(q0, q2)))[0]
    assert isinstance(gate, qf.XX)
    assert gate.params['t'] == 1.0

    gate = cirq_to_circuit(cq.Circuit(cq.XX(q0, q2) ** 0.3))[0]
    assert isinstance(gate, qf.XX)
    assert gate.params['t'] == 0.3

    gate = cirq_to_circuit(cq.Circuit(cq.YY(q0, q2)))[0]
    assert isinstance(gate, qf.YY)
    assert gate.params['t'] == 1.0

    gate = cirq_to_circuit(cq.Circuit(cq.YY(q0, q2) ** 0.3))[0]
    assert isinstance(gate, qf.YY)
    assert gate.params['t'] == 0.3

    gate = cirq_to_circuit(cq.Circuit(cq.ZZ(q0, q2)))[0]
    assert isinstance(gate, qf.ZZ)
    assert gate.params['t'] == 1.0

    gate = cirq_to_circuit(cq.Circuit(cq.ZZ(q0, q2) ** 0.3))[0]
    assert isinstance(gate, qf.ZZ)
    assert gate.params['t'] == 0.3

    # Check that cirq's parity gates are the same as QF's XX, YY, ZZ
    # upto parity
    U = (cq.XX(q0, q2) ** 0.8)._unitary_()
    gate0 = qf.Unitary(U)
    assert qf.gates_close(gate0, qf.XX(0.8))

    U = (cq.YY(q0, q2) ** 0.3)._unitary_()
    gate0 = qf.Unitary(U)
    assert qf.gates_close(gate0, qf.YY(0.3))

    U = (cq.ZZ(q0, q2) ** 0.2)._unitary_()
    gate0 = qf.Unitary(U)
    assert qf.gates_close(gate0, qf.ZZ(0.2))


def version_to_versioninfo(version):
    return tuple(int(c) if c.isdigit() else c for c in version.split('.'))


@pytest.mark.skipif(version_to_versioninfo(cq.__version__) >= (0, 7),
                    reason='Deprecated in 0.7.0')
def test_cirq_to_circuit_0_6():
    q0 = cq.LineQubit(0)
    gate = cirq_to_circuit(cq.Circuit(cq.Rx(0.5).on(q0)))[0]
    assert isinstance(gate, qf.TX)
    assert gate.params['t'] == 0.5/pi

    gate = cirq_to_circuit(cq.Circuit(cq.Ry(0.5).on(q0)))[0]
    assert isinstance(gate, qf.TY)
    assert gate.params['t'] == 0.5/pi

    gate = cirq_to_circuit(cq.Circuit(cq.Rz(0.5).on(q0)))[0]
    assert isinstance(gate, qf.TZ)
    assert gate.params['t'] == 0.5/pi


@pytest.mark.skipif(version_to_versioninfo(cq.__version__) < (0, 7),
                    reason='New interface in 0.7.0')
def test_cirq_to_circuit_0_7():
    q0 = cq.LineQubit(0)
    q1 = cq.LineQubit(1)
    gate = cirq_to_circuit(cq.Circuit(cq.rx(0.5).on(q0)))[0]
    assert isinstance(gate, qf.TX)
    assert gate.params['t'] == 0.5/pi

    gate = cirq_to_circuit(cq.Circuit(cq.ry(0.5).on(q0)))[0]
    assert isinstance(gate, qf.TY)
    assert gate.params['t'] == 0.5/pi

    gate = cirq_to_circuit(cq.Circuit(cq.rz(0.5).on(q0)))[0]
    assert isinstance(gate, qf.TZ)
    assert gate.params['t'] == 0.5/pi

    gate = cirq_to_circuit(cq.Circuit(cq.IdentityGate(2).on(q0, q1)))[0]
    assert isinstance(gate, qf.IDEN)

    op = (cq.PhasedISwapPowGate()**0.5).on(q0, q1)
    circ = cirq_to_circuit(cq.Circuit(op))
    assert qf.gates_close(circ.asgate(), qf.Givens(0.5 * pi/2))

    op = cq.PhasedXZGate(x_exponent=0.125,
                         z_exponent=0.25,
                         axis_phase_exponent=0.375).on(q0)
    circ = cirq_to_circuit(cq.Circuit(op))
    assert len(circ) == 3


def test_cirq_to_circuit2():

    q0 = cq.GridQubit(0, 0)
    q1 = cq.GridQubit(1, 0)

    def basic_circuit(meas=False):
        sqrt_x = cq.X**0.5
        yield cq.X(q0) ** 0.5, sqrt_x(q1)
        yield cq.CZ(q0, q1)
        yield sqrt_x(q0), sqrt_x(q1)
        if meas:
            yield cq.measure(q0, key='q0'), cq.measure(q1, key='q1')

    cqc = cq.Circuit()
    cqc.append(basic_circuit())

    print()
    print(cqc)

    circ = cirq_to_circuit(cqc)
    print()
    print(qf.circuit_to_diagram(circ))


def test_circuit_to_circ():
    q0, q1, q2 = 'q0', 'q1', 'q2'

    circ0 = qf.Circuit()
    circ0 += qf.I(q0)
    circ0 += qf.X(q1)
    circ0 += qf.Y(q2)

    circ0 += qf.Z(q0)
    circ0 += qf.S(q1)
    circ0 += qf.T(q2)

    circ0 += qf.H(q0)
    circ0 += qf.H(q1)
    circ0 += qf.H(q2)

    circ0 += qf.TX(0.6, q0)
    circ0 += qf.TY(0.6, q1)
    circ0 += qf.TZ(0.6, q2)

    circ0 += qf.XX(0.2, q0, q1)
    circ0 += qf.YY(0.3, q1, q2)
    circ0 += qf.ZZ(0.4, q2, q0)

    circ0 += qf.CZ(q0, q1)
    circ0 += qf.CNOT(q0, q1)
    circ0 += qf.SWAP(q0, q1)
    circ0 += qf.ISWAP(q0, q1)

    circ0 += qf.CCZ(q0, q1, q2)
    circ0 += qf.CCNOT(q0, q1, q2)
    circ0 += qf.CSWAP(q0, q1, q2)

    circ0 += qf.FSim(1, 2, q0, q1)

    diag0 = qf.circuit_to_diagram(circ0)
    print()
    print(diag0)

    cqc = circuit_to_cirq(circ0)
    circ1 = cirq_to_circuit(cqc)

    diag1 = qf.circuit_to_diagram(circ1)
    print()
    print(diag1)

    assert diag0 == diag1


def test_circuit_to_circ_exception():
    circ0 = qf.Circuit([qf.CAN(0.2, 0.3, 0.1)])
    with pytest.raises(NotImplementedError):
        circuit_to_cirq(circ0)


def test_circuit_to_circ_translate():
    circ0 = qf.Circuit([qf.CAN(0.2, 0.3, 0.1)])
    cqc = circuit_to_cirq(circ0, translate=True)
    circ2 = cirq_to_circuit(cqc)
    assert qf.circuits_close(circ0, circ2)


def test_cirq_simulator():
    q0, q1, q2 = 'q0', 'q1', 'q2'

    circ0 = qf.Circuit()
    circ0 += qf.I(q0)
    circ0 += qf.I(q1)
    circ0 += qf.I(q2)
    circ0 += qf.X(q1)
    circ0 += qf.Y(q2)

    circ0 += qf.Z(q0)
    circ0 += qf.S(q1)
    circ0 += qf.T(q2)

    circ0 += qf.H(q0)
    circ0 += qf.H(q1)
    circ0 += qf.H(q2)

    circ0 += qf.TX(0.6, q0)
    circ0 += qf.TY(0.6, q1)
    circ0 += qf.TZ(0.6, q2)

    circ0 += qf.XX(0.2, q0, q1)
    circ0 += qf.YY(0.3, q1, q2)
    circ0 += qf.ZZ(0.4, q2, q0)

    circ0 += qf.CZ(q0, q1)
    circ0 += qf.CNOT(q0, q1)
    circ0 += qf.SWAP(q0, q1)
    circ0 += qf.ISWAP(q0, q1)

    circ0 += qf.CCZ(q0, q1, q2)
    circ0 += qf.CCNOT(q0, q1, q2)
    circ0 += qf.CSWAP(q0, q1, q2)

    ket0 = qf.random_state([q0, q1, q2])
    ket1 = circ0.run(ket0)
    sim = CirqSimulator(circ0)
    ket2 = sim.run(ket0)

    assert ket1.qubits == ket2.qubits

    print(qf.state_angle(ket1, ket2))
    assert qf.states_close(ket1, ket2)

    assert qf.states_close(circ0.run(), sim.run())


def test_circuit_to_circ_unitary():
    gate0 = qf.random_gate([4, 5])
    circ0 = qf.Circuit([gate0])
    cqc = circuit_to_cirq(circ0)
    circ1 = cirq_to_circuit(cqc)
    assert qf.circuits_close(circ0, circ1)
