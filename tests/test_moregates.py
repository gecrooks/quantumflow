
from numpy import pi, random

import quantumflow as qf

from . import REPS, ALMOST_ZERO, skip_torch

# TODO: Refactor to match split of gates in gates subpackage.


def test_BARENCO():
    gate = qf.BARENCO(0.1, 0.2, 0.3, 0, 1)
    print('BARENCO gate')
    qf.print_gate(gate)
    print(gate.qubits)


def test_V():
    gate = qf.V(0)
    print(gate.tensor, gate.qubits)
    qf.print_gate(gate)

    assert qf.gates_close(qf.V(0), qf.TX(0.5, 0))
    assert qf.gates_close(qf.V_H(0), qf.TX(-0.5, 0))
    assert qf.gates_close(qf.V(1).H, qf.V_H(1))
    assert qf.gates_close(qf.V(1).H.H, qf.V(1))
    assert qf.gates_close(qf.V() ** 0.5, qf.TX(0.25))
    assert qf.gates_close(qf.V_H() ** 0.5, qf.TX(-0.25))


def test_CV():
    gate0 = qf.CNOT(0, 1) ** 0.5
    gate1 = qf.CV(0, 1)
    assert qf.gates_close(gate0, gate1)

    gate2 = qf.CV_H(0, 1)
    assert qf.gates_close(gate0.H, gate2)

    assert qf.gates_close(gate1.H.H, gate1)


def test_CY():
    gate0 = qf.control_gate(0, qf.Y(1))
    gate1 = qf.CY(0, 1)
    assert qf.gates_close(gate0, gate1)
    assert gate1 is gate1.H


def test_CH():
    gate0 = qf.control_gate(0, qf.H(1))
    gate1 = qf.CH(0, 1)
    assert qf.gates_close(gate0, gate1)
    assert gate1 is gate1.H

    # I picked up this circuit for a CH gate from qiskit
    # qiskit/extensions/standard/ch.py
    # But it clearly far too long. CH is locally equivelent to CNOT,
    # so should require only one CNOT gate.
    circ2 = qf.Circuit([
        qf.H(1),
        qf.S_H(1),
        qf.CNOT(0, 1),
        qf.H(1),
        qf.T(1),
        qf.CNOT(0, 1),
        qf.T(1),
        qf.H(1),
        qf.S(1),
        qf.X(1),
        qf.S(0)
        ])
    assert qf.gates_close(gate1, circ2.asgate())

    # Here's a better  decomposition
    circ1 = qf.Circuit([
        qf.TY(+0.25, 1),
        qf.CNOT(0, 1),
        qf.TY(-0.25, 1)
        ])
    assert qf.gates_close(gate1, circ1.asgate())
    assert qf.circuits_close(circ1, circ2)


def test_U3():
    theta = 0.2
    phi = 2.3
    lam = 1.1

    gate3 = qf.Circuit([
        qf.U3(theta, phi, lam),
        qf.U3(theta, phi, lam).H
        ]).asgate()
    assert qf.almost_identity(gate3)

    gate2 = qf.Circuit([
        qf.U2(phi, lam),
        qf.U2(phi, lam).H
        ]).asgate()
    assert qf.almost_identity(gate2)

    gate1 = qf.Circuit([
        qf.U1(lam),
        qf.U1(lam).H
        ]).asgate()
    assert qf.almost_identity(gate1)


@skip_torch  # FIXME
def test_CU3():
    theta = 0.2
    phi = 2.3
    lam = 1.1

    gate = qf.Circuit([
        qf.CU3(theta, phi, lam),
        qf.CU3(theta, phi, lam).H
        ]).asgate()
    assert qf.almost_identity(gate)


def test_CRZ():
    theta = 0.23
    gate0 = qf.CRZ(theta)
    coords = qf.canonical_coords(gate0)
    assert ALMOST_ZERO == coords[0] - 0.5*theta/pi
    assert ALMOST_ZERO == coords[1]
    assert ALMOST_ZERO == coords[2]

    coords = qf.canonical_coords(gate0 ** 3.3)
    assert ALMOST_ZERO == coords[0] - 3.3*0.5*theta/pi

    gate1 = qf.Circuit([qf.CRZ(theta), qf.CRZ(theta).H]).asgate()
    assert qf.almost_identity(gate1)


def test_RZZ():
    theta = 0.23
    gate0 = qf.RZZ(theta)
    gate1 = qf.ZZ(theta/pi)
    assert qf.gates_close(gate0, gate1)
    assert qf.gates_close(gate0.H, gate1.H)
    assert qf.gates_close(gate0 ** 0.12, gate1 ** 0.12)


def test_FSIM():
    for _ in range(REPS):
        theta = random.uniform(-pi, +pi)
        phi = random.uniform(-pi, +pi)
        gate0 = qf.FSIM(theta, phi)

        # Test with decomposition from Cirq.
        # TODO: Add decomposition somewhere in QF
        circ = qf.Circuit()
        circ += qf.XX(theta / pi, 0, 1)
        circ += qf.YY(theta / pi, 0, 1)
        circ += qf.CZ(0, 1) ** (- phi / pi)
        gate1 = circ.asgate()
        assert qf.gates_close(gate0, gate1)

        assert qf.gates_close(gate1.H, gate0.H)


def test_W_TW():
    q0 = '4'
    for _ in range(REPS):
        t = random.uniform(0, 1)
        p = random.uniform(-2, +2)

        gate0 = qf.W(p, q0)
        gate1 = qf.TW(p, t, q0)
        assert qf.gates_close(gate0 ** t, gate1)
        assert (gate0 ** t).qubits == (q0,)

        gate0.H
        gate1.H
        gate2 = gate1 ** t
        p2, t2 = gate2.params.values()
        assert p2 == p
        assert t2 == t ** 2

# fin
