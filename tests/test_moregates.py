
from numpy import pi, random
import numpy as np

import quantumflow as qf
from quantumflow.visualization import kwarg_to_symbol

import pytest

from . import REPS, ALMOST_ZERO

# TODO: Refactor to match split of gates in gates subpackage.


def test_XY():
    gate = qf.XY(-0.1)
    assert qf.gates_close(gate, qf.ISWAP()**0.2)


# TODO: more tests
def test_BARENCO():
    gate = qf.Barenco(0.1, 0.2, 0.3, 0, 1)
    print('Barenco gate')
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
    # But it clearly far too long. CH is locally equivalent to CNOT,
    # so requires only one CNOT gate.
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

    # Here's a better decomposition
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


def test_cu1():
    # Test that QASM's cu1 gate is the same as CPHASE up to global phase

    for _ in range(REPS):
        theta = random.uniform(0, 4)
        circ0 = qf.Circuit([
            qf.RZ(theta / 2, 0),
            qf.CNOT(0, 1),
            qf.RZ(-theta/2, 1),
            qf.CNOT(0, 1),
            qf.RZ(theta / 2, 1),
            ])
        gate0 = circ0.asgate()
        gate1 = qf.CPHASE(theta, 0, 1)
        assert qf.gates_close(gate0, gate1)


def test_CU3():
    theta = 0.2
    phi = 2.3
    lam = 1.1

    gate = qf.Circuit([
        qf.CU3(theta, phi, lam),
        qf.CU3(theta, phi, lam).H
        ]).asgate()

    # assert qf.gates_close(gate, qf.IDEN(0, 1))
    assert qf.almost_identity(gate)

    cgate = qf.control_gate(0, qf.U3(theta, phi, lam, 1))
    print()
    qf.print_gate(cgate.su())
    print()
    qf.print_gate(qf.CU3(theta, phi, lam).su())

    print(qf.gate_angle(qf.CU3(theta, phi, lam).su(), cgate.su()))

    print(qf.gate_angle(qf.CU3(theta, phi, lam).su(), cgate.su()))

    print(cgate.su().tensor - qf.CU3(theta, phi, lam).su().tensor)

    assert qf.gates_close(qf.CU3(theta, phi, lam), cgate)


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


def test_FSim():
    for _ in range(REPS):
        theta = random.uniform(-pi, +pi)
        phi = random.uniform(-pi, +pi)
        gate0 = qf.FSim(theta, phi)

        # Test with decomposition from Cirq.
        circ = qf.Circuit()
        circ += qf.XX(theta / pi, 0, 1)
        circ += qf.YY(theta / pi, 0, 1)
        circ += qf.CZ(0, 1) ** (- phi / pi)
        gate1 = circ.asgate()
        assert qf.gates_close(gate0, gate1)

        assert qf.gates_close(gate1.H, gate0.H)


def test_PhasedX():
    q0 = '4'
    for _ in range(REPS):
        t = random.uniform(0, 1)
        p = random.uniform(-2, +2)

        gate0 = qf.PhasedX(p, q0)
        gate1 = qf.PhasedXPow(p, t, q0)
        assert qf.gates_close(gate0 ** t, gate1)
        assert (gate0 ** t).qubits == (q0,)

        gate0.H
        gate1.H
        gate2 = gate1 ** t
        p2, t2 = gate2.params.values()
        assert p2 == p
        assert t2 - t ** 2 == ALMOST_ZERO

        assert qf.gates_close(gate0, gate0.specialize())

    assert qf.gates_close(qf.PhasedX(-2.0, q0).specialize(), qf.X(q0))


# FIXME: Move to compounds
# def test_IDEN():
#     gate0 = qf.IDEN(0, 1)
#     assert gate0.qubit_nb == 2

#     gate1 = qf.IDEN(0, 1, 4, 5)
#     assert gate1.qubit_nb == 4

#     assert gate1 ** 0.5 == gate1

#     ket = qf.random_state([0, 1])
#     assert gate0.run(ket) == ket

#     rho = qf.random_density([0, 1])
#     assert gate0.evolve(rho) == rho


optimized_run_gates = [
    qf.I(), qf.X(), qf.Z(), qf.Y(), qf.H(), qf.T(), qf.S(), qf.T_H(),
    qf.S_H(), qf.TX(0.1), qf.TY(0.2), qf.TZ(0.2),
    qf.CNOT(), qf.CZ(), qf.SWAP(), qf.CCNOT(),
    qf.CSWAP(), qf.CCZ(), qf.PhaseShift(0.2),
    qf.ISWAP()]


@pytest.mark.parametrize("gate", optimized_run_gates)
def test_optimized_run(gate):
    # Some gates have specially optimized run() methods for faster simulation.
    # Here, make sure optimizations produces same result as direct application
    # of gate tensor.

    ket = qf.random_state([0, 1, 2])

    gate0 = gate
    gate1 = qf.Unitary(gate0.tensor)

    ket0 = gate0.run(ket)
    ket1 = gate1.run(ket)

    print()
    qf.print_state(ket0)
    print()
    qf.print_state(ket1)
    print()

    assert qf.states_close(ket0, ket1)


def test_gates_unitary():
    for _ in range(REPS):
        param_values = {name: np.random.uniform(-4*pi, 4*pi)
                        for name in kwarg_to_symbol}
        for gatet in qf.STD_GATESET:
            args = [param_values[a] for a in gatet.args()]
            gate = gatet(*args)

            print(gate)
            assert qf.almost_unitary(gate)


@pytest.mark.parametrize("gatet", qf.STD_GATESET)
def test_gate_hamiltonians(gatet):

    param_values = {name: np.random.uniform(-4, 4) for name in kwarg_to_symbol}
    qubits = [5, 3, 4, 2, 8]

    args = [param_values[a] for a in gatet.args()]
    gate0 = gatet(*args)
    gate0 = gate0.on(*qubits[:gate0.qubit_nb])

    print(gate0)

    qbs = gate0.qubits
    ham = gate0.hamiltonian
    gate1 = qf.unitary_from_hamiltonian(ham, *qbs)

    assert qf.gates_close(gate0, gate1)

    # Check that gates that gates have same phase
    # FIXME!
    # Currently Gate.hamiltonian is broken here, so only check
    # subclasses that override hamiltonian
    if gatet.hamiltonian is not qf.Gate.hamiltonian:
        print('Checking gates phase close...')
        assert qf.gates_phase_close(gate0, gate1)


def test_tz_specialize():
    for t in [-0.25, 0, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0]:
        gate0 = qf.TZ(t)
        gate1 = gate0.specialize()
        assert qf.gates_close(gate0, gate1)
        assert not isinstance(gate1, qf.TZ)


# fin
