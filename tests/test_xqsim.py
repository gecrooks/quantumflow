

import pytest
pytest.importorskip("cirq")      # noqa: 402
pytest.importorskip("qsimcirq")  # noqa: 402


import quantumflow as qf
from quantumflow.xqsim import QSimSimulator


def test_qsim_simulator():
    q0, q1, q2 = 'q0', 'q1', 'q2'

    circ0 = qf.Circuit()

    circ0 += qf.I(q0)

    circ0 += qf.H(q0)
    circ0 += qf.Z(q1)
    circ0 += qf.Z(q2)
    circ0 += qf.Z(q1)
    circ0 += qf.S(q1)
    circ0 += qf.T(q2)

    circ0 += qf.H(q0)
    circ0 += qf.H(q2)

    # Waiting for bugfix in qsim
    circ0 += qf.Z(q1)**0.2
    circ0 += qf.X(q1)**0.2
    circ0 += qf.TX(0.2, q0)
    circ0 += qf.TY(0.2, q1)
    circ0 += qf.TZ(0.5, q2)

    circ0 += qf.CZ(q0, q1)
    circ0 += qf.CNOT(q0, q1)
    # circ0 += qf.SWAP(q0, q1)   # No SWAP!
    #  circ0 += qf.ISWAP(q0, q1) # Waiting for bugfix in qsim
    circ0 += qf.FSim(0.1, 0.2, q0, q1)

    # No 3-qubit gates

    # Initial state not yet supported in qsim
    # ket0 = qf.random_state([q0, q1, q2])
    ket1 = circ0.run()
    sim = QSimSimulator(circ0)
    ket2 = sim.run()

    assert ket1.qubits == ket2.qubits

    print('QF', ket1)
    print('QS', ket2)

    assert qf.states_close(ket1, ket2)
    assert qf.states_close(circ0.run(), sim.run())


def test_qsim_translate():
    q0, q1, q2 = 'q0', 'q1', 'q2'

    circ0 = qf.Circuit()
    circ0 += qf.H(q0)
    circ0 += qf.X(q1)
    circ0 += qf.S_H(q0)
    circ0 += qf.XX(0.2, q0, q1)
    circ0 += qf.CAN(0.2, 0.1, 0.4, q0, q2)
    circ0 += qf.SWAP(q0, q1)

    ket1 = circ0.run()
    sim = QSimSimulator(circ0, translate=True)
    # print(sim._circuit)
    ket2 = sim.run()

    assert qf.states_close(ket1, ket2)
