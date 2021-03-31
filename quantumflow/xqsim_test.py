# Copyright 2020-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import pytest

pytest.importorskip("cirq")
pytest.importorskip("qsimcirq")


import quantumflow as qf  # noqa: E402
from quantumflow.xqsim import QSimSimulator  # noqa: E402


def test_qsim_simulator() -> None:
    q0, q1, q2 = "q0", "q1", "q2"

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
    circ0 += qf.Z(q1) ** 0.2
    circ0 += qf.X(q1) ** 0.2
    circ0 += qf.XPow(0.2, q0)
    circ0 += qf.YPow(0.2, q1)
    circ0 += qf.ZPow(0.5, q2)

    circ0 += qf.CZ(q0, q1)
    circ0 += qf.CNot(q0, q1)
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

    print("QF", ket1)
    print("QS", ket2)

    assert qf.states_close(ket1, ket2)
    assert qf.states_close(circ0.run(), sim.run())


def test_qsim_translate() -> None:
    q0, q1, q2 = "q0", "q1", "q2"

    circ0 = qf.Circuit()
    circ0 += qf.H(q0)
    circ0 += qf.X(q1)
    circ0 += qf.S_H(q0)
    circ0 += qf.XX(0.2, q0, q1)
    circ0 += qf.Can(0.2, 0.1, 0.4, q0, q2)
    circ0 += qf.Swap(q0, q1)

    ket1 = circ0.run()
    sim = QSimSimulator(circ0, translate=True)
    # print(sim._circuit)
    ket2 = sim.run()

    assert qf.states_close(ket1, ket2)
