# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import pytest

pytest.importorskip("qutip")

from qutip.qip.operations import gate_sequence_product  # noqa: E402

import quantumflow as qf  # noqa: E402
from quantumflow import xqutip  # noqa: E402

# from qutip.qip.circuit import QubitCircuit


def test_circuit_to_qutip() -> None:
    q0, q1, q2 = 0, 1, 2

    circ0 = qf.Circuit()
    circ0 += qf.I(q0)
    circ0 += qf.Ph(0.1, q0)
    circ0 += qf.X(q0)
    circ0 += qf.Y(q1)

    circ0 += qf.Z(q0)
    circ0 += qf.S(q1)
    circ0 += qf.T(q2)

    circ0 += qf.H(q0)
    circ0 += qf.H(q1)
    circ0 += qf.H(q2)

    circ0 += qf.CNot(q0, q1)
    circ0 += qf.CNot(q1, q0)
    circ0 += qf.Swap(q0, q1)
    circ0 += qf.ISwap(q0, q1)

    circ0 += qf.CCNot(q0, q1, q2)
    circ0 += qf.CSwap(q0, q1, q2)

    circ0 == qf.I(q0)
    circ0 += qf.Rx(0.1, q0)
    circ0 += qf.Ry(0.2, q1)
    circ0 += qf.Rz(0.3, q2)
    circ0 += qf.V(q0)
    circ0 += qf.H(q1)
    circ0 += qf.CY(q0, q1)
    circ0 += qf.CZ(q0, q1)

    circ0 += qf.CS(q1, q2)
    circ0 += qf.CT(q0, q1)

    circ0 += qf.SqrtSwap(q0, q1)
    circ0 += qf.SqrtISwap(q0, q1)
    circ0 += qf.CCNot(q0, q1, q2)
    circ0 += qf.CSwap(q0, q1, q2)

    circ0 += qf.CPhase(0.1, q1, q2)

    # Not yet supported
    # circ0 += qf.B(q1, q2)
    # circ0 += qf.Swap(q1, q2) ** 0.1

    qbc = xqutip.circuit_to_qutip(circ0)
    U = gate_sequence_product(qbc.propagators())
    gate0 = qf.Unitary(U.full(), qubits=[0, 1, 2])
    assert qf.gates_close(gate0, circ0.asgate())

    circ1 = xqutip.qutip_to_circuit(qbc)

    assert qf.gates_close(circ0.asgate(), circ1.asgate())


def test_translate_to_qutip() -> None:
    circ0 = qf.Circuit()
    circ0 += qf.Can(0.1, 0.2, 0.3, 0, 1)
    qbc = xqutip.circuit_to_qutip(circ0, translate=True)
    U = gate_sequence_product(qbc.propagators())
    gate0 = qf.Unitary(U.full(), qubits=[0, 1])
    assert qf.gates_close(gate0, circ0.asgate())

    with pytest.raises(ValueError):
        xqutip.circuit_to_qutip(circ0, translate=False)

    circ1 = qf.Circuit()
    circ1 += qf.Can(0.1, 0.2, 0.3, "a", "b")
    with pytest.raises(ValueError):
        xqutip.circuit_to_qutip(circ1, translate=True)


# Not yet supported
# def test_swapalpha() -> None:
#     alpha = 0.1
#     qbc = QubitCircuit(2)
#     qbc.add_gate("SWAPalpha", targets=[0, 1], arg_value=alpha)
#     U = gate_sequence_product(qbc.propagators()).full()

#     gate0 = qf.Unitary(U, [0, 1])
#     gate1 = qf.Swap(0, 1) ** alpha
#     assert qf.gates_close(gate0, gate1)
#     assert qf.gates_close(gate0, qf.Exch(alpha / 2, 0, 1))

# Not yet supported
# def test_berkeley() -> None:
#     from qutip.qip.operations import berkeley
#     U = berkeley(2, [0, 1]).full()
#     gate0 = qf.Unitary(U, [0, 1])
#     gate1 = qf.B(0, 1)
#     assert qf.gates_close(gate0, gate1)


# import numpy as np
# def test_molmer_sorensen() -> None:
#     from qutip.qip.operations import molmer_sorensen
#     theta = 0.1
#     U = molmer_sorensen(theta, 2, [0, 1]).full()
#     gate0 = qf.Unitary(U, [0, 1])
#     gate1 = qf.XX(theta/np.pi, 0, 1)
#     assert qf.gates_close(gate0, gate1)


# fin
