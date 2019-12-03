

import quantumflow as qf
from quantumflow.cliffords import _clifford_gates, _clifford_circuits

import pytest


cliff_gates = [qf.Z(), qf.Y(), qf.X(), qf.V(), qf.V_H(), qf.S(), qf.S_H(),
               qf.Y()**0.5, qf.Y()**-0.5, qf.H()]
cliff_gates += list(_clifford_gates)
cliff_gates += [circ.asgate() for circ in _clifford_circuits]

cliff_gates_ids = [str(cg) for cg in cliff_gates]


@pytest.mark.parametrize("gate", cliff_gates, ids=cliff_gates_ids)
def test_clifford_from_gate(gate):
    cg = qf.Clifford.from_gate(gate)
    assert qf.gates_close(gate, cg)


def test_clifford_mul():
    for gate0 in cliff_gates:
        cg0 = qf.Clifford.from_gate(gate0)
        for gate1 in cliff_gates:
            cg1 = qf.Clifford.from_gate(gate1)

            cg2 = cg1 @ cg0
            assert isinstance(cg2, qf.Clifford)
            assert qf.gates_close(cg2, gate1 @ gate0)

            circ = qf.Circuit([gate0, gate1])
            assert qf.gates_close(cg2, circ.asgate())

            cg3 = qf.Circuit([cg0, cg1]).asgate()
            assert isinstance(cg3, qf.Clifford)
            assert cg2.index == cg3.index


def test_clifford_error():
    with pytest.raises(ValueError):
        qf.Clifford.from_gate(qf.SWAP())


@pytest.mark.parametrize("gate", cliff_gates, ids=cliff_gates_ids)
def test_clifford_adjoint(gate):
    cg0 = qf.Clifford.from_gate(gate)
    assert qf.gates_close(cg0.H.asgate(), gate.H)


@pytest.mark.parametrize("gate", cliff_gates, ids=cliff_gates_ids)
def test_clifford_dek(gate):
    cg0 = qf.Clifford.from_gate(gate)
    circ = cg0.ascircuit()
    assert qf.gates_close(gate, circ.asgate())
