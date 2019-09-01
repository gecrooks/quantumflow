

import quantumflow as qf


def test_compile():
    circ0 = qf.addition_circuit([0], [1], [2, 3])
    circ1 = qf.compile_circuit(circ0)
    assert qf.circuits_close(circ0, circ1)
    # print(qf.circuit_diagram(circ1, transpose=False))
    assert circ1.size() == 76

    dagc = qf.DAGCircuit(circ1)
    assert dagc.depth(local=False) == 16
    counts = qf.count_operations(dagc)
    assert counts[qf.TZ] == 27
    assert counts[qf.TX] == 32
    assert counts[qf.CZ] == 17
