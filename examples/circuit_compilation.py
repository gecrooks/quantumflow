#!/usr/bin/env python
"""
QuantumFlow: Examples of compiling circuits to native gates
"""

import quantumflow as qf

example_circuits = [
    ["3-qubit addition", qf.addition_circuit([0, 1, 2], [3, 4, 5], [6, 7])],
    ["7-qubit QFT", qf.Circuit(qf.QFTGate([0, 1, 2, 3, 4, 5, 6, 7]).decompose())],
]

for title, example in example_circuits:
    print()
    print(title)
    print(qf.circuit_to_diagram(example))
    print("Gate count:", example.size())

    print()
    print("Simplified circuit")
    circ = qf.compile_circuit(example)
    print(qf.circuit_to_diagram(circ, transpose=True))

    qf.circuit_to_image(circ).show()
    dagc = qf.DAGCircuit(circ)
    print("Gate depth", dagc.depth(local=False))
    print("Operation count", qf.count_operations(dagc))
