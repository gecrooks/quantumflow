#!/usr/bin/env python
"""
QuantumFlow: Examples of compiling circuits to native gates
"""

import quantumflow as qf


example_circuits = [
    ['3-qubit addition', qf.addition_circuit([0, 1, 2], [3, 4, 5], [6,  7])],
    ['7-qubit QFT', qf.qft_circuit([0, 1, 2, 3, 4, 5, 6, 7])]
    ]

for title, example in example_circuits:
    qf.circuit_to_image(example).show()

    circ = qf.compile_circuit(example)

    print()
    print(title)
    print('Gate count:', circ.size())
    qf.circuit_to_image(circ).show()
    dagc = qf.DAGCircuit(circ)
    print('Gate depth', dagc.depth(local=False))
    print('Operation count', qf.count_operations(dagc))
