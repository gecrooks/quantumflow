#!/usr/bin/env python

"""
QuantumFlow Examples:
    Demonstration of decomposing a CSWAP into CZ and 1-qubit gates
"""
import quantumflow as qf

circ0 = qf.Circuit([qf.CSWAP(0, 1, 2)])

translators = [qf.translate_cswap_to_ccnot,
               qf.translate_ccnot_to_cnot,
               qf.translate_cnot_to_cz]
circ1 = qf.translate_circuit(circ0, translators)

assert circ1.size() == 33
print(circ1)
qf.circuit_to_image(circ1).show()
