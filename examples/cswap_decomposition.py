#!/usr/bin/env python

# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
QuantumFlow Examples:
    Demonstration of decomposing a CSwap into CZ and 1-qubit gates
"""
import quantumflow as qf

circ0 = qf.Circuit([qf.CSwap(0, 1, 2)])

translators = [
    qf.translate_cswap_to_ccnot,
    qf.translate_ccnot_to_cnot,
    qf.translate_cnot_to_cz,
]
circ1 = qf.circuit_translate(circ0, translators)

assert circ1.size() == 33
print(qf.circuit_to_diagram(circ1))
