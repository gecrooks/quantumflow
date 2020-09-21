#!/usr/bin/env python

# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Example visualizations of Circuits using LaTeX.
Requires external dependencies `pdflatex` and `poppler`.
"""

import shutil
import sys

import quantumflow as qf

if shutil.which("pdflatex") is None:
    print("Failed: External dependency `pdflatex` not found.")
    sys.exit()

if shutil.which("pdftocairo") is None:
    print("Failed: External dependency `pdftocairo` not found. " "Install `poppler`.")
    sys.exit()

# Display Bell state preparation
qf.circuit_to_image(qf.ghz_circuit([0, 1])).show()

# 4-qubit GHZ state preparation
qf.circuit_to_image(qf.ghz_circuit([0, 1, 2, 3])).show()

# 4-qubit ripple add
circ = qf.addition_circuit(
    ["a[0]", "a[1]", "a[2]", "a[3]"], ["b[0]", "b[1]", "b[2]", "b[3]"], ["cin", "cout"]
)
order = ["cin", "a[0]", "b[0]", "a[1]", "b[1]", "a[2]", "b[2]", "a[3]", "b[3]", "cout"]
latex = qf.circuit_to_latex(circ, order)
img = qf.latex_to_image(latex)
img.show()
