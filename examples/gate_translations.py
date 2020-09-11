#!/usr/bin/env python

# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""QuantumFlow: Validate and display various gate translations."""

# Note: Used not only as an illustrative example, but as part of
# the testing suite, and as a source for circuit diagrams in docstrings.

from itertools import zip_longest

import numpy as np

import quantumflow as qf
from quantumflow.translate import translation_source_gate
from quantumflow.visualization import kwarg_to_symbol as syms


def _check_circuit_translations():
    # Concrete values with which to test that circuits are
    # functionally identical.
    concrete = {name: np.random.uniform(-4, 4) for name in syms.values()}

    for name, trans in qf.TRANSLATORS.items():
        gatet = translation_source_gate(trans)
        args = [syms[a] for a in gatet.cv_args]

        gate = gatet(*args, *range(gatet.cv_qubit_nb))
        circ0 = qf.Circuit([gate])
        circ1 = qf.Circuit(trans(gate))

        # FIXME: Fails if no doc string
        annote = trans.__doc__.splitlines()[0]

        _print_circuit_identity(annote, circ0, circ1)

        circ0f = circ0.resolve(concrete)
        circ1f = circ1.resolve(concrete)
        assert qf.gates_close(circ0f.asgate(), circ1f.asgate())


def _print_circuit_identity(
    name, circ0, circ1, min_col_width=0, col_sep=5, left_margin=8
):

    print()
    print("", name)
    print()

    circ0 = qf.Circuit(circ0)
    circ1 = qf.Circuit(circ1)

    gates0 = qf.circuit_to_diagram(circ0, qubit_labels=False).splitlines()
    gates1 = qf.circuit_to_diagram(circ1, qubit_labels=False).splitlines()

    for gate0, gate1 in zip_longest(gates0, gates1, fillvalue=""):
        line = (" " * col_sep).join(
            [gate0.ljust(min_col_width), gate1.ljust(min_col_width)]
        )
        line = (" " * left_margin) + line
        line = line.rstrip()
        print(line)

    print()
    print()


if __name__ == "__main__":
    print()
    print("# Validate and display various gate translations  " "(up to global phase)")
    print()
    _check_circuit_translations()
