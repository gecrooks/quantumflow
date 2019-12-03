#!/usr/bin/env python


"""QuantumFlow: Validate and display various gate translations."""

# Note: Used not only as an illustrative example, but as part of
# the testing suite, and as a source for circuit diagrams in docstrings.

from itertools import zip_longest

import numpy as np
# from sympy import Symbol

import quantumflow as qf
from quantumflow.translate import translation_source_gate

from quantumflow.visualization import kwarg_to_symbol as syms
# TODO: Redundant with  visualizations.kwarg_to_symbol
# Pretty print gate arguments
# syms = {
#     'alpha':    Symbol('α'),
#     'lam':      Symbol('λ'),
#     'nx':       Symbol('nx'),
#     'ny':       Symbol('ny'),
#     'nz':       Symbol('nz'),
#     'p':        Symbol('p'),
#     'phi':      Symbol('φ'),
#     't':        Symbol('t'),
#     't0':       Symbol('t0'),
#     't1':       Symbol('t1'),
#     't2':       Symbol('t2'),
#     'theta':    Symbol('θ'),
#     'tx':       Symbol('tx'),
#     'ty':       Symbol('ty'),
#     'tz':       Symbol('tz'),
#     's':        Symbol('s'),
#     'b':        Symbol('b'),
#     'c':        Symbol('c'),
#     }


def _check_circuit_translations():
    # Concrete values with which to test that circuits are
    # functionally identical.
    concrete = {name: np.random.uniform(-4, 4) for name in syms.values()}

    for name, trans in qf.TRANSLATORS.items():
        gatet = translation_source_gate(trans)
        args = [syms[a] for a in gatet.args()]

        gate = gatet(*args)
        circ0 = qf.Circuit([gate])
        circ1 = qf.Circuit(trans(gate))

        # FIXME: Fails if no doc string
        annote = trans.__doc__.splitlines()[0]

        _print_circuit_identity(annote, circ0, circ1)

        circ0f = circ0.resolve(concrete)
        circ1f = circ1.resolve(concrete)
        assert qf.gates_close(circ0f.asgate(), circ1f.asgate())


# TODO: Fixup and move to visualization?
def _print_circuit_identity(name, circ0, circ1,
                            min_col_width=0,
                            col_sep=5,
                            left_margin=8):

    print()
    print("", name)
    print()

    circ0 = qf.Circuit(circ0)
    circ1 = qf.Circuit(circ1)

    gates0 = qf.circuit_to_diagram(circ0, qubit_labels=False).splitlines()
    gates1 = qf.circuit_to_diagram(circ1, qubit_labels=False).splitlines()

    for gate0, gate1 in zip_longest(gates0, gates1, fillvalue=""):
        line = (' '*col_sep).join([gate0.ljust(min_col_width),
                                   gate1.ljust(min_col_width)])
        line = (' '*left_margin) + line
        line = line.rstrip()
        print(line)

    print()
    print()


if __name__ == "__main__":
    print()
    print("# Validate and display various gate translations  "
          "(up to global phase)")
    print()
    _check_circuit_translations()
