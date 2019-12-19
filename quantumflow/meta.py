
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Meta
"""

import sys
import typing
import numpy as np
import networkx as nx
import cvxpy as cvx
import sympy

import quantumflow as qf
import quantumflow.backend as bk


def print_versions(file: typing.TextIO = None) -> None:
    """
    Print version strings of currently installed dependencies

     ``> python -m quantumflow.meta``

    Args:
        file: Output stream. Defaults to stdout.
    """
    print()
    print('** QuantumFlow dependencies (> python -m quantumflow.meta) **')
    print('quantumflow \t', qf.__version__, file=file)
    print('python      \t', sys.version[0:5], file=file)
    print('numpy       \t', np.__version__, file=file)
    print('networkx    \t', nx.__version__, file=file)
    print('cvxpy      \t', cvx.__version__, file=file)
    print('sympy      \t', sympy.__version__, file=file)

    print(bk.name, '   \t', bk.version, '\t(BACKEND)', file=file)

    # Optional packages
    try:
        import pyquil
        print('pyquil      \t', pyquil.__version__, '\t(Optional)', file=file)
    except ImportError:
        pass

    try:
        import cirq
        print('cirq        \t', cirq.__version__, ' \t(Optional)', file=file)
    except ImportError:
        pass

    try:
        import qiskit
        print('qiskit        \t', qiskit.__qiskit_version__['qiskit'],
              ' \t(Optional)', file=file)
        print('qiskit (terra)\t', qiskit.__version__, ' \t(Optional)',
              file=file)
    except ImportError:
        pass

    try:
        from mypy.version import __version__ as mypy_version
        print('mypy        \t', mypy_version, ' \t(Dev.)', file=file)
    except ImportError:
        pass

    print()
    print()


if __name__ == '__main__':
    print_versions()
