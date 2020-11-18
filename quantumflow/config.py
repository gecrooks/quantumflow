# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Package wide configuration
"""

import platform
import re
import sys
import typing

from .utils import importlib_metadata

__all__ = ["__version__", "about"]


package_name = "quantumflow"

try:
    __version__ = importlib_metadata.version(package_name)  # type: ignore
except Exception:  # pragma: no cover
    # package is not installed
    __version__ = "?.?.?"


# See https://numpy.org/doc/stable/reference/generated/numpy.allclose.html

RTOL = 1e-05
"""Default relative tolerance for numerical comparisons"""

ATOL = 1e-07
"""Default absolute tolerance for numerical comparisons"""


CIRCUIT_INDENT = 4
"""Number of spaces to indent when listing circuits"""


# --- Defaults for gate visualization as unicode text ---
# Used by visualizations.circuit_to_diagram()

TARGET = "X"
# TARGET = '⨁'              # "n-ary circled plus", U+2A01
SWAP_TARGET = "x"
# SWAP_TARGET = '×'         # Multiplication sign
XCTRL = "⊖"  # ⊖ 'circled minus'
NXCTRL = "⊕"  # ⊕ 'circled plus'
YCTRL = "⊘"  # ⊘ 'circled division slash'
NYCTRL = "⊗"  # ⊗ 'circled times'
CTRL = "●"
NCTRL = "○"
CONJ = "⁺"  # Unicode "superscript plus sign"
SQRT = "√"


def about(file: typing.TextIO = None) -> None:
    """Print information about the configuration

     ``> python -m quantumflow.about``

    Args:
        file: Output stream (Defaults to stdout)
    """
    name_width = 24
    versions = {}
    versions["platform"] = platform.platform(aliased=True)
    versions[package_name] = __version__
    versions["python"] = sys.version[0:5]

    for req in importlib_metadata.requires(package_name):  # type: ignore
        name = re.split("[; =><]", req)[0]
        try:
            versions[name] = importlib_metadata.version(name)  # type: ignore
        except Exception:  # pragma: no cover
            pass

    print(file=file)
    print("# Configuration (> python -m quantumflow.about)", file=file)
    for name, vers in versions.items():
        print(name.ljust(name_width), vers, file=file)
    print(file=file)


# fin
