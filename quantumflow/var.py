# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# DOCME

from typing import Mapping, Union

import numpy as np
import sympy

from . import utils
from .config import ATOL, RTOL

__all__ = ("Variable", "PI")


Symbol = sympy.Symbol
"""Class for symbols in symbolic expressions"""


Variable = Union[float, sympy.Expr]
"""Type for parameters. Either a float, sympy.Symbol or sympy.Expr"""


PI = sympy.pi
"""Symbolic constant pi"""


def is_symbolic(x: Variable) -> bool:
    """Returns true if a symbolic expression"""
    return isinstance(x, sympy.Expr)


def isclose(x: Variable, y: Variable, atol: float = ATOL, rtol: float = RTOL) -> bool:
    """Compares two variables.

    Returns: True if variables are almost identical concrete numbers,
        of if they are the same symbolic expression, else False.
    """
    if not is_symbolic(x) and not is_symbolic(y):
        return bool(np.isclose(x, y, atol=atol, rtol=rtol))
    if is_symbolic(x) and is_symbolic(y):
        return x == y
    return False


# DOCME # Testme
def almost_zero(x: Variable, atol: float = ATOL) -> bool:
    """Is the variable symbolically zero, or numerically almost zero."""
    if x == sympy.S.Zero:
        return True
    return isclose(x, 0.0, atol=atol)


def asfloat(x: Variable, subs: Mapping[str, float] = None) -> float:
    """Convert a variable to a float"""
    if is_symbolic(x) and subs:
        x = x.evalf(subs=subs)  # type: ignore
    return float(x)


def asexpression(flt: float) -> sympy.Expr:
    """Attempt to convert a real number into a simpler symbolic
    representation.

    Returns:
        A sympy Symbol. (Convert to string with str(sym) or to latex with
            sympy.latex(sym)
    Raises:
        ValueError:     If cannot simplify float
    """
    try:
        ratio = utils.rationalize(flt)
        res = sympy.simplify(ratio)
    except ValueError:
        try:
            ratio = utils.rationalize(flt / np.pi)
            res = sympy.simplify(ratio) * sympy.pi
        except ValueError:
            ratio = utils.rationalize(flt * np.pi)
            res = sympy.simplify(ratio) / sympy.pi
    return res


# The following math functions act on Variables, and return symbolic expression if
# the variable is symbolic, or numerical values if the variable is a number


def arccos(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.acos(x)
    return np.arccos(x)


def arcsin(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.asin(x)
    return np.arcsin(x)


def arctan(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.atan(x)
    return np.arctan(x)


def arctan2(x1: Variable, x2: Variable) -> Variable:
    if is_symbolic(x1) or is_symbolic(x2):
        return sympy.atan2(x1, x2)
    return np.arctan2(x1, x2)


def cos(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.cos(x)
    return np.cos(x)


def exp(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.exp(x)
    return np.exp(x)


def sign(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.sign(x)
    return np.sign(x)


def sin(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.sin(x)
    return np.sin(x)


def sqrt(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.sqrt(x)
    return np.sqrt(x)


def tan(x: Variable) -> Variable:
    if is_symbolic(x):
        return sympy.tan(x)
    return np.tan(x)


# fin
