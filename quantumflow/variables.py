# Copyright 2019-, Gavin E. Crooks and the QuantumFlow contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# TESTME
# DOCME

"""
QuantumFlow: Variables
"""


from typing import Union
from numbers import Number

import numpy as np

import sympy
from sympy import pi as Pi

__all__ = [
    'Variable',
    'variable_almost_zero',
    'variables_close',
    'variable_is_symbolic',
    'variable_is_number',
    'Pi'
    ]

# Patch sympy.Expr so that Symbol's work with numpy
# This works because numpy delegates many functions to
# the corresponding method defined on objects, if such a method
# exists.
# DOCME
# TESTME
# TODO: What other functions should be patched?
# TODO: Does this work with other backends? JAX?
# sympy.Expr.arccsc = lambda self: sympy.acsc(self)
# sympy.Expr.arccos = lambda self: sympy.acos(self)
# sympy.Expr.arccot = lambda self: sympy.acot(self)
# sympy.Expr.arccsc = lambda self: sympy.acsc(self)
# sympy.Expr.arcsec = lambda self, x: sympy.asec(self, x)
# sympy.Expr.arcsin = lambda self, x: sympy.asin(self, x)
# sympy.Expr.arctan = lambda self, x: sympy.atan(self, x)
# sympy.Expr.arctan2 = lambda self, x: sympy.atan2(self, x)
# sympy.Expr.cos = lambda self: sympy.cos(self)
# sympy.Expr.cot = lambda self: sympy.cot(self)
# sympy.Expr.exp = lambda self: sympy.exp(self)
# sympy.Expr.sec = lambda self: sympy.sec(self)
# sympy.Expr.sin = lambda self: sympy.sin(self)
# sympy.Expr.sqrt = lambda self: sympy.sqrt(self)
# sympy.Expr.tan = lambda self: sympy.tan(self)


Variable = Union[sympy.Expr, float]
"""Type for parameters. Either a float, sympy.Symbol or sympy.Expr"""


# DOCME TESTME
# FIXME: Replace with variables_close?
def variable_almost_zero(var: Variable) -> bool:
    # if variable_is_number(var) and np.isclose(var, 0.0):
    #     return True
    # return False
    if var == sympy.S.Zero:
        return True
    return variables_close(var, 0.0)


# DOCME TESTME
def variables_close(var0: Variable, var1: Variable) -> bool:
    if variable_is_number(var0) and variable_is_number(var1):
        return np.isclose(var0, var1)
    if variable_is_symbolic(var0) and variable_is_symbolic(var1):
        return var0 == var1     # pragma: no cover  # TESTME
    return False


# DOCME TESTME
def variable_is_symbolic(var: Variable) -> bool:
    return isinstance(var, sympy.Expr)


# DOCME TESTME
def variable_is_number(var: Variable) -> bool:
    return isinstance(var, Number)
