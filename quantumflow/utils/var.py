# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import cmath
from numbers import Complex
from typing import Any, Union

from sympy import Expr

from .future import TypeAlias

__all__ = "Variable", "ComplexVariable", "is_symbolic", "is_complex_variable", "almost_zero_variable"

Variable: TypeAlias = Union[float, Expr]
"""Type for parameters. Either a float, sympy.Symbol or sympy.Expr"""


ComplexVariable: TypeAlias = Union[Complex, Expr]
"""Type for complex parameters. Either a complex number, sympy.Symbol or sympy.Expr"""


def is_symbolic(x: Any) -> bool:
    return isinstance(x, Expr)


def is_complex_variable(x: Any) -> bool:
    return isinstance(x, Complex) or isinstance(x, Expr)


def almost_zero_variable(x: ComplexVariable) -> bool:
    if isinstance(x, Complex) and cmath.isclose(x, 0.0):
        return True
    if is_symbolic(x):
        return x.is_zero
    return False
