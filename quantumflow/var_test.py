# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np
import pytest

from quantumflow import var

funcnames = ["arccos", "arcsin", "arctan", "cos", "exp", "sign", "sin", "sqrt", "tan"]


@pytest.mark.parametrize("funcname", funcnames)
def test_scalar_functions(funcname: str) -> None:
    x = var.Symbol("x")
    nx = 0.76  # Gives real answer for all functions
    subs = {"x": nx}

    symfn = getattr(var, funcname)
    npfn = getattr(np, funcname)

    assert np.isclose(var.asfloat(symfn(x), subs), npfn(nx))
    assert np.isclose(var.asfloat(symfn(nx), subs), npfn(nx))


def test_scalar_arctan2() -> None:
    x = var.Symbol("x")
    nx = 0.76
    y = var.Symbol("y")
    ny = -0.5
    subs = {"x": nx, "y": ny}

    assert np.isclose(var.asfloat(var.arctan2(x, y), subs), np.arctan2(nx, ny))
    assert np.isclose(var.asfloat(var.arctan2(nx, ny), subs), np.arctan2(nx, ny))


def test_almost_zero() -> None:
    assert var.almost_zero(0)
    assert var.almost_zero(0.0)
    assert var.almost_zero(0.000000000000000001)
    assert var.almost_zero(var.Symbol("x").evalf(subs={"x": 0}))


def test_isclose() -> None:
    assert var.isclose(1.0, 1.0000000001)
    assert not var.isclose(0.0, 0.0000000001, atol=0.000000000001)
    assert var.isclose(var.Symbol("x"), var.Symbol("x"))
    assert not var.isclose(var.Symbol("x"), 1.0)


def test_asexpression() -> None:
    s = var.asexpression(1.0)
    assert str(s) == "1"

    with pytest.raises(ValueError):
        _ = var.asexpression(1.13434538345)

    s = var.asexpression(np.pi * 123)
    assert str(s) == "123*pi"

    s = var.asexpression(np.pi / 64)
    assert str(s) == "pi/64"

    s = var.asexpression(np.pi * 3 / 64)
    assert str(s) == "3*pi/64"

    s = var.asexpression(np.pi * 8 / 64)
    assert str(s) == "pi/8"

    s = var.asexpression(-np.pi * 3 / 8)
    assert str(s) == "-3*pi/8"

    s = var.asexpression(5 / 8)
    assert str(s) == "5/8"

    s = var.asexpression(2 / np.pi)
    assert str(s) == "2/pi"


# fin
