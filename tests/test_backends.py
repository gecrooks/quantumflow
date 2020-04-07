
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.backend
"""

import numpy as np

import pytest

from . import ALMOST_ZERO

from quantumflow.backends import BACKENDS

# Only test ctf backend if ctf is installed.
try:
    import ctf                  # noqa: F401
except ImportError:
    del BACKENDS['ctf']

backends = list(backend() for backend in BACKENDS.values())


@pytest.mark.parametrize('bk', backends)
def test_astensor(bk):
    t = bk.astensor(1)
    assert isinstance(t, bk.tensor_type)

    t = bk.astensor([1, 2])


@pytest.mark.parametrize('bk', backends)
def test_size(bk):
    """size(tensor) should return the number of elements"""
    t = bk.astensor([1, 2])
    assert bk.size(t) == 2

    t = bk.astensor(np.zeros(shape=[2, 2, 2]))
    assert bk.size(t) == 8

    t = bk.astensor([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    assert bk.size(t) == 16


@pytest.mark.parametrize('bk', backends)
def test_ndim(bk):
    t = bk.astensor([1, 2])
    assert bk.ndim(t) == 1

    t = bk.astensor(np.zeros(shape=[2, 2, 2]))
    assert bk.ndim(t) == 3


@pytest.mark.parametrize('bk', backends)
def test_inner(bk):
    v0 = np.random.normal(size=[2, 2, 2, 2]) \
        + 1.0j * np.random.normal(size=[2, 2, 2, 2])
    v1 = np.random.normal(size=[2, 2, 2, 2]) \
        + 1.0j * np.random.normal(size=[2, 2, 2, 2])
    res = np.vdot(v0, v1)

    bkres = bk.evaluate(bk.inner(bk.astensor(v0), bk.astensor(v1)))

    assert np.abs(res-bkres) == ALMOST_ZERO


@pytest.mark.parametrize('bk', backends)
def test_outer(bk):
    s0 = np.random.normal(size=[2, 2]) + 1.0j * np.random.normal(size=[2, 2])
    s1 = np.random.normal(size=[2, 2, 2]) \
        + 1.0j * np.random.normal(size=[2, 2, 2])

    res = bk.astensorproduct(bk.outer(bk.astensor(s0), bk.astensor(s1)))
    assert bk.ndim(res) == 5

    res2 = np.outer(s0, s1).reshape([2]*5)
    assert np.allclose(bk.evaluate(res), res2)


@pytest.mark.parametrize('bk', backends)
def test_absolute(bk):
    t = bk.astensor([-2.25 + 4.75j, -3.25 + 5.75j])
    t = bk.absolute(t)

    assert np.allclose(bk.evaluate(t), [5.25594902, 6.60492229])

    t = bk.astensor([-2.25 + 4.75j])
    t = bk.absolute(t)
    # print(bk.evaluate(t))
    assert np.allclose(bk.evaluate(t), [5.25594902])


@pytest.mark.parametrize('bk', backends)
def test_random_seed(bk):
    # Also tested indirectly by test_config::test_seed
    # But that doesn't get captured by coverage tool
    bk.set_random_seed(42)
    bk.seed(12)


@pytest.mark.parametrize('bk', backends)
def test_real_imag(bk):
    tensor = bk.astensor([1.0 + 2.0j, 0.5 - 0.2j])
    t = bk.real(tensor)
    t = bk.evaluate(t)
    assert np.allclose(bk.evaluate(bk.real(tensor)), [1.0, 0.5])
    assert np.allclose(bk.evaluate(bk.imag(tensor)), [2.0, -0.2])


@pytest.mark.parametrize('bk', backends)
def test_diag(bk):
    t = bk.astensor([[0., 0., 0., 6.],
                     [0., 1., 0., 0.],
                     [0., 0., 2., 1.],
                     [4., 0., 1., 3.]])
    d = bk.diag(t)
    a = bk.sum(d)
    assert bk.numpy(a) - 6. == ALMOST_ZERO

    a = bk.reduce_sum(d)


@pytest.mark.parametrize('bk', backends)
def test_evaluate(bk):
    t = [1, 3]
    bk.evaluate(bk.astensor(t))


@pytest.mark.parametrize('bk', backends)
def test_trace(bk):
    tensor = bk.astensor(np.asarray([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 2.7, 1],
                                     [0, 0, 1, 0.3j]]))
    tr = bk.evaluate(bk.trace(tensor))

    assert tr - (2.7+0.3j) == ALMOST_ZERO


@pytest.mark.parametrize('bk', backends)
def test_productdiag(bk):
    t = bk.astensorproduct([[0., 0., 0., 6.],
                            [0., 1., 0., 0.],
                            [0., 0., 2., 1.],
                            [4., 0., 1., 3.]])
    assert np.allclose(bk.evaluate(bk.productdiag(t)), [[0, 1], [2, 3]])


@pytest.mark.parametrize('bk', backends)
def test_cast(bk):

    f = bk.numpy(bk.ccast(1.2))
    assert f - 1.2 == ALMOST_ZERO

    c = bk.numpy(bk.ccast(1.0j))
    assert c - 1.0j == ALMOST_ZERO


@pytest.mark.parametrize('bk', backends)
def test_math(bk):
    f = 0.34

    assert np.exp(f*1.0j) - bk.numpy(bk.cis(f)) == ALMOST_ZERO

    assert np.arccos(f) - bk.numpy(bk.arccos(f)) == ALMOST_ZERO
    assert np.arcsin(f) - bk.numpy(bk.arcsin(f)) == ALMOST_ZERO
    assert np.arctan(f) - bk.numpy(bk.arctan(f)) == ALMOST_ZERO
    assert np.cos(f) - bk.numpy(bk.cos(f)) == ALMOST_ZERO
    assert np.exp(f) - bk.numpy(bk.exp(f)) == ALMOST_ZERO
    assert np.sin(f) - bk.numpy(bk.sin(f)) == ALMOST_ZERO
    assert np.sqrt(f) - bk.numpy(bk.sqrt(f)) == ALMOST_ZERO
    assert np.tan(f) - bk.numpy(bk.tan(f)) == ALMOST_ZERO
    assert np.absolute(f) - bk.numpy(bk.absolute(f)) == ALMOST_ZERO

    assert np.arctan2(f, 2*f) - bk.numpy(bk.arctan2(f, 2*f)) == ALMOST_ZERO

    assert np.real(f + 2.0j*f) - bk.numpy(bk.real(f + 2.0j*f)) == ALMOST_ZERO
    assert np.imag(f + 2.0j*f) - bk.numpy(bk.imag(f + 2.0j*f)) == ALMOST_ZERO
    assert np.conj(f + 2.0j*f) - bk.numpy(bk.conj(f + 2.0j*f)) == ALMOST_ZERO


@pytest.mark.parametrize('bk', backends)
def test_symmath(bk):

    import sympy
    nf = 0.34
    f = sympy.Symbol('f')
    subs = {f: nf}

    assert np.cos(nf) - bk.cos(f).evalf(subs=subs) == ALMOST_ZERO

    assert np.arccos(nf) - bk.arccos(f).evalf(subs=subs) == ALMOST_ZERO
    assert np.arcsin(nf) - bk.arcsin(f).evalf(subs=subs) == ALMOST_ZERO
    assert np.arctan(nf) - bk.arctan(f).evalf(subs=subs) == ALMOST_ZERO
    assert np.cos(nf) - bk.cos(f).evalf(subs=subs) == ALMOST_ZERO
    assert np.exp(nf) - bk.exp(f).evalf(subs=subs) == ALMOST_ZERO
    assert np.sin(nf) - bk.sin(f).evalf(subs=subs) == ALMOST_ZERO
    assert np.sqrt(nf) - bk.sqrt(f).evalf(subs=subs) == ALMOST_ZERO
    assert np.tan(nf) - bk.tan(f).evalf(subs=subs) == ALMOST_ZERO

    assert np.arctan2(nf, 2*nf) - bk.arctan2(f, 2*f).evalf(subs=subs) \
        == ALMOST_ZERO

    assert np.real(nf + 2.0j*nf) - bk.real(f + 2.0j*f).evalf(subs=subs) \
        == ALMOST_ZERO
    assert np.imag(nf + 2.0j*nf) - bk.imag(f + 2.0j*f).evalf(subs=subs) \
        == ALMOST_ZERO


@pytest.mark.parametrize('bk', backends)
def test_sign(bk):
    assert bk.sign(-40) == -1
    assert bk.sign(40) == 1
    assert bk.sign(0) == 0

    import sympy
    f = sympy.Symbol('f')
    assert bk.sign(f).evalf(subs={'f': -10}) == -1


@pytest.mark.parametrize('bk', backends)
def test_minimum(bk):
    t1 = bk.astensor([2, 3, 4, 5, 1], dtype=bk.float_type)
    t2 = bk.astensor([1, 2, 3, 4, 5], dtype=bk.float_type)
    assert bk.sum(bk.minimum(t1, t2)) == 11.0


@pytest.mark.parametrize('bk', backends)
def test_transpose(bk):
    tensor = bk.astensor(np.asarray([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 2.7, 0],
                                     [0, 0, 5.0, 0.3j]]))
    tensor = bk.numpy(bk.transpose(tensor))
    assert tensor[2, 3] == 5.0


@pytest.mark.parametrize('bk', backends)
def test_matmul(bk):
    a0 = np.asarray([[1, 2], [3, 4]])
    a1 = np.asarray([[0.3, 2], [3j, 4]])

    a2 = a0 @ a1

    t0 = bk.astensor(a0)
    t1 = bk.astensor(a1)
    t2 = bk.matmul(t0, t1)

    a3 = bk.numpy(t2)

    assert np.allclose(a2, a3)


@pytest.mark.parametrize('bk', backends)
def test_copy(bk):
    t0 = bk.astensor(np.asarray([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 2.7, 0],
                                 [0, 0, 5.0, 0.3j]]))

    t1 = bk.copy(t0)
    assert np.allclose(bk.numpy(t0), bk.numpy(t1))


@pytest.mark.parametrize('bk', backends)
def test_numpy(bk):
    a0 = np.asarray([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 2.7, 0],
                     [0, 0, 5.0, 0.3j]])
    t0 = bk.astensor(a0)
    t0 = bk.astensor(t0)

    assert np.allclose(bk.numpy(a0), bk.numpy(t0))


@pytest.mark.parametrize('bk', backends)
def test_tensormul(bk):
    a0 = np.asarray([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 2.7, 0],
                     [0, 0, 5.0, 0.3j]])
    t0 = bk.astensorproduct(a0)
    t1 = bk.astensorproduct(a0)

    bk.tensormul(t0, t1, [0, 1])

# fin
