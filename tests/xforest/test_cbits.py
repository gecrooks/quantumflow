
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import pytest
pytest.importorskip("pyquil")      # noqa: 402

from quantumflow import xforest as pq


def test_register():
    ro = pq.Register()
    assert ro.name == 'ro'
    assert str(ro) == "Register('ro', 'BIT')"


def test_register_ordered():
    assert pq.Register() == pq.Register('ro')
    assert pq.Register('a') < pq.Register('b')
    assert pq.Register('a') != pq.Register('b')
    assert pq.Register('c') != 'foobar'

    with pytest.raises(TypeError):
        pq.Register('c') < 'foobar'


def test_addr():
    c = pq.Register('c')
    c0 = c[0]
    assert c0.register.name == 'c'
    assert c0.key == 0
    assert c0.register.dtype == 'BIT'
    assert c0.dtype == 'BIT'

    assert str(c0) == 'c[0]'
    assert repr(c0) == "Register('c', 'BIT')[0]"


def test_addr_ordered():
    key = pq.Register('c')[0]
    d = dict({key: '1234'})
    assert d[key] == '1234'

    assert pq.Register('c')[0] == pq.Register('c')[0]
    assert pq.Register('c')[0] != pq.Register('c')[1]
    assert pq.Register('d')[0] != pq.Register('c')[0]

    assert pq.Register('c')[0] != 'foobar'
    assert pq.Register('c')[0] < pq.Register('c')[1]

    with pytest.raises(TypeError):
        pq.Register('c')[0] < 'foobar'
