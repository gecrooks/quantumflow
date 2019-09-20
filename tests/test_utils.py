# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.utils
"""

from math import pi

import networkx as nx

import pytest

from quantumflow.utils import (
    invert_map, FrozenDict, bitlist_to_int, int_to_bitlist,
    to_graph6, from_graph6,
    spanning_tree_count, octagonal_tiling_graph, deprecated,
    cached_property,
    rationalize, symbolize)


def test_invert_dict():
    foo = {1: 7, 2: 8, 3: 9}
    bar = invert_map(foo)
    assert bar == {7: 1, 8: 2, 9: 3}

    foo = {1: 7, 2: 8, 3: 7}
    bar = invert_map(foo, one_to_one=False)
    assert bar == {7: set([1, 3]), 8: set([2])}


def test_frozen_dict():
    f0 = FrozenDict({'a': 1, 'b': 2})
    assert str(f0) == "FrozenDict({'a': 1, 'b': 2})"

    hash(f0)

    f1 = f0.copy()
    assert f0 == f1

    f2 = f0.copy(a=0, c=3)
    assert f2['a'] == 0

    assert 'c' in f2
    assert len(f2) == 3

    assert list(f2.keys()) == ['a', 'b', 'c']


def test_frozen_dict_generic():
    # Test code should pass mypy as well as pytest

    f0: FrozenDict[str, int] = FrozenDict({'a': 1, 'b': 2})

    k1 = 3
    k1 = f0['b']

    f1: FrozenDict[str, str] = FrozenDict({'a': 1, 'b': 2})  # noqa
    k1: int = f0['b']                                        # noqa


def test_deprecated():
    class Something:
        @deprecated
        def some_thing(self):
            pass

    obj = Something()

    with pytest.deprecated_call():
        obj.some_thing()


def test_cached_property():

    class thing():
        def __init__(self, value):
            self.value = value

        @cached_property
        def plus1(self):
            return self.value+1

        @cached_property
        def plus2(self):
            return self.value+2

    two = thing(2)
    assert two.plus1 == 2+1
    assert two.plus1 == 2+1
    assert two.plus2 == 2+2
    assert two.plus1 == 2+1

    ten = thing(10)
    assert ten.plus1 == 10+1
    assert ten.plus1 == 10+1
    assert ten.plus2 == 10+2
    assert ten.plus2 == 10+2

    assert two.plus1 == 2+1
    assert two.plus2 == 2+2


def test_bitlist_to_int():
    assert bitlist_to_int([1, 0, 0]) == 4


def test_int_to_bitlist():
    assert int_to_bitlist(4, 4) == [0, 1, 0, 0]
    assert int_to_bitlist(4) == [1, 0, 0]


def test_graph6():
    graph0 = nx.random_regular_graph(4, 10)
    g0 = to_graph6(graph0)
    graph1 = from_graph6(g0)
    g1 = to_graph6(graph1)

    assert g0 == g1


def test_spanning_tree_count():
    grp = nx.grid_2d_graph(3, 3)
    count = spanning_tree_count(grp)
    assert count == 192


def test_octagonal_tiling_graph():
    grp = octagonal_tiling_graph(4, 4)
    assert len(grp) == 128


def test_rationalize():
    flt = 1/12
    rationalize(flt)

    with pytest.raises(ValueError):
        rationalize(pi)


def test_symbolize():
    s = symbolize(1.0)
    assert str(s) == '1'

    with pytest.raises(ValueError):
        s = symbolize(1.13434538345)

    s = symbolize(pi * 123)
    assert str(s) == '123*pi'

    s = symbolize(pi / 64)
    assert str(s) == 'pi/64'

    s = symbolize(pi * 3 / 64)
    assert str(s) == '3*pi/64'

    s = symbolize(pi * 8 / 64)
    assert str(s) == 'pi/8'

    s = symbolize(-pi * 3 / 8)
    assert str(s) == '-3*pi/8'

    s = symbolize(5/8)
    assert str(s) == '5/8'

# fin
