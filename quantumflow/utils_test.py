# Copyright 2020-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import networkx as nx
import numpy as np
import pytest

from quantumflow import utils


def test_cached_property() -> None:
    class Thing:
        def __init__(self, value: int) -> None:
            self.value = value

        @utils.cached_property
        def plus1(self) -> int:
            return self.value + 1

        @utils.cached_property
        def plus2(self) -> int:
            return self.value + 2

    two = Thing(2)
    assert two.plus1 == 2 + 1
    assert two.plus1 == 2 + 1
    assert two.plus2 == 2 + 2
    assert two.plus1 == 2 + 1

    ten = Thing(10)
    assert ten.plus1 == 10 + 1
    assert ten.plus1 == 10 + 1
    assert ten.plus2 == 10 + 2
    assert ten.plus2 == 10 + 2

    assert two.plus1 == 2 + 1
    assert two.plus2 == 2 + 2


def test_deprecated() -> None:
    class Something:
        @utils.deprecated
        def some_thing(self) -> None:
            pass

    obj = Something()

    with pytest.deprecated_call():
        obj.some_thing()


def test_invert_dict() -> None:
    foo = {1: 7, 2: 8, 3: 9}
    bar = utils.invert_map(foo)
    assert bar == {7: 1, 8: 2, 9: 3}

    foo = {1: 7, 2: 8, 3: 7}
    bar = utils.invert_map(foo, one_to_one=False)
    assert bar == {7: set([1, 3]), 8: set([2])}


def test_frozen_dict() -> None:
    f0: utils.FrozenDict[str, int] = utils.FrozenDict({"a": 1, "b": 2})
    assert str(f0) == "FrozenDict({'a': 1, 'b': 2})"

    hash(f0)

    f1 = f0.copy()
    assert f0 == f1

    f2 = f0.update(a=0, c=3)
    assert f2["a"] == 0

    assert "c" in f2
    assert len(f2) == 3

    assert list(f2.keys()) == ["a", "b", "c"]


def test_frozen_dict_generic() -> None:
    f0: utils.FrozenDict[str, int] = utils.FrozenDict({"a": 1, "b": 2})

    k1 = f0["b"]  # noqa

    f1: utils.FrozenDict[str, str] = utils.FrozenDict({"a": 1, "b": 2})  # noqa
    k2: int = f0["b"]  # noqa


def test_bitlist_to_int() -> None:
    assert utils.bitlist_to_int([1, 0, 0]) == 4


def test_int_to_bitlist() -> None:
    assert utils.int_to_bitlist(4, 4) == [0, 1, 0, 0]
    assert utils.int_to_bitlist(4) == [1, 0, 0]


def test_rationalize() -> None:
    flt = 1 / 12
    utils.rationalize(flt)

    with pytest.raises(ValueError):
        utils.rationalize(np.pi)


def test_graph6() -> None:
    graph0 = nx.random_regular_graph(4, 10)
    g0 = utils.to_graph6(graph0)
    graph1 = utils.from_graph6(g0)
    g1 = utils.to_graph6(graph1)

    assert g0 == g1


def test_spanning_tree_count() -> None:
    grp = nx.grid_2d_graph(3, 3)
    count = utils.spanning_tree_count(grp)
    assert count == 192


def test_octagonal_tiling_graph() -> None:
    grp = utils.octagonal_tiling_graph(4, 4)
    assert len(grp) == 128


def test_truncated_grid_2d_graph() -> None:
    G = utils.truncated_grid_2d_graph(12, 11)
    assert len(G) == 72

    G = utils.truncated_grid_2d_graph(8, 8, 2)
    assert len(G) == 8 * 8 - 4 * 3


def test_multi_slice() -> None:
    ms = utils.multi_slice([3, 0], [1, 2])
    assert ms == (2, slice(None, None, None), slice(None, None, None), 1, Ellipsis)


def test_almost_integer() -> None:
    assert utils.almost_integer(1)
    assert utils.almost_integer(1.0)
    assert utils.almost_integer(11239847819349871423)
    assert utils.almost_integer(-4)
    assert utils.almost_integer(-4 + 0j)
    assert utils.almost_integer(1.0000000000000001)
    assert not utils.almost_integer(1.0000001)
    assert not utils.almost_integer(1 + 1j)


# fin
