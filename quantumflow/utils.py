
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: utilities

Useful routines not necessarily intended to be part of the public API.
"""

from typing import Any, Sequence, Callable, Set, Tuple, Iterator, Dict
from typing import Optional, Mapping, TypeVar, List

import warnings
import functools
from fractions import Fraction
# from collections import deque

import numpy as np
import networkx as nx

import sympy


# from scipy.linalg import fractional_matrix_power as matpow # Matrix power
# from scipy.linalg import sqrtm as matsqrt   # Matrix square root

__all__ = [
    'multi_slice',
    'invert_map',
    'FrozenDict',
    'bitlist_to_int',
    'int_to_bitlist',
    'deprecated',
    'cached_property',
    'from_graph6',
    'to_graph6',
    'spanning_tree_count',
    'octagonal_tiling_graph',
    'rationalize',
    'symbolize',
    'complex_ginibre_ensemble',
    'unitary_ensemble',
    ]


def multi_slice(axes: Sequence, items: Sequence,
                axes_nb: int = None) -> Tuple:
    """Construct a multidimensional slice to access array data.
    e.g. mydata[multi_slice([3,5], [1,2]))] is equivelent as
    mydata[:,:,:,1,:,2,...]
    """

    N = max(axes)+1 if axes_nb is None else axes_nb
    slices: List[Any] = [slice(None)] * N
    for axis, item in zip(axes, items):
        slices[axis] = item
    if axes_nb is None:
        slices.append(Ellipsis)
    return tuple(slices)


def invert_map(mapping: dict, one_to_one: bool = True) -> dict:
    """Invert a dictionary. If not one_to_one then the inverted
    map will contain lists of former keys as values.
    """
    if one_to_one:
        inv_map = {value: key for key, value in mapping.items()}
    else:
        inv_map = {}
        for key, value in mapping.items():
            inv_map.setdefault(value, set()).add(key)

    return inv_map


# For no apparently good reason, the python standard library contains
# immutable lists and sets (tuple and frozenset), but does not have an
# immutable dictionary.

KT = TypeVar('KT')
VT = TypeVar('VT')


class FrozenDict(Mapping[KT, VT]):
    """
    An immutable frozen dictionary.

    The FrozenDict is hashable if all the keys and values are hashable.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._dict: Dict[KT, VT] = dict(*args, **kwargs)
        self._hash: Optional[int] = None

    def __getitem__(self, key: KT) -> VT:
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def copy(self, *args: Any, **kwargs: Any) -> 'FrozenDict':
        return self.update()

    def update(self, *args: Any, **kwargs: Any) -> 'FrozenDict':
        """Update mappings, and return a new FrizenDict"""
        d = self._dict.copy()
        d.update(*args, **kwargs)
        return type(self)(d)  # TODO: type(self)(d)??

    def __iter__(self) -> Iterator[KT]:
        yield from self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return '%s(%r)' % (self.__class__.__name__, self._dict)

    def __hash__(self) -> int:
        if not self._hash:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash


def bitlist_to_int(bitlist: Sequence[int]) -> int:
    """Converts a sequence of bits to an integer.

    >>> from quantumflow.utils import bitlist_to_int
    >>> bitlist_to_int([1, 0, 0])
    4
    """
    return int(''.join([str(d) for d in bitlist]), 2)


def int_to_bitlist(x: int, pad: int = None) -> Sequence[int]:
    """Converts an integer to a binary sequence of bits.

    Pad prepends with sufficient zeros to ensures that the returned list
    contains at least this number of bits.

    >>> from quantumflow.utils import int_to_bitlist
    >>> int_to_bitlist(4, 4))
    [0, 1, 0, 0]
    """
    if pad is None:
        form = '{:0b}'
    else:
        form = '{:0' + str(pad) + 'b}'

    return [int(b) for b in form.format(x)]


# -- Function decorators --

def deprecated(func: Callable) -> Callable:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def _new_func(*args: Any, **kwargs: Any) -> Any:
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(f'Call to deprecated function {func.__name__}.',
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return _new_func


try:
    # Python >3.8
    from functools import cached_property   # type: ignore
except ImportError:

    def cached_property(func):  # type: ignore
        """
         Method decorator for immutable properties.
         The result is cached on the first call.
        """
        def wrapper(instance):  # type: ignore
            attr = '_cached_property_'+func.__name__
            if hasattr(instance, attr):
                return getattr(instance, attr)
            result = func(instance)
            setattr(instance, attr, result)
            return result
        return property(wrapper)


# Note: I can't get mypy to like this decorator
# class cached_property:
#     """
#     Method decorator for immutable properties.
#     The result is cached on the first call.
#     """
#     # Kudos: Adapted from django's cached_property. Simpler
#     # implementation because we don't try to support Python < 3.6
#     # Essentially same idea added to Python 3.8

#     # This class uses the descriptor protocol
#     # https://docs.python.org/3.6/howto/descriptor.html

#     def __init__(self, func) -> None:
#         self.func = func
#         self.__doc__ = func.__doc__

#     def __set_name__(self, owner, name: str) -> None:
#         self.name = name
#         print(name)
#         sys.exit()

#     def __get__(self, instance, owner=None) -> Any:
#         if instance is None:
#             return self
#         result = self.func(instance)
#         instance.__dict__[self.name] = result
#         return result


# -- Graphs --

def from_graph6(graph6: str) -> nx.Graph:
    """Convert a string encoded in graph6 format to a networkx graph"""
    return nx.from_graph6_bytes(bytes(graph6, "utf-8"))


def to_graph6(graph: nx.Graph) -> str:
    """Convert a networkx graph to a string in to a graph6 format"""
    # Networkx makes this surprisingly tricky.
    return nx.to_graph6_bytes(graph)[10:-1].decode("utf-8")


def spanning_tree_count(graph: nx.Graph) -> int:
    """Return the number of unique spanning trees of a graph, using
    Kirchhoff's matrix tree theorem.
    """
    laplacian = nx.laplacian_matrix(graph).toarray()
    comatrix = laplacian[:-1, :-1]
    det = np.linalg.det(comatrix)
    count = int(round(det))
    return count


def octagonal_tiling_graph(M: int, N: int) -> nx.Graph:
    """Return the octagonal tiling graph (4.8.8, truncated square tiling,
    truncated quadrille) of MxNx8 nodes

    To visualize with networkx and matplot lib:
    > import matplotlib.pyplot as plt
    > nx.draw(G, pos={node: node for node in G.nodes})
    >plt.show()
    """

    grp = nx.Graph()
    octogon = [[(1, 0), (0, 1)],
               [(0, 1), (0, 2)],
               [(0, 2), (1, 3)],
               [(1, 3), (2, 3)],
               [(2, 3), (3, 2)],
               [(3, 2), (3, 1)],
               [(3, 1), (2, 0)],
               [(2, 0), (1, 0)]]
    left = [[(1, 0), (1, -1)], [(2, 0), (2, -1)]]
    up = [[(0, 1), (-1, 1)], [(0, 2), (-1, 2)]]

    for m in range(M):
        for n in range(N):
            edges = octogon
            if n != 0:
                edges = edges + left
            if m != 0:
                edges = edges + up

            for (x0, y0), (x1, y1) in edges:
                grp.add_edge((m*4+x0, n*4+y0), (m*4+x1, n*4+y1))
    return grp


def truncated_grid_2d_graph(m: int, n: int, t: int = None) -> nx.Graph:
    """Generate a rectangular grid graph (of width `m` and height `n`),
    with corners removed. It the truncation `t` is not given, then it
    is set to half the shortest side (rounded down)

    truncated_grid_graph(12, 11) returns the topology of Google's
    bristlecone chip.
    """
    G = nx.grid_2d_graph(m, n)
    if t is None:
        t = min(m, n) // 2

    for mm in range(t):
        for nn in range(t-mm):
            G.remove_node((mm, nn))
            G.remove_node((m-mm-1, nn))
            G.remove_node((mm, n-nn-1))
            G.remove_node((m-mm-1, n-nn-1))
    return G


# -- More Math --


_DENOMINATORS = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64,
                     128, 256, 512, 1024, 2048, 4096, 8192])


def rationalize(flt: float, denominators: Set[int] = None) -> Fraction:
    """Convert a floating point number to a Fraction with a small
    denominator.

    Args:
        flt:            A floating point number
        denominators:   Collection of standard denominators. Default is
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64, 128, 256, 512,
            1024, 2048, 4096, 8192

    Raises:
        ValueError:     If cannot rationalize float
    """
    if denominators is None:
        denominators = _DENOMINATORS
    frac = Fraction.from_float(flt).limit_denominator()
    if frac.denominator not in denominators:
        raise ValueError('Cannot rationalize')
    return frac


def symbolize(flt: float) -> sympy.Symbol:
    """Attempt to convert a real number into a simpler symbolic
    representation.

    Returns:
        A sympy Symbol. (Convert to string with str(sym) or to latex with
            sympy.latex(sym)
    Raises:
        ValueError:     If cannot simplify float
    """
    try:
        ratio = rationalize(flt)
        res = sympy.simplify(ratio)
    except ValueError:
        try:
            ratio = rationalize(flt/np.pi)
            res = sympy.simplify(ratio) * sympy.pi
        except ValueError:
            ratio = rationalize(flt*np.pi)
            res = sympy.simplify(ratio) / sympy.pi
    return res


def complex_ginibre_ensemble(size: Tuple[int, ...]) -> np.ndarray:
    """Returns a random complex matrix with values draw from a standard normal
    distribution.

    Ref:
        Ginibre, Jean (1965). "Statistical ensembles of complex, quaternion,
        and real matrices". J. Math. Phys. 6: 440–449.
        doi:10.1063/1.1704292.
    """
    return np.random.normal(size=size) + 1j * np.random.normal(size=size)


def unitary_ensemble(dim: int) -> np.ndarray:
    """Return a random unitary of size (dim, dim) drawn from Harr measure

     Ref:
        "How to generate random matrices from the classical compact groups",
         Francesco Mezzadri, Notices Am. Math. Soc. 54, 592 (2007).
         arXiv:math-ph/0609050
    """
    import scipy
    return scipy.stats.unitary_group.rvs(dim)


# itertools

# DOCME
# # TESTME
# def last(seq: Iterator) -> Any:
#     return deque(seq, 1)


# Unicode and ASCII characters for drawing boxes.
#                  t 0000000011111111
#                  r 0000111100001111
#                  b 0011001100110011
#                  l 0101010101010101
BOX_CHARS         = " ╴╷┐╶─┌┬╵┘│┤└┴├┼"   # noqa: E221
BOLD_BOX_CHARS    = " ╸╻┓╺━┏┳╹┛┃┫┗┻┣╋"   # noqa: E221
DOUBLE_BOX_CHARS  = " ═║╗══╔╦║╝║╣╚╩╠╬"   # noqa: E221  # No half widths
ASCII_BOX_CHARS  = r"   \ -/+ /|+\+++"   # noqa: E221

BOX_TOP, BOX_RIGHT, BOX_BOT, BOX_LEFT = 8, 4, 2, 1
BOX_CROSS = BOX_TOP+BOX_RIGHT+BOX_BOT+BOX_LEFT


# fin
