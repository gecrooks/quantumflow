
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. module:: quantumflow
.. contents:: :local:
.. currentmodule:: quantumflow


States, gates, and various other methods accept a list of qubits labels upon
which the given State or Gate acts. A Qubit label can be any hashable python
object, but typically an integer or string. e.g. `[0, 1, 2]`, or
`['a', 'b', 'c']`. Note that some operations expect the qubits to be sortable,
so don't mix different uncomparable data types.


.. autoclass:: QubitVector
    :members:

"""

from typing import Hashable, Sequence, Union, Tuple, List
from copy import copy

import numpy as np

from .config import TOLERANCE
from . import backend as bk
from .backend.numpybk import EINSUM_SUBSCRIPTS

__all__ = ['Qubit', 'Qubits', 'asarray', 'QubitVector',
           'inner_product', 'outer_product', 'fubini_study_angle',
           'vectors_close']


Qubit = Hashable
"""Type for qubits. Any hashable python object."""


Qubits = Sequence[Qubit]
"""Type for sequence of qubits"""


# TODO: DOCME
def asarray(tensor: bk.BKTensor) -> bk.TensorLike:
    """Convert QuantumFlow backend tensor to numpy array"""
    return bk.evaluate(tensor)


class QubitVector:
    """
    A container for a tensor representation of a quantum state or operation
    on qubits. The QubitTensor contains a list of qubits, and a backend
    complex tensor in product form.

    The rank of the quantum tensor represents the order of the quantum
    state of operation::

        Rank
        1   vectors
        2   operators
        4   super-operators
        8   super-duper-operators

    A vector is an element of a vector space.
    An operator (normally represented by a 2 dimensional matrix) is a linear
    map on a vector space. Super-operators are operators of operators --
    linear maps on the vector space of operators. Similarly rank 8 super-duper
    operators are operators of super-operators.

    Pure states are represented by vectors; gates and mixed states by rank 2
    operators; and quantum channels by rank 4 super-operators.

    A tensor of rank R on N qubits is stored as a tensor product with (N*R)
    dimensions, each of length 2. (Except for scalers, rank=0, which have
    only a single element) In other words the shape of the tensor array is
    [2]*(N*R). This representation is convenient for addressing individual
    qubits (e.g. For a state of 4 qubits, the state amplitude of 0110 is
    located in tensor element [0,1,1,0])

    A quantum logic gate is a unitary operator acting on a collection of K
    qubits, which can be represented by a matrix of shape (2**K, 2**K).
    This operator is stored as a mixed tensor of shape
    ([2]*(2*K)). e.g. for 4 qubits, the gate shape is
    (2, 2, 2, 2, 2, 2, 2, 2). Gates have ket and bra components, which are
    ordered kets then bras, e.g.  ``gate[k0, k1, k2, k3, b0, b1, b2, b3]``.
    This is so we can go back and forth between the product tensor and more
    common matrix representations with a simple reshape.::

        operator_matrix = np.reshape(gate_tensor, shape=(2**K, 2**K))
        gate_tensor = np.reshape(operator_matrix, shape=([2]*(2*K))

    The indices of superoperators are ordered
    [ket_out, bra_out, bra_in, ket_in].


    Args:
        qubits: A sequence of qubit labels
        array: An tensor or tensor like object

    Attributes:
        QubitVector.qubits:     Qubit labels
        QubitVector.qubit_nb:   The number of qubits
        QubitVector.rank:       Order of the tensor, which has (qubit_nb*rank)
                                dimensions

    """

    def __init__(self,
                 tensor: bk.TensorLike,
                 qubits: Qubits,
                 rank: int = None
                 ) -> None:

        tensor = bk.astensorproduct(tensor)
        self.tensor = tensor

        N = len(qubits)
        if rank is None:
            if N == 0:   # FIXME: Why this special case!?
                rank = 1
            else:
                rank = bk.ndim(tensor) // N

        if rank not in [1, 2, 4, 8] or rank * N != bk.ndim(tensor):
            raise ValueError('Incompatibility between tensor and qubits: '
                             'rank={} qubit_nb={} shape={}'
                             .format(rank, N, tensor.shape))

        self.qubits = tuple(qubits)
        self.qubit_nb = N

        # TODO: Rename rank to order or degree in effort to disambiguate
        # different uses of rank from tensor?
        self.rank = rank

    # # FIXME: Does not respect qubits
    # def __getitem__(self, key: Any) -> bk.BKTensor:
    #     return bk.getitem(self.tensor, key)

    def asarray(self) -> np.ndarray:
        """Return the tensor as a numpy array"""
        return bk.evaluate(self.tensor)

    def flatten(self) -> bk.BKTensor:
        """Return tensor with with qubit indices flattened"""
        N = self.qubit_nb
        R = self.rank
        return bk.reshape(self.tensor, [2**N]*R)

    def relabel(self, qubits: Qubits) -> 'QubitVector':
        """Return a copy of this vector with new qubits"""
        qubits = tuple(qubits)
        assert len(qubits) == self.qubit_nb
        vec = copy(self)
        vec.qubits = qubits
        return vec

    def permute(self, qubits: Qubits) -> 'QubitVector':
        """Permute the order of the qubits"""

        if qubits == self.qubits:
            return self

        N = self.qubit_nb
        assert len(qubits) == N

        # Will raise a value error if qubits don't match
        indices = [self.qubits.index(q) for q in qubits]  # type: ignore
        perm: List[int] = []
        for rr in range(0, self.rank):
            perm += [rr * N + idx for idx in indices]
        tensor = bk.transpose(self.tensor, perm)

        return QubitVector(tensor, qubits)

    # TESTME For superoperators
    def transpose(self, perm: Sequence[int] = None) -> 'QubitVector':
        """(Super)-operator transpose. Permutes the meta-indices.
        Default is to invert the meta-index order.
        """
        N = self.qubit_nb
        R = self.rank

        # (super) operator transpose
        tensor = self.tensor
        tensor = bk.reshape(tensor, [2**(N*R//2)] * 2)
        tensor = bk.transpose(tensor)
        tensor = bk.reshape(tensor, [2] * R * N)

        return QubitVector(tensor, self.qubits)

    # TESTME For superoperators, needs more tests
    @property
    def H(self) -> 'QubitVector':
        """Return the conjugate transpose of this tensor."""
        tensor = self.transpose().tensor
        tensor = bk.conj(tensor)

        return QubitVector(tensor, self.qubits)

    def norm(self) -> bk.BKTensor:
        """Return the norm of this vector"""
        return bk.absolute(bk.inner(self.tensor, self.tensor))

    # TESTME
    def trace(self) -> bk.BKTensor:
        """
        Return the trace, the sum of the diagonal elements of the (super)
        operator.
        """
        N = self.qubit_nb
        R = self.rank
        if R == 1:
            raise ValueError('Cannot take trace of vector')

        tensor = bk.reshape(self.tensor, [2**(N*R//2)] * 2)
        tensor = bk.trace(tensor)
        return tensor

    # TESTME on channels and density
    # DOCME
    def partial_trace(self, qubits: Qubits) -> 'QubitVector':
        """
        Return the partial trace over some subset of qubits,

        args:
            qubits: The set of qubits that should be kept.
        """
        N = self.qubit_nb
        R = self.rank
        if R == 1:
            raise ValueError('Cannot take trace of vector')

        if not qubits:
            raise ValueError('Cannot remove all qubits with partial_trace.')

        contract_qubits: List[Qubit] = list(self.qubits)
        for q in qubits:
            contract_qubits.remove(q)

        indices = [self.qubits.index(qubit) for qubit in contract_qubits]
        subscripts = list(EINSUM_SUBSCRIPTS)[0:N*R]
        for idx in indices:
            for r in range(1, R):
                subscripts[r * N + idx] = subscripts[idx]
        subscript_str = ''.join(subscripts)

        # Only numpy's einsum works with repeated subscripts
        tensor = self.asarray()
        tensor = np.einsum(subscript_str, tensor)

        return QubitVector(tensor, qubits)

    # TODO: Generalize for channels?
    # DOCME
    # TODO: Need better name?
    def promote_to_operator(self, qubits: Qubits = None) -> 'QubitVector':
        """
        Promote a vector to an operator. If qubits argument given then
        return operator for that subset of qubits (By taking a partial trace
        over the rest.)
        """
        # We do two steps in one to avoid creating a large intermediate
        # operator.

        # FIXME: Feels complicated. Is there a similar way?

        N = self.qubit_nb
        R = self.rank
        assert R == 1  # FIXME: Generalize?

        if qubits is None:
            qubits = self.qubits

        contract_qubits: List[Qubit] = list(self.qubits)
        for q in qubits:
            contract_qubits.remove(q)

        left = np.asarray([list(range(r*N, (r+1)*N)) for r in range(R)])
        right = np.asarray([list(range(r*N, (r+1)*N)) for r in range(R, 2*R)])

        # FIXME: Add method to vec
        indices = [self.qubits.index(qubit) for qubit in contract_qubits]
        for idx in indices:
            for r in range(R):
                right[r, idx] = left[r, idx]

        left_subs = tuple(left.flat)
        right_subs = tuple(right.flat)

        left_tensor = self.asarray()
        right_tensor = np.conj(left_tensor)

        tensor = bk.contract(left_tensor, left_subs, right_tensor, right_subs)

        # NN = len(qubits)
        # tensor = bk.reshape(tensor, ([2**NN] * R) + ([2**NN] * R))
        # perm = [idx for ij in zip(range(0, R), range(R, 2*R)) for idx in ij]
        # tensor = bk.transpose(tensor, perm)

        return QubitVector(tensor, qubits)

# End QubitVector


def inner_product(vec0: QubitVector, vec1: QubitVector) -> bk.BKTensor:
    """ Hilbert-Schmidt inner product between qubit vectors

    The tensor rank and qubits must match.
    """
    if vec0.rank != vec1.rank or vec0.qubit_nb != vec1.qubit_nb:
        raise ValueError('Incompatibly vectors. Qubits and rank must match')

    vec1 = vec1.permute(vec0.qubits)  # Make sure qubits in same order
    return bk.inner(vec0.tensor, vec1.tensor)


# TESTME
# TODO: replace join_states ect?, join ...
# also used in gates to channels
def outer_product(vec0: QubitVector, vec1: QubitVector) -> QubitVector:
    """Direct product of qubit vectors

    The tensor ranks must match and qubits must be disjoint.
    """
    R = vec0.rank
    R1 = vec1.rank

    N0 = vec0.qubit_nb
    N1 = vec1.qubit_nb

    if R != R1:
        raise ValueError('Incompatibly vectors. Rank must match')

    if not set(vec0.qubits).isdisjoint(vec1.qubits):
        raise ValueError('Overlapping qubits')

    qubits: Qubits = tuple(vec0.qubits) + tuple(vec1.qubits)

    tensor = bk.outer(vec0.tensor, vec1.tensor)

    # Interleave (super)-operator axes
    # R = 1  perm = (0, 1)
    # R = 2  perm = (0, 2, 1, 3)
    # R = 4  perm = (0, 4, 1, 5, 2, 6, 3, 7)
    tensor = bk.reshape(tensor, ([2**N0] * R) + ([2**N1] * R))
    perm = [idx for ij in zip(range(0, R), range(R, 2*R)) for idx in ij]
    tensor = bk.transpose(tensor, perm)

    return QubitVector(tensor, qubits)


# TODO: move to measures.py ?
# TESTME
def fubini_study_angle(vec0: QubitVector, vec1: QubitVector) -> bk.BKTensor:
    """Calculate the Fubini–Study metric between elements of a Hilbert space.

    The Fubini–Study metric is a distance measure between vectors in a
    projective Hilbert space. For gates this space is the Hilbert space of
    operators induced by the Hilbert-Schmidt inner product.
    For 1-qubit rotation gates, RX, RY and RZ, this is half the angle (theta)
    in the Bloch sphere.

    The Fubini–Study metric between states is equal to the Burr angle
    between pure states.
    """
    fs_fidelity = fubini_study_fidelity(vec0, vec1)
    return bk.arccos(fs_fidelity)


def fubini_study_fidelity(vec0: QubitVector, vec1: QubitVector) -> bk.BKTensor:
    """
    Cosine of the Fubini–Study metric.
    """
    # Suffers from less floating poitn errors compared to fubini_study_angle
    if vec0.rank != vec1.rank or vec0.qubit_nb != vec1.qubit_nb:
        raise ValueError('Incompatibly vectors. Qubits and rank must match')

    vec1 = vec1.permute(vec0.qubits)  # Make sure qubits in same order

    t0 = vec0.tensor
    t1 = vec1.tensor
    hs01 = bk.inner(t0, t1)  # Hilbert-Schmidt inner product
    hs00 = bk.inner(t0, t0)
    hs11 = bk.inner(t1, t1)
    ratio = bk.absolute(hs01) / bk.sqrt(bk.absolute(hs00*hs11))
    fid = bk.minimum(ratio, bk.fcast(1.))  # Compensate for rounding errors.
    return fid


# TODO: move to measures.py ?
def vectors_close(vec0: QubitVector, vec1: QubitVector,
                  tolerance: float = TOLERANCE) -> bool:
    """Return True if vectors in close in the projective Hilbert space.

    Similarity is measured with the Fubini–Study metric.
    """
    if vec0.rank != vec1.rank:
        return False

    if vec0.qubit_nb != vec1.qubit_nb:
        return False

    if set(vec0.qubits) ^ set(vec1.qubits):
        return False

    return 1 - bk.evaluate(fubini_study_fidelity(vec0, vec1)) <= tolerance


# Note: Not a public method
def qubits_count_tuple(qubits: Union[int, Qubits]) -> Tuple[int, Qubits]:
    """Utility method for unraveling 'qubits: Union[int, Qubits]' arguments"""
    if isinstance(qubits, int):
        return qubits, tuple(range(qubits))
    return len(qubits), qubits
