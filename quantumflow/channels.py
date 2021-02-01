# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
.. contents:: :local:
.. currentmodule:: quantumflow


Mixed States and Quantum Channels
#################################
.. autoclass:: Density
    :members:

.. autoclass:: Channel
    :members:

.. autoclass:: Kraus
    :members:

.. autoclass:: UnitaryMixture
    :members:

Actions on Densities
####################
.. autofunction:: mixed_density
.. autofunction:: random_density
.. autofunction:: join_densities


Actions on Channels
###################
.. autofunction:: join_channels
.. autofunction:: channel_to_kraus
.. autofunction:: kraus_iscomplete


Standard channels
#################
.. autoclass:: Dephasing
    :members:

.. autoclass:: Damping
    :members:

.. autoclass:: Depolarizing
    :members:

.. autofunction:: random_channel
"""

# Kudos: Kraus maps originally adapted from Nick Rubin's reference-qvm

from functools import reduce
from operator import add
from typing import Sequence

import numpy as np
from scipy import linalg

from . import tensors, utils
from .ops import Channel, Gate, Operation, Unitary
from .qubits import Qubit, Qubits
from .states import Density, State
from .stdgates import I, X, Y, Z

__all__ = [
    "Kraus",
    "UnitaryMixture",
    "Depolarizing",
    "Damping",
    "Dephasing",
    "join_channels",
    "channel_to_kraus",
    "kraus_iscomplete",
    "random_channel",
]


class Kraus(Operation):
    """A Kraus representation of a quantum channel"""

    # DOCME: operator-value-sum representation
    # FIXME: Shouldn't take Gate, since may not be Unitary operators

    def __init__(
        self, operators: Sequence[Gate], weights: Sequence[float] = None
    ) -> None:
        self.operators = operators

        if weights is None:
            weights = [1.0] * len(operators)

        self.weights = tuple(weights)

    def asgate(self) -> Gate:
        """Not possible in general. (But see UnitaryMixture)

        Raises: TypeError
        """
        raise TypeError("Not possible in general")

    def aschannel(self) -> Channel:
        """Returns: Action of Kraus operators as a superoperator Channel"""
        qubits = self.qubits
        N = len(qubits)
        ident = Unitary(np.eye(2 ** N), qubits).aschannel()

        tensors = [(op.aschannel() @ ident).tensor for op in self.operators]
        if self.weights is not None:
            tensors = [t * w for t, w in zip(tensors, self.weights)]
        chan_tensor = reduce(add, tensors)

        return Channel(chan_tensor, self.qubits)

    def run(self, ket: State) -> State:
        """Apply the action of this Kraus quantum operation upon a state"""
        res = [op.run(ket) for op in self.operators]
        probs = np.asarray(list(ket.norm() * w for ket, w in zip(res, self.weights)))
        probs = np.abs(probs)
        probs /= np.sum(probs)
        n = np.random.choice(len(res), p=probs)
        newket = res[n]
        return newket.normalize()

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this Kraus quantum operation upon a density"""
        qubits = rho.qubits
        results = [op.evolve(rho) for op in self.operators]
        tensors = [rho.tensor * w for rho, w in zip(results, self.weights)]
        tensor = reduce(add, tensors)
        return Density(tensor, qubits)

    @property
    def qubits(self) -> Qubits:
        """Returns: List of qubits acted upon by this Kraus operation

        The list of qubits is ordered if the qubits labels can be sorted,
        else the the order is indeterminate.

        Raises:
            TypeError: If qubits cannot be sorted into unique order.
        """
        qbs = [q for elem in self.operators for q in elem.qubits]  # gather
        qbs = list(set(qbs))  # unique
        qbs = sorted(qbs)  # sort
        return tuple(qbs)

    @property
    def H(self) -> "Kraus":
        """Return the complex conjugate of this Kraus operation"""
        operators = [op.H for op in self.operators]
        return Kraus(operators, self.weights)


# End class Kraus


class UnitaryMixture(Kraus):
    """A Kraus channel which is a convex mixture of unitary dynamics.

    This Channel is unital, but not all unital channels are unitary
    mixtures.
    """

    def __init__(
        self, operators: Sequence[Gate], weights: Sequence[float] = None
    ) -> None:

        from .info import almost_unitary

        for op in operators:
            if not almost_unitary(op):
                raise ValueError("Operators not all unitary")  # pragma: no cover

        if weights is not None and not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to unity")  # pragma: no cover  # TESTME

        super().__init__(operators, weights)
        # TODO: Sanity check. operators unitary, weights unit

    def asgate(self) -> Gate:
        """Return one of the composite Kraus operators at random with
        the appropriate weights"""
        return np.random.choice(self.operators, p=self.weights)

    def run(self, ket: State) -> State:
        return self.asgate().run(ket)


class Depolarizing(UnitaryMixture):
    """A Kraus representation of a depolarizing channel on 1-qubit.

    Args:
        prob:   The one-step depolarizing probability.
        q0:     The qubit on which to act.
    """

    def __init__(self, prob: float, q0: Qubit) -> None:
        operators = [I(q0), X(q0), Y(q0), Z(q0)]
        weights = [1 - prob, prob / 3.0, prob / 3.0, prob / 3.0]
        super().__init__(operators, weights)


class Damping(Kraus):
    """A Kraus representation of an amplitude-damping (spontaneous emission)
    channel on one qubit

    Args:
        prob:   The one-step damping probability.
        q0:     The qubit on which to act.
    """

    def __init__(self, prob: float, q0: Qubit) -> None:
        kraus0 = Unitary([[1.0, 0.0], [0.0, np.sqrt(1 - prob)]], [q0])
        kraus1 = Unitary([[0.0, np.sqrt(prob)], [0.0, 0.0]], [q0])
        super().__init__([kraus0, kraus1])


class Dephasing(UnitaryMixture):
    """A Kraus representation of a phase-damping quantum channel

    Args:
        prob:   The one-step damping probability.
        q0:     The qubit on which to act.
    """

    def __init__(self, prob: float, q0: Qubit) -> None:
        operators = [I(q0), Z(q0)]
        weights = [1 - prob / 2, prob / 2]
        super().__init__(operators, weights)


def join_channels(chan0: Channel, chan1: Channel) -> Channel:
    """Join two channels acting on different qubits into a single channel
    acting on all qubits"""
    tensor = tensors.outer(chan0.tensor, chan1.tensor, rank=4)
    return Channel(tensor, tuple(chan0.qubits) + tuple(chan1.qubits))


# TESTME
def channel_to_kraus(chan: Channel) -> "Kraus":
    """Convert a channel superoperator into a Kraus operator representation
    of the same channel."""
    qubits = chan.qubits
    N = chan.qubit_nb

    choi = chan.choi()
    evals, evecs = np.linalg.eig(choi)
    evecs = np.transpose(evecs)

    assert np.allclose(evals.imag, 0.0)  # FIXME exception
    assert np.all(evals.real >= 0.0)  # FIXME exception

    values = np.sqrt(evals.real)

    ops = []
    for i in range(2 ** (2 * N)):
        if not np.isclose(values[i], 0.0):
            mat = np.reshape(evecs[i], (2 ** N, 2 ** N)) * values[i]
            g = Unitary(mat, qubits)
            ops.append(g)

    return Kraus(ops)


def kraus_iscomplete(kraus: Kraus) -> bool:
    """Returns True if the collection of (weighted) Kraus operators are
    complete. (Which is necessary for a CPTP map to preserve trace)
    """
    qubits = kraus.qubits
    N = kraus.qubit_nb

    ident = Unitary(np.eye(2 ** N), qubits)

    tensors = [(op.H @ op @ ident).asoperator() for op in kraus.operators]
    tensors = [t * w for t, w in zip(tensors, kraus.weights)]

    tensor = reduce(np.add, tensors)
    res = Unitary(tensor, qubits)

    N = res.qubit_nb
    return np.allclose(res.asoperator(), np.eye(2 ** N))


# TODO: as class RandomChannel?
# Author: GEC (2019)
def random_channel(qubits: Qubits, rank: int = None, unital: bool = False) -> Channel:
    """
    Returns: A randomly sampled Channel drawn from the BCSZ ensemble with
    the specified Kraus rank.

    Args:
        qubits: A list, or number, of qubits.
        rank: Kraus rank of channel. (Defaults to full rank)

    Ref:
        "Random quantum operations", Bruzda, Cappellini, Sommers, and
        Zyczkowski, Physics Letters A 373, 320 (2009). arXiv:0804.2361
    """
    qubits = tuple(qubits)
    N = len(qubits)
    dim = 2 ** N  # Hilbert space dimension
    size = (dim ** 2, dim ** 2) if rank is None else (dim ** 2, rank)

    # arXiv:0804.2361 page 4, steps 1 to 4
    # arXiv:0709.0824 page 6
    X = utils.complex_ginibre_ensemble(size)
    XX = X @ X.conj().T

    if unital:
        Y = np.einsum("ikjk -> ij", XX.reshape([dim, dim, dim, dim]))
        Q = np.kron(linalg.sqrtm(linalg.inv(Y)), np.eye(dim))
    else:
        Y = np.einsum("kikj -> ij", XX.reshape([dim, dim, dim, dim]))
        Q = np.kron(np.eye(dim), linalg.sqrtm(linalg.inv(Y)))

    choi = Q @ XX @ Q

    return Channel.from_choi(choi, qubits)


# Fin
