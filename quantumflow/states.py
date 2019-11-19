
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
QuantumFlow representations of pure quantum states and actions on states.

.. contents:: :local:

State objects
#############
.. autoclass:: State
    :members:


Standard states
###############
.. autofunction:: zero_state
.. autofunction:: w_state
.. autofunction:: ghz_state
.. autofunction:: random_state


Actions on states
#################
.. autofunction:: join_states
.. autofunction:: print_state
.. autofunction:: print_probabilities

"""

from math import sqrt
from typing import Union, TextIO, Any, Mapping, List
from functools import reduce
from collections import ChainMap

import numpy as np

from . import backend as bk
from .qubits import Qubits, QubitVector, qubits_count_tuple
from .qubits import outer_product
from .utils import complex_ginibre_ensemble, unitary_ensemble
from .utils import FrozenDict

__all__ = ['State', 'ghz_state',
           'join_states', 'print_probabilities', 'print_state',
           'random_state', 'w_state', 'zero_state',
           'Density', 'mixed_density', 'random_density', 'join_densities']


class State:
    """The quantum state of a collection of qubits.

    Note that memory usage grows exponentially with the number of qubits.
    (16*2^N bytes for N qubits)

    """

    def __init__(self,
                 tensor: bk.TensorLike,
                 qubits: Qubits = None,
                 memory: Mapping = None) -> None:
        """Create a new State from a tensor of qubit amplitudes

        Args:
            tensor: A vector or tensor of state amplitudes
            qubits: A sequence of qubit names.
                (Defaults to integer indices, e.g. [0, 1, 2] for 3 qubits)
            memory: Classical data storage. Stored as an immutable dictionary.
        """
        if qubits is None:
            tensor = bk.astensorproduct(tensor)
            bits = bk.ndim(tensor)
            qubits = range(bits)

        self.vec = QubitVector(tensor, qubits)

        if memory is None:
            self.memory: FrozenDict = FrozenDict()
        elif isinstance(memory, FrozenDict):
            self.memory = memory
        else:
            self.memory = FrozenDict(memory)

    @property
    def tensor(self) -> bk.BKTensor:
        """Returns the tensor representation of state vector"""
        return self.vec.tensor

    @property
    def qubits(self) -> Qubits:
        """Return qubit labels of this state"""
        return self.vec.qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits"""
        return self.vec.qubit_nb

    def qubit_indices(self, qubits: Qubits) -> List[int]:
        """Convert qubits to index positions."""
        return [self.qubits.index(q) for q in qubits]

    def norm(self) -> bk.BKTensor:
        """Return the state vector norm"""
        return self.vec.norm()

    # DOCME TESTME
    def store(self, *args: Any, **kwargs: Any) -> 'State':
        mem = self.memory.copy(*args, **kwargs)
        return State(self.tensor, self.qubits, mem)

    def relabel(self, qubits: Qubits) -> 'State':
        """Return a copy of this state with new qubits"""
        return State(self.vec.tensor, qubits, self.memory)

    def permute(self, qubits: Qubits = None) -> 'State':
        """Return a copy of this state with state tensor transposed to
        put qubits in the given order. If an explicet qubit
        ordering isn't supplied, we put qubits in sorted order.
        """
        if qubits is None:
            qubits = sorted(self.qubits)
        vec = self.vec.permute(qubits)
        return State(vec.tensor, vec.qubits, self.memory)

    def normalize(self) -> 'State':
        """Normalize the state"""
        tensor = self.tensor / bk.ccast(bk.sqrt(self.norm()))
        return State(tensor, self.qubits, self.memory)

    def probabilities(self) -> bk.BKTensor:
        """
        Returns:
            The state probabilities
        """
        value = bk.absolute(self.tensor)
        return value * value

    def sample(self, trials: int) -> np.ndarray:
        """Measure the state in the computational basis the the given number
        of trials, and return the counts of each output configuration.
        """
        # TODO: Can we do this within backend?
        probs = np.real(bk.evaluate(self.probabilities()))
        res = np.random.multinomial(trials, probs.ravel())
        res = res.reshape(probs.shape)
        return res

    def expectation(self, diag_hermitian: bk.TensorLike,
                    trials: int = None) -> bk.BKTensor:
        """Return the expectation of a measurement. Since we can only measure
        our computer in the computational basis, we only require the diagonal
        of the Hermitian in that basis.

        If the number of trials is specified, we sample the given number of
        times. Else we return the exact expectation (as if we'd performed an
        infinite number of trials. )
        """
        if trials is None:
            probs = self.probabilities()
        else:
            probs = bk.real(bk.astensorproduct(self.sample(trials) / trials))

        diag_hermitian = bk.astensorproduct(diag_hermitian)
        return bk.reduce_sum(bk.real(diag_hermitian) * probs)

    def measure(self) -> np.ndarray:
        """Measure the state in the computational basis.

        Returns:
            A [2]*bits array of qubit states, either 0 or 1
        """
        # TODO: Can we do this within backend?
        probs = np.real(bk.evaluate(self.probabilities()))
        indices = np.asarray(list(np.ndindex(*[2] * self.qubit_nb)))
        res = np.random.choice(probs.size, p=probs.ravel())
        res = indices[res]
        return res

    def asdensity(self, qubits: Qubits = None) -> 'Density':
        """Convert a pure state to a density matrix.

        args:
            qubits: The qubit subspace. If not given return the density
                    matrix for all the qubits (which can take a lot of memory!)
        """
        vec = self.vec.promote_to_operator(qubits)
        return Density(vec.tensor, vec.qubits, self.memory)

    def __str__(self) -> str:
        state = self.vec.asarray()
        s = []
        count = 0
        MAX_ELEMENTS = 64
        for index, amplitude in np.ndenumerate(state):
            if not np.isclose(amplitude, 0.0):
                ket = '|' + ''.join([str(n) for n in index]) + '>'
                s.append(f'({amplitude.real:0.04g}'
                         f'{amplitude.imag:+0.04g}i) {ket}')
                count += 1
                if count > MAX_ELEMENTS:
                    s.append('...')
                    break
        return ' + '.join(s)

# End class State


def zero_state(qubits: Union[int, Qubits]) -> State:
    """Return the all-zero state on N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    ket[(0,) * N] = 1
    return State(ket, qubits)


def w_state(qubits: Union[int, Qubits]) -> State:
    """Return a W state on N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    for n in range(N):
        idx = np.zeros(shape=N, dtype=int)
        idx[n] += 1
        ket[tuple(idx)] = 1 / sqrt(N)
    return State(ket, qubits)


def ghz_state(qubits: Union[int, Qubits]) -> State:
    """Return a GHZ state on N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    ket[(0, ) * N] = 1 / sqrt(2)
    ket[(1, ) * N] = 1 / sqrt(2)
    return State(ket, qubits)


def random_state(qubits: Union[int, Qubits]) -> State:
    """Return a random state from the space of N qubits"""
    N, qubits = qubits_count_tuple(qubits)
    ket = np.random.normal(size=([2] * N)) \
        + 1j * np.random.normal(size=([2] * N))
    return State(ket, qubits).normalize()


# == Actions on States ==


def join_states(*states: State) -> State:
    """Join two state vectors into a larger qubit state"""
    vectors = [ket.vec for ket in states]
    vec = reduce(outer_product, vectors)
    return State(vec.tensor, vec.qubits)


# = Output =

# FIXME: clean up. Move to visulization?

def print_state(state: State, file: TextIO = None) -> None:
    """Print a state vector"""
    state = state.vec.asarray()
    for index, amplitude in np.ndenumerate(state):
        ket = "".join([str(n) for n in index])
        print(ket, ":", amplitude, file=file)


# TODO: Should work for density also. Check
def print_probabilities(state: State, ndigits: int = 4,
                        file: TextIO = None) -> None:
    """
    Pretty print state probabilities.

    Args:
        state:
        ndigits: Number of digits of accuracy
        file: Output stream (Defaults to stdout)
    """
    prob = bk.evaluate(state.probabilities())
    for index, prob in np.ndenumerate(prob):
        prob = round(prob, ndigits)
        if prob == 0.0:
            continue
        ket = "".join([str(n) for n in index])
        print(ket, ":", prob, file=file)


# --  Mixed Quantum States --

class Density(State):
    """A density matrix representation of a mixed quantum state"""
    def __init__(self,
                 tensor: bk.TensorLike,
                 qubits: Qubits = None,
                 memory: Mapping = None) -> None:
        if qubits is None:
            tensor = bk.astensorproduct(tensor)
            bits = bk.ndim(tensor) // 2
            qubits = range(bits)

        super().__init__(tensor, qubits, memory)

    def trace(self) -> bk.BKTensor:
        """Return the trace of this density operator"""
        return self.vec.trace()

    def relabel(self, qubits: Qubits) -> 'Density':
        """Return a copy of this state with new qubits"""
        return Density(self.vec.tensor, qubits, self.memory)

    def permute(self, qubits: Qubits = None) -> 'Density':
        """Return a copy of this state with qubit labels permuted"""
        if qubits is None:
            qubits = sorted(self.qubits)
        vec = self.vec.permute(qubits)
        return Density(vec.tensor, vec.qubits, self.memory)

    def normalize(self) -> 'Density':
        """Normalize state"""
        tensor = self.tensor / self.trace()
        return Density(tensor, self.qubits, self.memory)

    # TESTME
    def probabilities(self) -> bk.BKTensor:
        """Returns: The state probabilities """
        prob = bk.productdiag(self.tensor)
        return prob

    def asoperator(self) -> bk.BKTensor:
        """Return the density matrix as a square array"""
        return self.vec.flatten()

    # TESTME, DOCME
    def asdensity(self, qubits: Qubits = None) -> 'Density':
        if qubits is None:
            return self
        vec = self.vec.partial_trace(qubits)
        return Density(vec.tensor, vec.qubits, self.memory)

    # DOCME TESTME
    def store(self, *args: Any, **kwargs: Any) -> 'Density':
        mem = self.memory.copy(*args, **kwargs)
        return Density(self.tensor, self.qubits, mem)


def mixed_density(qubits: Union[int, Qubits]) -> Density:
    """Returns the completely mixed density matrix"""
    N, qubits = qubits_count_tuple(qubits)
    matrix = np.eye(2**N) / 2**N
    return Density(matrix, qubits)


def random_density(qubits: Union[int, Qubits],
                   rank: int = None,
                   ensemble: str = 'Hilbert–Schmidt') -> Density:
    """
    Returns: A randomly sampled Density

    Args:
        qubits: A list or number of qubits.
        rank: Rank of density matrix. (Defaults to full rank)
        ensemble: Either 'Hilbert–Schmidt' (default) or 'Burr'

    Ref:
        - "Induced measures in the space of mixed quantum states" Karol
          Zyczkowski, Hans-Juergen Sommers, J. Phys. A34, 7111-7125 (2001)
          arXiv:quant-ph/0012101
        - "Random Bures mixed states and the distribution of their purity",
          Osipov, Sommers, and Zyczkowski, J. Phys. A: Math. Theor. 43,
          055302 (2010). arXiv:0909.5094


    """
    if ensemble == 'Hilbert–Schmidt':
        return random_density_hs(qubits, rank)
    elif ensemble == 'Bures':
        return random_density_bures(qubits, rank)
    raise ValueError("Unknown ensemble. "
                     "Valid Options are 'Hilbert–Schmidt' or 'Bures")


# TODO: Check math
def random_density_hs(qubits: Union[int, Qubits],
                      rank: int = None) -> Density:
    """
    Returns: A randomly sampled Density from the Hilbert–Schmidt
                ensemble of quantum states.

    Args:
        qubits: A list or number of qubits.
        rank: Rank of density matrix. (Defaults to full rank)

    Ref:
        "Induced measures in the space of mixed quantum states" Karol
        Zyczkowski, Hans-Juergen Sommers, J. Phys. A34, 7111-7125 (2001)
        arXiv:quant-ph/0012101
    """
    N, qubits = qubits_count_tuple(qubits)
    size = (2**N, 2**N) if rank is None else (2**N, rank)

    X = complex_ginibre_ensemble(size)
    matrix = X @ X.conj().T
    matrix /= np.trace(matrix)

    return Density(matrix, qubits=qubits)


# TODO: Check math
def random_density_bures(qubits: Union[int, Qubits],
                         rank: int = None) -> Density:
    """
    Returns: A random Density drawn from the Bures measure.

    Args:
        qubits: A list or number of qubits.
        rank: Rank of density matrix. (Defaults to full rank)

    Ref:
        "Random Bures mixed states and the distribution of their purity",
         Osipov, Sommers, and Zyczkowski, J. Phys. A: Math. Theor. 43,
         055302 (2010). arXiv:0909.5094

    """
    N, qubits = qubits_count_tuple(qubits)
    dim = 2 ** N
    size = (dim, dim) if rank is None else (dim, rank)
    P = np.eye(dim) + unitary_ensemble(dim)
    G = complex_ginibre_ensemble(size=size)
    B = P @ G @ G.conj().T @ P.conj().T
    B /= np.trace(B)

    return Density(B, qubits=qubits)


# TESTME
def join_densities(*densities: Density) -> Density:
    """Join two mixed states into a larger qubit state"""
    vectors = [rho.vec for rho in densities]
    vec = reduce(outer_product, vectors)

    memory:  FrozenDict[str, Any] = \
        FrozenDict(ChainMap(*[rho.memory for rho in densities]))  # TESTME
    return Density(vec.tensor, vec.qubits, memory)

# fin
