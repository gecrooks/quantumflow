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

from abc import ABC
from math import sqrt
from typing import Any, Dict, List, Mapping, TextIO, Tuple, TypeVar, Union

import numpy as np
import opt_einsum
from numpy.typing import ArrayLike

from . import tensors, utils
from .qubits import Qubit, Qubits
from .tensors import QubitTensor

__all__ = [
    "State",
    "join_states",
    "print_probabilities",
    "print_state",
    "random_state",
    "ghz_state",
    "w_state",
    "zero_state",
    "Density",
    "mixed_density",
    "random_density",
    "join_densities",
]


QuantumStateType = TypeVar("QuantumStateType", bound="QuantumState")
"""TypeVar for quantum states"""


class QuantumState(ABC):
    def __init__(
        self, tensor: ArrayLike, qubits: Qubits, memory: Mapping = None
    ) -> None:
        """
        Abstract base class for representations of a quantum state.

        Args:
            tensor: A vector or tensor of state amplitudes
            qubits: A sequence of qubit names.
                (Defaults to integer indices, e.g. [0, 1, 2] for 3 qubits)
            memory: Classical data storage. Stored as an immutable dictionary.
        """

        self._tensor = np.asarray(tensor)
        self._qubits = tuple(qubits)

        if memory is None:
            self.memory: utils.FrozenDict = utils.FrozenDict()
        elif isinstance(memory, utils.FrozenDict):
            self.memory = memory
        else:
            self.memory = utils.FrozenDict(memory)

    @property
    def tensor(self) -> QubitTensor:
        """
        Returns the tensor representation of this state (if possible).
        """
        if self._tensor is None:
            raise ValueError("Cannot access quantum state")  # pragma: no cover
        return self._tensor

    @property
    def qubits(self) -> Qubits:
        """Return qubit labels of this state"""
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits"""
        return len(self._qubits)

    def replace(
        self: QuantumStateType,
        *,
        tensor: ArrayLike = None,
        qubits: Qubits = None,
        memory: Mapping = None,
    ) -> QuantumStateType:
        """
        Creates a copy of this state, replacing the fields specified.
        """
        # Interface similar to dataclasses.replace
        tensor = self.tensor if tensor is None else tensor
        qubits = self.qubits if qubits is None else qubits
        memory = self.memory if memory is None else memory
        return type(self)(tensor, qubits, memory)

    def store(self: QuantumStateType, *args: Any, **kwargs: Any) -> QuantumStateType:
        """Update information in classical memory and return a new State."""
        mem = self.memory.update(*args, **kwargs)
        return self.replace(memory=mem)

    # TESTME
    def on(self: QuantumStateType, *qubits: Qubit) -> QuantumStateType:
        """Return a copy of this State with new qubits"""
        return self.replace(qubits=qubits)

    # TESTME
    def rewire(self: QuantumStateType, labels: Dict[Qubit, Qubit]) -> QuantumStateType:
        """Relabel qubits and return copy of this Operation"""
        qubits = tuple(labels[q] for q in self.qubits)
        return self.on(*qubits)

    def permute(self: QuantumStateType, qubits: Qubits = None) -> QuantumStateType:
        """Return a copy of this state with state tensor transposed to
        put qubits in the given order. If an explicit qubit
        ordering isn't supplied, we put qubits in sorted order.
        """
        if qubits is None:
            qubits = sorted(self.qubits)
        tensor = tensors.permute(self.tensor, self.qubit_indices(qubits))
        return self.replace(tensor=tensor, qubits=qubits)

    def qubit_indices(self, qubits: Qubits) -> List[int]:
        """Convert qubits to index positions.

        Raises:
            ValueError: If argument qubits are not found in state qubits
        """
        return [self.qubits.index(q) for q in qubits]

    def norm(self) -> QubitTensor:
        """Return the state vector norm"""
        return tensors.norm(self.tensor)


class State(QuantumState):
    """The quantum state of a collection of qubits.

    Note that memory usage grows exponentially with the number of qubits.
    (16*2^N bytes for N qubits)

    """

    def __init__(
        self, tensor: ArrayLike, qubits: Qubits = None, memory: Mapping = None
    ) -> None:
        """Create a new State from a tensor of qubit amplitudes

        Args:
            tensor: A vector or tensor of state amplitudes
            qubits: A sequence of qubit names.
                (Defaults to integer indices, e.g. [0, 1, 2] for 3 qubits)
            memory: Classical data storage. Stored as an immutable dictionary.
        """
        tensor = tensors.asqutensor(tensor)

        N = np.ndim(tensor)
        if qubits is None:
            qubits = range(N)
        elif len(qubits) != N:
            raise ValueError("Wrong number of qubits for tensor")

        super().__init__(tensor, qubits, memory)

    def normalize(self) -> "State":
        """Normalize the state"""
        tensor = self.tensor / np.sqrt(self.norm())
        return State(tensor, self.qubits, self.memory)

    def probabilities(self) -> QubitTensor:
        """
        Returns:
            The state probabilities
        """
        value = np.absolute(self.tensor)
        return value * value

    def sample(self, trials: int) -> np.ndarray:
        """Measure the state in the computational basis the the given number
        of trials, and return the counts of each output configuration.
        """
        # TODO: Can we do this within backend?
        probs = np.real(self.probabilities())
        res = np.random.multinomial(trials, probs.ravel())
        res = res.reshape(probs.shape)
        return res

    def expectation(self, diag_hermitian: ArrayLike, trials: int = None) -> QubitTensor:
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
            probs = np.real(tensors.asqutensor(self.sample(trials) / trials))

        diag_hermitian = tensors.asqutensor(diag_hermitian)
        return np.sum(np.real(diag_hermitian) * probs)

    def measure(self) -> np.ndarray:
        """Measure the state in the computational basis.

        Returns:
            A [2]*bits array of qubit states, either 0 or 1
        """
        probs = np.real(self.probabilities())
        indices = np.asarray(list(np.ndindex(*[2] * self.qubit_nb)))
        res = np.random.choice(probs.size, p=probs.ravel())
        res = indices[res]
        return res

    def asdensity(self, qubits: Qubits = None) -> "Density":
        """Convert a pure state to a density matrix.

        Args:
            qubits: The qubit subspace. If not given return the density
                    matrix for all the qubits (which can take a lot of memory!)
        """
        N = self.qubit_nb

        if qubits is None:
            qubits = self.qubits

        contract_qubits: List[Qubit] = list(self.qubits)
        for q in qubits:
            contract_qubits.remove(q)

        left_subs = np.asarray(list(range(0, N)))
        right_subs = np.asarray(list(range(N, 2 * N)))

        indices = [self.qubits.index(qubit) for qubit in contract_qubits]
        for idx in indices:
            right_subs[idx] = left_subs[idx]

        left_tensor = self.tensor
        right_tensor = np.conj(left_tensor)

        tensor = opt_einsum.contract(left_tensor, left_subs, right_tensor, right_subs)

        return Density(tensor, qubits, self.memory)

    def __str__(self) -> str:
        state = self.tensor
        s = []
        count = 0
        MAX_ELEMENTS = 64
        for index, amplitude in np.ndenumerate(state):
            if not np.isclose(amplitude, 0.0):
                ket = "|" + "".join([str(n) for n in index]) + ">"
                s.append(f"({amplitude.real:0.04g}" f"{amplitude.imag:+0.04g}i) {ket}")
                count += 1
                if count > MAX_ELEMENTS:
                    s.append("...")
                    break
        return " + ".join(s)


# End class State


def zero_state(qubits: Union[int, Qubits]) -> State:
    """Return the all-zero state on N qubits"""
    N, qubits = _qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    ket[(0,) * N] = 1
    return State(ket, qubits)


def w_state(qubits: Union[int, Qubits]) -> State:
    """Return a W state on N qubits"""
    N, qubits = _qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    for n in range(N):
        idx = np.zeros(shape=N, dtype=int)
        idx[n] += 1
        ket[tuple(idx)] = 1 / sqrt(N)
    return State(ket, qubits)


def ghz_state(qubits: Union[int, Qubits]) -> State:
    """Return a GHZ state on N qubits"""
    N, qubits = _qubits_count_tuple(qubits)
    ket = np.zeros(shape=[2] * N)
    ket[(0,) * N] = 1 / sqrt(2)
    ket[(1,) * N] = 1 / sqrt(2)
    return State(ket, qubits)


def random_state(qubits: Union[int, Qubits]) -> State:
    """Return a random state from the space of N qubits"""
    N, qubits = _qubits_count_tuple(qubits)
    ket = np.random.normal(size=([2] * N)) + 1j * np.random.normal(size=([2] * N))
    return State(ket, qubits).normalize()


# == Actions on States ==


def join_states(ket0: State, ket1: State) -> State:
    """Join two mixed states into a larger qubit state"""
    qubits = tuple(ket0.qubits) + tuple(ket1.qubits)
    tensor = tensors.outer(ket0.tensor, ket1.tensor, rank=1)
    memory = ket0.memory.update(ket1.memory)  # TESTME
    return State(tensor, qubits, memory)


# = Output =

# TODO: clean up. Move to visualization?


def print_state(state: State, file: TextIO = None) -> None:
    """Print a state vector"""
    for index, amplitude in np.ndenumerate(state.tensor):
        ket = "".join([str(n) for n in index])
        print(ket, ":", amplitude, file=file)


# TODO: Should work for density also. Check
def print_probabilities(state: State, ndigits: int = 4, file: TextIO = None) -> None:
    """
    Pretty print state probabilities.

    Args:
        state:
        ndigits: Number of digits of accuracy
        file: Output stream (Defaults to stdout)
    """
    prob = state.probabilities()
    for index, prob in np.ndenumerate(prob):
        p = round(float(prob), ndigits)
        if prob == 0.0:
            continue
        ket = "".join([str(n) for n in index])
        print(ket, ":", p, file=file)


# --  Mixed Quantum States --


class Density(QuantumState):
    """A density matrix representation of a mixed quantum state"""

    def __init__(
        self, tensor: ArrayLike, qubits: Qubits = None, memory: Mapping = None
    ) -> None:
        tensor = tensors.asqutensor(tensor)

        N = np.ndim(tensor) // 2
        if qubits is None:
            qubits = range(N)
        elif len(qubits) != N:
            raise ValueError("Wrong number of qubits for tensor")

        super().__init__(tensor, qubits, memory)

    def trace(self) -> float:
        """Return the trace of this density operator"""
        return tensors.trace(self.tensor, rank=2)

    def normalize(self) -> "Density":
        """Normalize state"""
        tensor = self.tensor / self.trace()
        return self.replace(tensor=tensor)

    # TESTME
    def probabilities(self) -> QubitTensor:
        """Returns: The state probabilities """
        prob = tensors.diag(self.tensor)
        return prob

    def asoperator(self) -> QubitTensor:
        """Return the density matrix as a square array"""
        return tensors.flatten(self.tensor, rank=2)

    def asdensity(self, qubits: Qubits = None) -> "Density":
        if qubits is None:
            return self
        tensor = tensors.partial_trace(self.tensor, self.qubit_indices(qubits))
        return Density(tensor, qubits, self.memory)


def mixed_density(qubits: Union[int, Qubits]) -> Density:
    """Returns the completely mixed density matrix"""
    N, qubits = _qubits_count_tuple(qubits)
    matrix = np.eye(2 ** N) / 2 ** N
    return Density(matrix, qubits)


def random_density(
    qubits: Union[int, Qubits], rank: int = None, ensemble: str = "Hilbert–Schmidt"
) -> Density:
    """
    Returns: A randomly sampled Density

    Args:
        qubits: A list or number of qubits.
        rank: Rank of density matrix. (Defaults to full rank)
        ensemble: Either 'Hilbert–Schmidt' (default) or 'Burr'

    Ref:
        - "Induced.info in the space of mixed quantum states" Karol
          Zyczkowski, Hans-Juergen Sommers, J. Phys. A34, 7111-7125 (2001)
          arXiv:quant-ph/0012101
        - "Random Bures mixed states and the distribution of their purity",
          Osipov, Sommers, and Zyczkowski, J. Phys. A: Math. Theor. 43,
          055302 (2010). arXiv:0909.5094
    """
    if ensemble == "Hilbert–Schmidt":
        return random_density_hs(qubits, rank)
    elif ensemble == "Bures":
        return random_density_bures(qubits, rank)
    raise ValueError(
        "Unknown ensemble. " "Valid Options are 'Hilbert–Schmidt' or 'Bures"
    )


# TODO: Check math
def random_density_hs(qubits: Union[int, Qubits], rank: int = None) -> Density:
    """
    Returns: A randomly sampled Density from the Hilbert–Schmidt
                ensemble of quantum states.

    Args:
        qubits: A list or number of qubits.
        rank: Rank of density matrix. (Defaults to full rank)

    Ref:
        "Induced.info in the space of mixed quantum states" Karol
        Zyczkowski, Hans-Juergen Sommers, J. Phys. A34, 7111-7125 (2001)
        arXiv:quant-ph/0012101
    """
    N, qubits = _qubits_count_tuple(qubits)
    size = (2 ** N, 2 ** N) if rank is None else (2 ** N, rank)

    X = utils.complex_ginibre_ensemble(size)
    matrix = X @ X.conj().T
    matrix /= np.trace(matrix)

    return Density(matrix, qubits=qubits)


# TODO: Check math
def random_density_bures(qubits: Union[int, Qubits], rank: int = None) -> Density:
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
    N, qubits = _qubits_count_tuple(qubits)
    dim = 2 ** N
    size = (dim, dim) if rank is None else (dim, rank)
    P = np.eye(dim) + utils.unitary_ensemble(dim)
    G = utils.complex_ginibre_ensemble(size=size)
    B = P @ G @ G.conj().T @ P.conj().T
    B /= np.trace(B)

    return Density(B, qubits=qubits)


def join_densities(rho0: Density, rho1: Density) -> Density:
    """Join two mixed states into a larger qubit state"""
    qubits = tuple(rho0.qubits) + tuple(rho1.qubits)
    tensor = tensors.outer(rho0.tensor, rho1.tensor, rank=2)
    memory = rho0.memory.update(rho1.memory)  # TESTME
    return Density(tensor, qubits, memory)


def _qubits_count_tuple(qubits: Union[int, Qubits]) -> Tuple[int, Qubits]:
    """Utility method for unraveling 'qubits: Union[int, Qubits]' arguments"""
    if isinstance(qubits, int):
        return qubits, tuple(range(qubits))
    return len(qubits), qubits


# fin
