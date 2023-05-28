# Copyright 2019-, Gavin E. Crooks and contributors
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
.. contents:: :local:
.. currentmodule:: quantumflow


QuantumFlow supports several different quantum operations that act upon either
pure or mixed states (or both). The four main types are Gate, which represents
the action of an operator (typically unitary) on a state; Channel, which
represents the action of a superoperator on a mixed state (used for mixed
quantum-classical dynamics); Kraus, which represents a Channel as a collection
of operators; and Circuit, which is a list of other operations that act in
sequence. Circuits can contain any instance of the abstract quantum operation
superclass, Operation, including other circuits.

Quantum operations are immutable, and transformations of these operations return
new copies.

The main types of Operation's are Gate, UnitaryGate, StdGate, Channel, Circuit,
DAGCircuit, and Pauli.

.. autoclass:: Operation
    :members:

"""

import inspect
from abc import ABC, abstractmethod
from copy import copy
from functools import total_ordering
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import scipy
from scipy.linalg import fractional_matrix_power as matpow
from scipy.linalg import logm

from . import tensors, var
from .future import cached_property
from .qubits import Qubit, Qubits
from .states import Density, State
from .tensors import QubitTensor
from .var import Variable

# standard workaround to avoid circular imports from type hints
if TYPE_CHECKING:
    from numpy.typing import ArrayLike  # pragma: no cover

    from .paulialgebra import Pauli  # pragma: no cover
    from .stdgates import StdGate  # noqa:F401  # pragma: no cover

__all__ = [
    "Operation",
    "Gate",
    "UnitaryGate",
    "Channel",
    "Unitary",
    "OPERATIONS",
    "GATES",
]


_EXCLUDED_OPERATIONS = set(
    ["Operation", "Gate", "StdGate", "StdCtrlGate", "In", "Out", "NoWire"]
)
# Names of operations to exclude from registration. Includes (effectively) abstract base
# classes and internal operations.


OPERATIONS: Dict[str, Type["Operation"]] = {}
"""All quantum operations (All non-abstract subclasses of Operation)"""

GATES: Dict[str, Type["Gate"]] = {}
"""All gates (All non-abstract subclasses of Gate)"""


OperationTV = TypeVar("OperationTV", bound="Operation")


@total_ordering
class Operation(ABC):
    """An operation on a quantum state. An element of a quantum circuit.

    Abstract Base Class for Gate, Circuit, and other quantum operations.

    Attributes:
        qubits: The qubits that this Operation acts upon.
        params: Optional keyword parameters used to create this gate
    """

    # Note: We prefix static class variables with "cv_" to avoid confusion
    # with instance variables

    __slots__ = ["_tensor", "_qubits", "_params"]

    cv_interchangeable: ClassVar[bool] = False
    """Is this a multi-qubit operation that is known to be invariant under
    permutations of qubits?"""

    cv_qubit_nb: ClassVar[int] = 0
    """The number of qubits, for operations with a fixed number of qubits"""

    cv_args: ClassVar[Union[Tuple[str, ...], Tuple]] = ()
    """The names of the parameters for this operation (For operations with a fixed
    number of float parameters)"""

    def __init_subclass__(cls) -> None:
        # Note: The __init_subclass__ initializes all subclasses of a given class.
        # see https://www.python.org/dev/peps/pep-0487/

        name = cls.__name__
        if name not in _EXCLUDED_OPERATIONS:
            OPERATIONS[name] = cls

    def __init__(
        self,
        qubits: Qubits,
        params: Optional[Sequence[Variable]] = None,
    ) -> None:
        self._qubits: Qubits = tuple(qubits)
        self._params: Tuple[Variable, ...] = ()
        if params is not None:
            self._params = tuple(params)
        self._tensor: Optional[QubitTensor] = None

        if self.cv_qubit_nb != 0:
            if self.cv_qubit_nb != len(self._qubits):
                raise ValueError(
                    "Wrong number of qubits for Operation"
                )  # pragma: no cover

    def __iter__(self) -> Iterator["Operation"]:
        yield self

    @property
    def name(self) -> str:
        """Return the name of this Operation"""
        return type(self).__name__

    @property
    def qubits(self) -> Qubits:
        """Return the total number of qubits"""
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits"""
        return len(self.qubits)

    def on(self: OperationTV, *qubits: Qubit) -> OperationTV:
        """Return a copy of this Operation with new qubits"""
        if len(qubits) != self.qubit_nb:
            raise ValueError("Wrong number of qubits")
        op = copy(self)
        op._qubits = qubits
        return op

    def rewire(self: OperationTV, labels: Dict[Qubit, Qubit]) -> OperationTV:
        """Relabel qubits and return copy of this Operation"""
        qubits = tuple(labels[q] for q in self.qubits)
        return self.on(*qubits)

    def qubit_indices(self, qubits: Qubits) -> Tuple[int, ...]:
        """Convert qubits to index positions.

        Raises:
            ValueError: If argument qubits are not found in operation's qubits
        """
        try:
            return tuple(self.qubits.index(q) for q in qubits)
        except ValueError:
            raise ValueError("Incommensurate qubits")

    @property
    def params(self) -> Tuple[Variable, ...]:
        """Return all of the parameters of this Operation"""
        return self._params

    def param(self, name: str) -> Variable:
        """Return a a named parameters of this Operation.

        Raise:
            KeyError: If unrecognized parameter name
        """
        if self.cv_args is None:
            raise KeyError("No parameters")
        try:
            idx = self.cv_args.index(name)
        except ValueError:
            raise KeyError("Unknown parameter name", name)

        return self._params[idx]

    # rename? param_asfloat? Then use where needed.
    def float_param(
        self, name: str, subs: Optional[Mapping[str, float]] = None
    ) -> float:
        """Return a a named parameters of this Operation as a float.

        Args:
            name: The name of the parameter (should be in cls.cv_args)
            subs: Symbolic substitutions to resolve symbolic Variables
        Raise:
            KeyError:  If unrecognized parameter name
            ValueError: If Variable cannot be converted to float
        """
        return var.asfloat(self.param(name), subs)

    def resolve(self: OperationTV, subs: Mapping[str, float]) -> OperationTV:
        """Resolve symbolic parameters"""
        # params = {k: var.asfloat(v, subs) for k, v in self.params.items()}
        op = copy(self)
        _params = [var.asfloat(v, subs) for v in self.params]
        op._params = tuple(_params)
        op._tensor = None
        return op

    def asgate(self) -> "Gate":
        """
        Convert this quantum operation to a gate (if possible).

        Raises:
            ValueError: If this operation cannot be converted to a Gate
        """
        raise ValueError()  # pragma: no cover

    def aschannel(self) -> "Channel":
        """Convert this quantum operation to a channel (if possible).

        Raises:
            ValueError: If this operation cannot be converted to a Channel
        """
        raise ValueError()  # pragma: no cover

    @property
    def H(self) -> "Operation":
        """Return the Hermitian conjugate of this quantum operation.

        For unitary Gates (and Circuits composed of the same) the
        Hermitian conjugate returns the inverse Gate (or Circuit)

        Raises:
            ValueError: If this operation does not support Hermitian conjugate
        """
        raise ValueError(
            "This operation does not support Hermitian conjugate"
        )  # pragma: no cover

    @property
    def tensor(self) -> QubitTensor:
        """
        Returns the tensor representation of this operation (if possible)
        """
        raise NotImplementedError()

    @property
    def tensor_diagonal(self) -> QubitTensor:
        """
        Returns the diagonal of the tensor representation of this operation
        (if possible)
        """
        raise NotImplementedError()

    def run(self, ket: State) -> State:
        """Apply the action of this operation upon a pure state"""
        raise NotImplementedError()

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this operation upon a mixed state"""
        raise NotImplementedError()

    # Make Operations sortable. (So we can use Operations in opt_einsum
    # axis labels.)
    def __lt__(self, other: Any) -> bool:
        return id(self) < id(other)

    # TODO: rename? standardize?
    def specialize(self) -> "Operation":
        """For parameterized operations, return appropriate special cases
        for particular parameters. Else return the original Operation.

              e.g. Rx(0.0, 0).specialize() -> I(0)
        """
        return self  # pragma: no cover

    def decompose(self) -> Iterator["Operation"]:
        """Decompose this operation into smaller or more standard subunits.
        If cannot be decomposed, returns self.

        Returns: An iteration of operations.
        """
        yield self  # pragma: no cover

    def _repr_png_(self) -> Optional[bytes]:
        """Jupyter/IPython rich display"""
        from .circuits import Circuit

        return Circuit(self)._repr_png_()

    def _repr_html_(self) -> Optional[str]:
        """Jupyter/IPython rich display"""
        from .circuits import Circuit

        return Circuit(self)._repr_html_()

    def _diagram_labels_(self) -> List[str]:
        """Labels for text-based circuit diagrams.


        Multi-qubit operations should either return one label per
        qubit (which are then connected with vertical lines) or a
        single label, which will be replicated onto all qubits and
        not connected with vertical lines.
        """
        N = self.qubit_nb
        labels = [self.name] * N
        if N != 1 and not self.cv_interchangeable:
            # If not interchangeable, we have to label connections
            for i in range(N):
                labels[i] = labels[i] + "_%s" % i

        return labels


# End class Operation


GateTV = TypeVar("GateTV", bound="Gate")


class Gate(Operation):
    """
    A quantum logic gate. A unitary operator that acts upon a collection
    of qubits.
    """

    cv_hermitian: ClassVar[bool] = False
    """Is this Gate know to always be hermitian?"""

    cv_tensor_structure: ClassVar[Optional[str]] = None
    """
    Is the tensor representation of this Operation known to have a particular
    structure in the computational basis?

    Options:
        identity
        diagonal
        permutation
        swap
        monomial

    A permutation matrix permutes states. It has a single '1' in each row and column.
    All other entries are zero.

    A swap is a permutation matrix that permutes qubits.

    A monomial matrix is a product of a diagonal and a permutation matrix.
    Only 1 entry in each row and column is non-zero.

    """

    def __init_subclass__(cls) -> None:
        # Note: The __init_subclass__ initializes all subclasses of a given class.
        # see https://www.python.org/dev/peps/pep-0487/

        if inspect.isabstract(cls):
            return  # pragma: no cover

        super().__init_subclass__()

        name = cls.__name__
        if name not in _EXCLUDED_OPERATIONS:
            GATES[name] = cls

    @property
    def hamiltonian(self) -> "Pauli":
        """
        Returns the Hermitian Hamiltonian of corresponding to this
        unitary operation.

        .. math::
            U = e^{-i H)

        Returns:
            A Hermitian operator represented as an element of the Pauli algebra.
        """
        # See test_gate_hamiltonians()
        from .paulialgebra import pauli_decompose_hermitian

        H = -logm(self.asoperator()) / 1.0j
        pauli = pauli_decompose_hermitian(H, self.qubits)
        return pauli

    def asgate(self: GateTV) -> GateTV:
        return self

    def aschannel(self) -> "Channel":
        """Convert a Gate into a Channel"""
        N = self.qubit_nb
        R = 4

        # TODO: As Kraus?
        tensor = np.outer(self.tensor, self.H.tensor)
        tensor = np.reshape(tensor, [2**N] * R)
        tensor = np.transpose(tensor, [0, 3, 1, 2])

        return Channel(tensor, self.qubits)

    def __pow__(self, t: float) -> "Gate":
        """Return this gate raised to the given power."""
        matrix = matpow(self.asoperator(), t)
        return UnitaryGate(matrix, self.qubits)

    def permute(self, qubits: Qubits) -> "Gate":
        """Permute the order of the qubits"""
        qubits = tuple(qubits)
        if self.qubits == qubits:
            return self
        if self.cv_interchangeable:
            return self.on(*qubits)
        tensor = tensors.permute(self.tensor, self.qubit_indices(qubits))
        return UnitaryGate(tensor, qubits)

    def asoperator(self) -> QubitTensor:
        """Return tensor with with qubit indices flattened"""
        return tensors.flatten(self.tensor, rank=2)

    @property
    @abstractmethod
    def tensor(self) -> QubitTensor:
        pass

    @cached_property
    def tensor_diagonal(self) -> QubitTensor:
        """
        Returns the diagonal of the tensor representation of this operation
        (if possible)
        """
        return tensors.asqutensor(np.diag(self.asoperator()))

    def su(self) -> "UnitaryGate":
        """Convert gate tensor to the special unitary group."""
        rank = 2**self.qubit_nb
        U = self.asoperator()
        U /= np.linalg.det(U) ** (1 / rank)
        return UnitaryGate(U, self.qubits)

    @property
    def H(self) -> "Gate":
        return UnitaryGate(self.asoperator().conj().T, self.qubits)

    def __matmul__(self, other: "Gate") -> "Gate":
        """Apply the action of this gate upon another gate,
        self_gate @ other_gate (Note time runs right to left with
        matrix notation)

        Note that the gates must act on the same qubits.
        When gates don't act on the same qubits, use
        Circuit(self_gate, other_gate).asgate() instead.
        """
        if not isinstance(other, Gate):
            raise NotImplementedError()
        gate0 = self
        gate1 = other
        indices = gate1.qubit_indices(gate0.qubits)
        tensor = tensors.tensormul(gate0.tensor, gate1.tensor, tuple(indices))
        return UnitaryGate(tensor, gate1.qubits)

    def run(self, ket: State) -> State:
        """Apply the action of this gate upon a state"""
        qubits = self.qubits
        indices = ket.qubit_indices(qubits)

        if self.cv_tensor_structure == "identity":
            return ket
        elif self.cv_tensor_structure == "diagonal":
            tensor = tensors.tensormul_diagonal(
                self.tensor_diagonal, ket.tensor, tuple(indices)
            )
            return State(tensor, ket.qubits, ket.memory)

        tensor = tensors.tensormul(
            self.tensor,
            ket.tensor,
            tuple(indices),
        )
        return State(tensor, ket.qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this gate upon a density"""
        # TODO: implement without explicit channel creation? With Kraus?
        chan = self.aschannel()
        return chan.evolve(rho)

    def specialize(self) -> "Gate":
        return self

    def __str__(self) -> str:
        def _param_format(obj: Any) -> str:
            if isinstance(obj, float):
                try:
                    return str(var.asexpression(obj))
                except ValueError:
                    return f"{obj}"
            return str(obj)

        fqubits = " " + " ".join([str(qubit) for qubit in self.qubits])

        if self.params:
            fparams = "(" + ", ".join(_param_format(p) for p in self.params) + ")"
        else:
            fparams = ""

        return f"{self.name}{fparams}{fqubits}"

    # TODO: Move logic elsewhere
    def decompose(self) -> Iterator["StdGate"]:
        from .translate import TRANSLATIONS, translation_source_gate

        # Terminal gates  # FIXME: Move elsewhere?
        if self.name in ("I", "Ph", "X", "Y", "Z", "XPow", "YPow", "ZPow", "CNot"):
            yield self  # type: ignore
            return

        # Reversed so we favor translations added latter.
        for trans in reversed(TRANSLATIONS):
            from_gate = translation_source_gate(trans)
            if isinstance(self, from_gate):
                yield from trans(self)
                return

        # If we don't know how to perform an analytic translation, resort to a
        # numerical decomposition. Will fail for gates with symbolic parameters.
        from .decompositions import quantum_shannon_decomposition

        circ = quantum_shannon_decomposition(self)
        for gate in circ:
            yield from gate.decompose()  # type: ignore


# End class Gate


class UnitaryGate(Gate):
    """
    A quantum logic gate specified by an explicit unitary operator.
    """

    def __init__(self, tensor: "ArrayLike", qubits: Qubits) -> None:
        tensor = tensors.asqutensor(tensor)

        N = np.ndim(tensor) // 2
        if len(tuple(qubits)) != N:
            raise ValueError("Wrong number of qubits for tensor")

        super().__init__(qubits=qubits)
        self._tensor = tensor

    @classmethod
    def from_gate(cls, gate: Gate) -> "UnitaryGate":
        return cls(gate.tensor, gate.qubits)

    @classmethod
    def from_hamiltonian(cls, hamiltonian: "Pauli", qubits: Qubits) -> "UnitaryGate":
        """Create a Unitary gate U from a Pauli operator H, U = exp(-i H)"""
        op = hamiltonian.asoperator(qubits)
        U = scipy.linalg.expm(-1j * op)
        return cls(U, qubits)

    @cached_property
    def tensor(self) -> QubitTensor:
        """Returns the tensor representation of gate operator"""
        if self._tensor is None:
            raise ValueError("No tensor representation")
        return self._tensor


# End class UnitaryGate

# Deprecated. Renamed to UnitaryGate
Unitary = UnitaryGate


# FIXME, more like UnitaryGate, with a 'Channel' superclass that's like Gate
class Channel(Operation):
    """A quantum channel"""

    def __init__(
        self,
        tensor: "ArrayLike",
        qubits: Qubits,
        params: Optional[Sequence[var.Variable]] = None,
        name: Optional[str] = None,  # FIXME
    ) -> None:
        tensor = tensors.asqutensor(tensor)

        N = np.ndim(tensor) // 4
        if len(qubits) != N:
            raise ValueError("Wrong number of qubits for tensor")

        super().__init__(qubits=qubits, params=params)
        self._tensor = tensor
        self._name = type(self).__name__ if name is None else name

    @cached_property
    def tensor(self) -> QubitTensor:
        """Return the tensor representation of the channel's superoperator"""
        if self._tensor is None:
            raise ValueError("No tensor representation")
        return self._tensor

    @property
    def name(self) -> str:
        return self._name

    def permute(self, qubits: Qubits) -> "Channel":
        """Return a copy of this channel with qubits in new order"""
        if self.qubits == qubits:
            return self
        # if self.cv_interchangeable:   # TODO
        #     return self.on(*qubits)
        tensor = tensors.permute(self.tensor, self.qubit_indices(qubits))
        return Channel(tensor, qubits=qubits)

    @property
    def H(self) -> "Channel":
        return Channel(tensor=tensors.conj_transpose(self.tensor), qubits=self.qubits)

    # TESTME
    @property
    def sharp(self) -> "Channel":
        r"""Return the 'sharp' transpose of the superoperator.

        The transpose :math:`S^\#` switches the two covariant (bra)
        indices of the superoperator. (Which in our representation
        are the 2nd and 3rd super-indices)

        If :math:`S^\#` is Hermitian, then :math:`S` is a Hermitian-map
        (i.e. transforms Hermitian operators to Hermitian operators)

        Flattening the :math:`S^\#` superoperator to a matrix gives
        the Choi matrix representation. (See channel.choi())
        """

        N = self.qubit_nb

        # TODO: Use tensor_transpose, or remove tensor_transpose
        tensor = self.tensor
        tensor = np.reshape(tensor, [2**N] * 4)
        tensor = np.transpose(tensor, (0, 2, 1, 3))
        tensor = np.reshape(tensor, [2] * 4 * N)
        return Channel(tensor, self.qubits)

    def choi(self) -> QubitTensor:
        """Return the Choi matrix representation of this super
        operator"""
        # Put superop axes in the order [out_ket, in_bra, out_bra, in_ket]
        # and reshape to matrix
        N = self.qubit_nb
        return np.reshape(self.sharp.tensor, [2 ** (N * 2)] * 2)

    @classmethod
    def from_choi(cls, tensor: "ArrayLike", qubits: Qubits) -> "Channel":
        """Return a Channel from a Choi matrix"""
        return cls(tensor, qubits).sharp

    # TESTME
    # FIXME: Can't be right, same as choi?
    def chi(self) -> QubitTensor:
        """Return the chi (or process) matrix representation of this
        superoperator"""
        N = self.qubit_nb
        return np.reshape(self.sharp.tensor, [2 ** (N * 2)] * 2)

    def run(self, ket: State) -> "State":
        raise TypeError()  # Not possible in general

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this channel upon a density"""
        N = rho.qubit_nb
        qubits = rho.qubits

        indices = list([qubits.index(q) for q in self.qubits]) + list(
            [qubits.index(q) + N for q in self.qubits]
        )

        tensor = tensors.tensormul(self.tensor, rho.tensor, tuple(indices))
        return Density(tensor, qubits, rho.memory)

    def asgate(self) -> "Gate":
        raise TypeError()  # Not possible in general

    def aschannel(self) -> "Channel":
        return self

    def __matmul__(self, other: "Channel") -> "Channel":
        """Apply the action of this channel upon another channel,
        self_chan @ other_chan (Note time runs right to left with
        matrix notation)

        Note that the channels must act on the same qubits.
        When gates don't act on the same qubits, use
        Circuit(other_chan, self_chan).aschannel() instead.
        """
        if not isinstance(other, Channel):
            raise NotImplementedError()
        chan0 = self
        chan1 = other
        N = chan1.qubit_nb
        qubits = chan1.qubits
        indices = list([chan1.qubits.index(q) for q in chan0.qubits]) + list(
            [chan1.qubits.index(q) + N for q in chan0.qubits]
        )

        tensor = tensors.tensormul(chan0.tensor, chan1.tensor, tuple(indices))

        return Channel(tensor, qubits)

    # TESTME
    def trace(self) -> float:
        """Return the trace of this super operator"""
        return tensors.trace(self.tensor, rank=4)

    # TESTME # TODO
    # def partial_trace(self, qubits: Qubits) -> "Channel":
    #     """Return the partial trace over the specified qubits"""
    #     vec = tensors.partial_trace(self.tensor, qubits, rank=4)
    #     return Channel(vec.tensor, vec.qubits)


# end class Channel


# fin
