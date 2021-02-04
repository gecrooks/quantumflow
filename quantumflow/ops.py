# Copyright 2020-, Gavin E. Crooks and contributors
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
new copies. (An exception is the composite operation DAGCircuit.)

The main types of Operation's are Gate, Unitary, StdGate, Channel, Circuit, DAGCircuit,
and Pauli.

.. autoclass:: Operation
    :members:

"""

from abc import ABC  # Abstract Base Class
from copy import copy
from functools import total_ordering
from typing import (
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
)

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import fractional_matrix_power as matpow
from scipy.linalg import logm

from . import tensors, utils, var
from .qubits import Qubit, Qubits
from .states import Density, State
from .tensors import QubitTensor
from .var import Variable

__all__ = ["Operation", "Gate", "StdGate", "Unitary", "Channel"]


OperationType = TypeVar("OperationType", bound="Operation")
"""Generic type annotations for subtypes of Operation"""

GateType = TypeVar("GateType", bound="Gate")
"""Generic type annotations for subtypes of Gate"""


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

    cv_qubit_nb: ClassVar[int] = None
    """The number of qubits, for operations with a fixed number of qubits"""

    cv_args: Tuple[str, ...] = ()
    """The names of the parameters for this operation"""

    _diagram_labels: ClassVar[Optional[Sequence[str]]] = None
    """Override default labels for drawing text circuit diagrams.
    See visualizations.circuit_to_diagram()"""

    _diagram_noline: ClassVar[bool] = False
    """Override default to not draw a line between qubit wires for multi-qubit
    operations. See visualizations.circuit_to_diagram()"""

    def __init__(
        self,
        qubits: Qubits,
        params: Sequence[Variable] = None,
    ) -> None:
        self._qubits: Qubits = tuple(qubits)
        self._params: Tuple[Variable, ...] = ()
        if params is not None:
            self._params = tuple(params)
        self._tensor: QubitTensor = None

        if self.cv_qubit_nb is not None:
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

    def on(self: OperationType, *qubits: Qubit) -> OperationType:
        """Return a copy of this Operation with new qubits"""
        if len(qubits) != self.qubit_nb:
            raise ValueError("Wrong number of qubits")
        op = copy(self)
        op._qubits = qubits
        return op

    def rewire(self: OperationType, labels: Dict[Qubit, Qubit]) -> OperationType:
        """Relabel qubits and return copy of this Operation"""
        qubits = tuple(labels[q] for q in self.qubits)
        return self.on(*qubits)

    def qubit_indices(self, qubits: Qubits) -> List[int]:
        """Convert qubits to index positions.

        Raises:
            ValueError: If argument qubits are not found in operation's qubits
        """
        return [self.qubits.index(q) for q in qubits]

    @property
    def params(self) -> Tuple[Variable, ...]:
        """Return all of the parameters of this Operation"""
        return self._params

    def param(self, name: str) -> Variable:
        """Return a a named parameters of this Operation.

        Raise:
            KeyError: If unrecognized parameter name
        """
        try:
            idx = self.cv_args.index(name)
        except ValueError:
            raise KeyError("Unknown parameter name", name)

        return self._params[idx]

    def float_param(self, name: str, subs: Mapping[str, float] = None) -> float:
        """Return a a named parameters of this Operation as a float.

        Args:
            name: The name of the parameter (should be in cls.cv_args)
            subs: Symbolic substitutions to resolve symbolic Variables
        Raise:
            KeyError:  If unrecognized parameter name
            ValueError: If Variable cannot be converted to float
        """
        return var.asfloat(self.param(name), subs)

    def resolve(self, subs: Mapping[str, float]) -> "Operation":
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

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        """
        Returns the tensor representation of this operation (if possible)
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


# End class Operation


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
        monomial

    A permutation matrix permutes states. It has a single '1' in each row and column.
    All other entries are zero.

    A monomial matrix is a product of a diagonal and a permutation matrix.
    Only 1 entry in each row and column is non-zero.

    """
    # TODO: Add "swap" t roster of tensor structures

    # Note: Circular import hell
    from .paulialgebra import Pauli

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

    def asgate(self: GateType) -> GateType:
        return self

    def aschannel(self) -> "Channel":
        """Convert a Gate into a Channel"""
        N = self.qubit_nb
        R = 4

        # TODO: As Kraus?
        tensor = np.outer(self.tensor, self.H.tensor)
        tensor = np.reshape(tensor, [2 ** N] * R)
        tensor = np.transpose(tensor, [0, 3, 1, 2])

        return Channel(tensor, self.qubits)

    def __pow__(self, t: float) -> "Gate":
        """Return this gate raised to the given power."""
        matrix = matpow(self.asoperator(), t)
        return Unitary(matrix, self.qubits)

    def permute(self, qubits: Qubits) -> "Gate":
        """Permute the order of the qubits"""
        if self.qubits == qubits:
            return self
        if self.cv_interchangeable:
            return self.on(*qubits)
        tensor = tensors.permute(self.tensor, self.qubit_indices(qubits))
        return Unitary(tensor, qubits)

    def asoperator(self) -> QubitTensor:
        """Return tensor with with qubit indices flattened"""
        return tensors.flatten(self.tensor, rank=2)
        # N = self.qubit_nb
        # return np.reshape(self.tensor, [2**N] * 2)

    def su(self) -> "Unitary":
        """Convert gate tensor to the special unitary group."""
        rank = 2 ** self.qubit_nb
        U = self.asoperator()
        U /= np.linalg.det(U) ** (1 / rank)
        return Unitary(U, self.qubits)

    @property
    def H(self) -> "Gate":
        return Unitary(self.asoperator().conj().T, self.qubits)

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
        return Unitary(tensor, gate1.qubits)

    def run(self, ket: State) -> State:
        """Apply the action of this gate upon a state"""
        if self.cv_tensor_structure == "identity":
            return ket

        qubits = self.qubits
        indices = ket.qubit_indices(qubits)
        tensor = tensors.tensormul(
            self.tensor,
            ket.tensor,
            tuple(indices),
            self.cv_tensor_structure == "diagonal",
        )
        return State(tensor, ket.qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this gate upon a density"""
        # TODO: implement without explicit channel creation? With Kraus?
        chan = self.aschannel()
        return chan.evolve(rho)

    def specialize(self) -> "Gate":
        return self


# End class Gate


class Unitary(Gate):
    """
    A quantum logic gate specified by an explicit unitary operator.
    """

    def __init__(self, tensor: ArrayLike, qubits: Qubits) -> None:

        tensor = tensors.asqutensor(tensor)

        N = np.ndim(tensor) // 2

        if len(tuple(qubits)) != N:
            raise ValueError("Wrong number of qubits for tensor")

        super().__init__(qubits=qubits)
        self._tensor = tensor

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        """Returns the tensor representation of gate operator"""
        return self._tensor


# End class Unitary


# TODO: Move?
class StdGate(Gate):
    """
    A standard gate. Standard gates have a name, a fixed number of real
    parameters, and act upon a fixed number of qubits.

    e.g. Rx(theta, q0), CNot(q0, q1), Can(tx, ty, tz, q0, q1, q2)

    In the argument list, parameters are first, then qubits. Parameters
    have type Variable (either a concrete floating point number, or a symbolic
    expression), and qubits have type Qubit (Any hashable python type).
    """

    cv_stdgates: Dict[str, Type["StdGate"]] = {}
    """List of all StdGate subclasses"""

    def __init_subclass__(cls) -> None:
        # Note: The __init_subclass__ initializes all subclasses of a given class.
        # see https://www.python.org/dev/peps/pep-0487/

        # Parse the Gate arguments and number of qubits from the arguments to __init__
        # Convention is that qubit names start with "q", but arguments do not.
        names = getattr(cls, "__init__").__annotations__.keys()
        args = tuple(s for s in names if s[0] != "q" and s != "return")
        qubit_nb = len(names) - len(args)
        if "return" in names:
            # For unknown reasons, "return" is often, but not always in names.
            qubit_nb -= 1

        cls.cv_args = args
        cls.cv_qubit_nb = qubit_nb

        cls.cv_stdgates[cls.__name__] = cls  # Subclass registration

    def __repr__(self) -> str:
        args: List[str] = []
        args.extend(str(p) for p in self.params)
        args.extend(str(qubit) for qubit in self.qubits)
        fargs = ", ".join(args)

        return f"{self.name}({fargs})"

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

    def decompose(self) -> Iterator["StdGate"]:
        from .translate import TRANSLATORS, translation_source_gate

        # Terminal gates
        if self.name in ("I", "Ph", "X", "Y", "Z", "XPow", "YPow", "ZPow", "CNot"):
            yield self
            return

        for trans in TRANSLATORS.values():
            from_gate = translation_source_gate(trans)
            if isinstance(self, from_gate):
                yield from trans(self)
                return

        yield self  # fall back  # pragma: no cover


# End class StdGate


class Channel(Operation):
    """A quantum channel"""

    def __init__(
        self,
        tensor: ArrayLike,
        qubits: Qubits,
        params: Sequence[var.Variable] = None,
        name: str = None,  # FIXME
    ) -> None:

        tensor = tensors.asqutensor(tensor)

        N = np.ndim(tensor) // 4
        if len(qubits) != N:
            raise ValueError("Wrong number of qubits for tensor")

        super().__init__(qubits=qubits, params=params)
        self._tensor = tensor
        self._name = type(self).__name__ if name is None else name

    @utils.cached_property
    def tensor(self) -> QubitTensor:
        """Return the tensor representation of the channel's superoperator"""
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
        tensor = np.reshape(tensor, [2 ** N] * 4)
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
    def from_choi(cls, tensor: ArrayLike, qubits: Qubits) -> "Channel":
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
