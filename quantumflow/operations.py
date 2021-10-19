# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
TODO
"""

import copy
import enum
import inspect
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Iterator,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import sympy as sym

from .config import quantum_dtype
from .states import (
    Addr,
    Addrs,
    QuantumStateType,
    Qubit,
    Qubits,
    State,
    Variable,
    Variables,
)
from .utils.math import tensormul


class OperatorStructure(enum.Enum):
    """
    An enumeration of possible structures the operator of a gate can take.
    """

    identity = enum.auto()
    """Identity matrix. A diagonal matrix with '1' in each diagonal position"""

    diagonal = enum.auto()
    """An operator where all off diagonal entires are zero."""

    permutation = enum.auto()
    """A permutation matrix has a single '1' in each row and column. All other entries
    are zero. Such an operator represents a permutation of states."""

    swap = enum.auto()
    """A swap is a permutation matrix that permutes qubits."""

    monomial = enum.auto()
    """A monomial matrix is a product of a diagonal and a permutation matrix.
    Only 1 entry in each row and column is non-zero."""

    unstructured = enum.auto()
    """Matrix has no obvious structure"""


# end class OperatorStructure


OperationType = TypeVar("OperationType", bound="QuantumOperation")
"""Generic type annotations for subtypes of QuantumOperation"""


GateType = TypeVar("GateType", bound="QuantumGate")
"""Generic type annotations for subtypes of QuantumGate"""

StdGateType = TypeVar("StdGateType", bound="QuantumStdGate")
"""Generic type annotations for subtypes of QuantumStdGate"""

CompositeType = TypeVar("CompositeType", bound="QuantumComposite")
"""Generic type annotations for subtypes of QuantumComposite"""


OPERATIONS: Set[Type["QuantumOperation"]] = set()
"""All quantum operations (All concrete subclasses of QuantumOperation)"""

GATES: Set[Type["QuantumGate"]] = set()
"""All gates (All concrete subclasses of QuantumGate)"""

STDGATES: Set[Type["QuantumStdGate"]] = set()
"""All standard gates (All concrete subclasses of QuantumStdGate)"""

STDCTRLGATES: Set[Type["QuantumStdCtrlGate"]] = set()
"""All standard controlled gates (All concrete subclasses of QuantumStdCtrlGate)"""


class QuantumOperation(ABC):
    _cv_collections: ClassVar[Tuple[Set, ...]] = (OPERATIONS,)
    """List of collections to add subclasses to."""

    _qubits: Qubits = ()
    _addrs: Addrs = ()
    name: str

    def __init_subclass__(cls) -> None:
        name = cls.__name__
        cls.name = name

        # Subclass registration
        if not inspect.isabstract(cls):
            for collection in cls._cv_collections:
                collection.add(cls)

    def __init__(self, qubits: Qubits, addrs: Addrs = ()) -> None:
        self._qubits = tuple(qubits)
        self._addrs = tuple(addrs)

    @abstractmethod
    def asgate(self) -> "QuantumGate":
        """
        Convert this quantum operation into a gate (if possible).

        Raises:
            ValueError: If this operation cannot be converted to a Gate.
        """
        pass

    @property
    def addrs(self) -> Addrs:
        """Return the addresses of classical data that this operation acts upon."""
        return self._addrs

    @property
    def addrs_nb(self) -> int:
        """Return the total number of addresses."""
        return len(self.addrs)

    @property
    @abstractmethod
    def H(self) -> "QuantumOperation":
        """Return the Hermitian conjugate of this quantum operation. For unitary Gates
        (and Circuits composed of the same) the Hermitian conjugate returns the inverse
        Gate (or Circuit).

        Raises:
            ValueError: If this operation does not support Hermitian conjugation.
        """
        pass

    def on(self: OperationType, qubits: Qubits) -> "OperationType":
        # DOCME
        qubits = tuple(qubits)
        if len(qubits) != self.qubit_nb:
            raise ValueError("Wrong number of qubits")
        qubit_map = dict(zip(self.qubits, qubits))
        return self.relabel(qubit_map, addr_map=None)

    # DOCME
    @abstractmethod
    def relabel(
        self: OperationType,
        qubit_map: Dict[Qubit, Qubit],
        addr_map: Dict[Addr, Addr],
    ) -> "OperationType":
        # DOCME
        pass

    @abstractmethod
    def run(self, state: QuantumStateType) -> QuantumStateType:
        raise NotImplementedError

    @property
    def qubits(self) -> Qubits:
        """Return the Qubits that this operation acts upon."""
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits."""
        return len(self.qubits)

    def __iter__(self) -> Iterator["QuantumOperation"]:
        yield self


# end class QuantumOperation


class QuantumGate(QuantumOperation):
    _cv_collections = QuantumOperation._cv_collections + (GATES,)

    cv_interchangable: ClassVar[bool] = False

    cv_hermitian: ClassVar[bool] = False
    """Is this gate's operator known to be always hermitian?"""

    cv_operator_structure: ClassVar[OperatorStructure] = OperatorStructure.unstructured
    """Structural properties of the matrix representation of this gate's operator in the
     in the computational basis"""

    _operator: np.ndarray = None
    """Instance variable for caching operators."""

    _sym_operator: sym.Matrix = None
    """Instance variable for symbolic operators."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

    def asgate(self: GateType) -> "GateType":
        return self

    @property
    @abstractmethod
    def H(self) -> "QuantumGate":
        pass

    @property
    @abstractmethod
    def operator(self) -> np.ndarray:
        pass

    def run(self, ket: State) -> State:
        indices = tuple(ket.qubits.index(q) for q in self.qubits)

        vector = tensormul(
            self.operator,
            ket.vector,
            tuple(indices),
        )
        return State(vector.flatten(), ket.qubits, ket.data)

    @property
    def sym_operator(self) -> sym.Matrix:
        if self._sym_operator is None:
            self._sym_operator = sym.Matrix(self.operator)
        return self._sym_operator

    def permute(self, qubits: Qubits) -> "QuantumGate":
        """Permute the order of the qubits."""
        from .gates import Unitary

        qubits = tuple(qubits)
        if self.qubits == qubits:
            return self

        perm = tuple(self.qubits.index(q) for q in qubits)
        N = self.qubit_nb

        tensor = np.reshape(self.operator, [2] * (2 * N))
        pperm = []

        for rr in range(0, 2):
            pperm += [rr * N + idx for idx in perm]
        tensor = np.transpose(tensor, pperm)
        tensor = np.reshape(tensor, (2 ** N, 2 ** N))

        return Unitary(tensor, qubits)

    def relabel(
        self: OperationType,
        qubit_map: Dict[Qubit, Qubit],
        addr_map: Dict[Addr, Addr] = None,
    ) -> "OperationType":
        qubits = tuple(qubit_map[q] for q in self.qubits)

        op = copy.copy(self)
        op._qubits = qubits
        return op

    def __matmul__(self, other: "QuantumGate") -> "QuantumGate":
        """Apply the action of this gate upon another gate, `self_gate @ other_gate`.
        Recall that time runs right to left with matrix notation.
        """
        from .gates import Identity, Unitary

        if not isinstance(other, QuantumGate):
            raise NotImplementedError()

        extra_qubits = tuple(set(self.qubits) - set(other.qubits))
        if len(extra_qubits) != 0:
            return self @ (other @ Identity(tuple(other.qubits) + extra_qubits))

        indices = tuple(other.qubits.index(q) for q in self.qubits)
        tensor = tensormul(self.operator, other.operator, indices)

        return Unitary(tensor, other.qubits)

    @abstractmethod
    def __pow__(self, t: Variable) -> "QuantumGate":
        """Return this gate raised to the given power."""
        pass


# end class QuantumGate


class QuantumStdGate(QuantumGate):
    _cv_collections = QuantumGate._cv_collections + (STDGATES,)

    cv_sym_operator: ClassVar[sym.Matrix] = None
    """A symbolic representation of this gates' operator as a sympy Matrix.
    Should be set by subclasses. (Subclasses of QuantumStdCtrlGate should set
    cv_target instead.)
    """

    cv_qubit_nb: ClassVar[int] = 0
    """The number of qubits of this gate. Set by subclass initialization."""

    cv_params: ClassVar[Tuple[str, ...]] = ()
    """The named parameters of this class. Parse by subclass initialization from the
    signature of the __init__ method"""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls.__name__.startswith("Quantum"):
            return

        # Parse the Gate parameters and number of qubits from the arguments to __init__
        params = []
        qubit_nb = 0
        for name, namet in cls.__init__.__annotations__.items():
            if namet == Variable:
                params.append(name)
            elif namet == Qubit:
                qubit_nb += 1

        cls.cv_params = tuple(params)
        cls.cv_qubit_nb = qubit_nb

        if cls.cv_sym_operator:  # Not yet set here for subclasses of QuantumCtrlGate
            cls._init_docstring()

    @classmethod
    def _init_docstring(cls) -> None:
        # Automatically add latex matrices to class doc string.
        # Note: We use replace, and not format, to avoid problems with other curly
        # brackets from latex code that might be in the docstring.
        latex = sym.latex(cls.cv_sym_operator)
        cls.__doc__ = cls.__doc__.replace("{autogenerated_latex}", latex)  # DOCME

    def __init__(self, *args_qubits: Union[Variable, Qubit]) -> None:
        # DOCME
        args = args_qubits[0 : -self.cv_qubit_nb]
        qubits = args_qubits[-self.cv_qubit_nb :]
        super().__init__(qubits)
        self._args = tuple(args)

    def __getattr__(self, name: str) -> Variable:
        # DOCME
        if name not in self.cv_params:
            raise AttributeError
        return self.args[self.cv_params.index(name)]

    @property
    def args(self) -> Variables:
        return self._args

    @property
    def operator(self) -> np.ndarray:
        if self._operator is None:
            args = (float(sym.N(x)) for x in self._args)
            M = sym.lambdify(self.cv_params, self.cv_sym_operator)(*args)
            self._operator = M.astype(quantum_dtype)
        return self._operator

    @property
    def sym_operator(self) -> sym.Matrix:
        if self._sym_operator is None:
            M = self.cv_sym_operator
            tmp = [sym.Symbol(f"_not_a_symbol_{i}") for i in range(len(self.cv_params))]
            M = M.subs(dict(zip(self.cv_params, tmp)))
            M = M.subs(dict(zip(tmp, self.args)))
            self._sym_operator = M
        return self._sym_operator

    def relabel(
        self: StdGateType,
        qubit_map: Dict[Qubit, Qubit],
        addr_map: Dict[Addr, Addr] = None,
    ) -> "StdGateType":
        qubits = tuple(qubit_map[q] for q in self.qubits)

        return type(self)(*self.args, *qubits)


# end class QuantumStdGate


class QuantumStdCtrlGate(QuantumStdGate):
    """A standard gate that is a controlled version of another standard gate.

    Concrete instances can be found in the ``stdgates`` module.
    Subclasses should set the ``cv_target`` class variable to the target gate type. The
    class variables ``cv_sym_operator``, ``cv_operator_structure``, and ``cv_hermitian``
    are all set automatically.
    """

    _cv_collections = QuantumStdGate._cv_collections + (STDCTRLGATES,)

    cv_target: ClassVar[Type[QuantumStdGate]]
    """StdGate type that is the target of this controlled gate.
    Should be set by subclasses."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        target = cls.cv_target
        assert target.cv_params == cls.cv_params  # Insanity check

        ctrl_block = sym.eye(2 ** cls.cv_qubit_nb - 2 ** target.cv_qubit_nb)
        target_block = target.cv_sym_operator
        cls.cv_sym_operator = sym.diag(ctrl_block, target_block)

        ctrl_structure = {
            OperatorStructure.identity: OperatorStructure.identity,
            OperatorStructure.diagonal: OperatorStructure.diagonal,
            OperatorStructure.permutation: OperatorStructure.permutation,
            OperatorStructure.swap: OperatorStructure.permutation,
            OperatorStructure.monomial: OperatorStructure.monomial,
            OperatorStructure.unstructured: OperatorStructure.unstructured,
        }

        cls.cv_operator_structure = ctrl_structure[target.cv_operator_structure]
        cls.cv_hermitian = target.cv_hermitian

        cls._init_docstring()

    @property
    def control_qubits(self) -> Qubits:
        return self.qubits[: self.control_qubit_nb]

    @property
    def control_qubit_nb(self) -> int:
        return self.cv_qubit_nb - self.cv_target.cv_qubit_nb

    @property
    def target(self) -> QuantumStdGate:
        target_qubits = self.qubits[self.control_qubit_nb :]
        return self.cv_target(*self.args, *target_qubits)


# end QuantumStdCtrlGate


class QuantumComposite(Collection, QuantumOperation):
    _elements: Tuple[QuantumOperation, ...] = ()

    def __init__(
        self,
        *elements: QuantumOperation,
        qubits: Qubits = None,
        addrs: Addrs = None,
    ):

        elements = tuple(elements)
        elem_qubits = tuple(sorted(set([q for elem in elements for q in elem.qubits])))
        elem_addrs = tuple(sorted(set([c for elem in elements for c in elem.addrs])))

        if qubits is None:
            qubits = elem_qubits
        else:
            qubits = tuple(qubits)
            if not set(elem_qubits).issubset(set(qubits)):
                raise ValueError("Incommensurate qubits")

        if addrs is None:
            addrs = elem_addrs
        else:
            addrs = tuple(addrs)
            if not set(elem_addrs).issubset(set(addrs)):
                raise ValueError(
                    "Incommensurate addresses"
                )  # pragma: no cover  # FIXME TESTME

        super().__init__(qubits, addrs)
        self._elements = tuple(elements)

    def relabel(
        self: CompositeType,
        qubit_map: Dict[Qubit, Qubit] = None,
        addr_map: Dict[Addr, Addr] = None,
    ) -> "CompositeType":
        elements = [elem.relabel(qubit_map, addr_map) for elem in self]
        qubits = tuple(qubit_map.values()) if qubit_map is not None else self.qubits
        addrs = tuple(addr_map.values()) if addr_map is not None else self.addrs

        return type(self)(*elements, qubits=qubits, addrs=addrs)

    # DOCME TESTME
    def run(self, ket: State) -> State:
        # TODO: Create state if None
        for elem in self:
            ket = elem.run(ket)
        return ket

    def __contains__(self, key: Any) -> bool:
        return key in self._elements

    def __iter__(self) -> Iterator[QuantumOperation]:
        yield from self._elements

    def __len__(self) -> int:
        return len(self._elements)


# end class QuantumComposite
