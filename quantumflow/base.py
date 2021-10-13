# Copyright 2021-, Gavin E. Crooks
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import enum
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Sequence, Set, Tuple, Type, TypeVar, Union

import numpy as np
import sympy as sym

from .bits import Cbits, Qubit, Qubits
from .config import quantum_dtype

if TYPE_CHECKING:
    # TODO: DOCME
    from numpy.typing import ArrayLike  # pragma: no cover


def _asarray(array: "ArrayLike", ndim: int = None) -> np.ndarray:
    """Converts an array like object to a numpy array with complex data type. We also
    check that the number of elements is a power of 2

    If rank is given (vectors ndim=1, operators ndim=2, super-operators ndim=4)
    we reshape the array to have than number of axes. Otherwise we reshape the array
    so that all axes have length 2.
    """
    arr = np.asarray(array, dtype=quantum_dtype)

    N = np.size(arr)
    K = int(np.log2(N))
    if 2 ** K != N:
        raise ValueError("Wrong number of elements. Must be 2**N where N is an integer")

    if ndim is None:
        shape = (2,) * K
    else:
        shape = (2 ** (K // ndim),) * ndim

    arr = arr.reshape(shape)

    return arr


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


Variable = Union[float, sym.Expr]
"""Type for parameters. Either a float, sympy.Symbol or sympy.Expr"""

Variables = Sequence[Variable]
"""A sequence of Variables"""


OperationType = TypeVar("OperationType", bound="BaseOperation")
"""Generic type annotations for subtypes of BaseOperation"""


GateType = TypeVar("GateType", bound="BaseGate")
"""Generic type annotations for subtypes of BaseGate"""


OPERATIONS: Set[Type["BaseOperation"]] = set()
"""All quantum operations (All concrete subclasses of BaseOperation)"""

GATES: Set[Type["BaseGate"]] = set()
"""All gates (All concrete subclasses of BaseGate)"""

STDGATES: Set[Type["BaseStdGate"]] = set()
"""All standard gates (All concrete subclasses of BaseStdGate)"""

STDCTRLGATES: Set[Type["BaseStdCtrlGate"]] = set()
"""All standard controlled gates (All concrete subclasses of BaseStdCtrlGate)"""


class BaseOperation(ABC):
    _cv_collections: ClassVar[Tuple[Set, ...]] = (OPERATIONS,)
    """List of collections to add subclasses to."""

    _qubits: Qubits = ()
    _cbits: Cbits = ()
    name: str

    def __init_subclass__(cls) -> None:
        name = cls.__name__
        cls.name = name

        # Subclass registration
        if not inspect.isabstract(cls):
            for collection in cls._cv_collections:
                collection.add(cls)

    def __init__(self, qubits: Qubits, cbits: Cbits = ()) -> None:
        self._qubits = tuple(qubits)
        self._cbits = tuple(cbits)

    @abstractmethod
    def asgate(self) -> "BaseGate":
        """
        Convert this quantum operation into a gate (if possible).

        Raises:
            ValueError: If this operation cannot be converted to a Gate.
        """
        pass

    @property
    def cbits(self) -> Cbits:
        """Return the classical bits that this operation acts upon."""
        return self._cbits

    @property
    def cbit_nb(self) -> int:
        """Return the total number of qubits."""
        return len(self.cbits)

    @property
    @abstractmethod
    def H(self) -> "BaseOperation":
        """Return the Hermitian conjugate of this quantum operation. For unitary Gates
        (and Circuits composed of the same) the Hermitian conjugate returns the inverse
        Gate (or Circuit).

        Raises:
            ValueError: If this operation does not support Hermitian conjugation.
        """
        pass

    @property
    def qubits(self) -> Qubits:
        """Return the Qubits that this operation acts upon."""
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits."""
        return len(self.qubits)


# end class BaseOperation


class BaseGate(BaseOperation):
    _cv_collections = BaseOperation._cv_collections + (GATES,)

    cv_interchangable: ClassVar[bool] = False

    cv_hermitian: ClassVar[bool] = False
    """Is this gate's operator known to be always hermitian?"""

    cv_operator_structure: ClassVar[OperatorStructure] = OperatorStructure.unstructured
    """Structural properties of the matrix representation of this gate's operator in the
     in the computational basis"""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

    def asgate(self: GateType) -> "GateType":
        return self

    @property
    @abstractmethod
    def H(self) -> "BaseGate":
        pass

    @property
    @abstractmethod
    def operator(self) -> np.ndarray:
        pass

    @abstractmethod
    def __pow__(self, t: Variable) -> "BaseGate":
        """Return this gate raised to the given power."""
        pass


# end class BaseGate


class BaseStdGate(BaseGate):
    _cv_collections = BaseGate._cv_collections + (STDGATES,)

    cv_sym_operator: ClassVar[sym.Matrix] = None

    cv_qubit_nb: ClassVar[int] = 0

    cv_params: ClassVar[Tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls.__name__.startswith("Base"):
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

        if cls.cv_sym_operator:  # Not yet set for subclasses of BaseCtrlGate
            # Automatically add latex matrices to class doc string.
            # Note: We use replace, and not format, to avoid problems with other curly
            # brackets from latex code that might be in the docstring.
            latex = sym.latex(cls.cv_sym_operator)
            cls.__doc__ = cls.__doc__.replace("{autogenerated_latex}", latex)  # DOCME

    def __init__(self, args: Variables, qubits: Qubits) -> None:
        super().__init__(qubits)
        self._args = tuple(args)

    @property
    def args(self) -> Variables:
        return self._args

    @property
    def operator(self) -> np.ndarray:
        args = (float(sym.N(x)) for x in self._args)
        M = sym.lambdify(self.cv_params, self.cv_sym_operator)(*args)
        return _asarray(M, ndim=2)

    @property
    def sym_operator(self) -> sym.Matrix:
        M = self.cv_sym_operator
        tmp = [sym.Symbol(f"_not_a_symbol_{i}") for i in range(len(self.cv_params))]
        M = M.subs(dict(zip(self.cv_params, tmp)))
        M = M.subs(dict(zip(tmp, self.args)))
        return M


# end class BaseStdGate


class BaseStdCtrlGate(BaseStdGate):
    """A standard gate that is a controlled version of another standard gate.

    Subclasses should set the `cv_target` class variable to the target gate type.
    """

    _cv_collections = BaseStdGate._cv_collections + (STDCTRLGATES,)

    cv_target: ClassVar[Type[BaseStdGate]]
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

        latex = sym.latex(cls.cv_sym_operator)
        cls.__doc__ = cls.__doc__.replace("{autogenerated_latex}", latex)

    @property
    def control_qubits(self) -> Qubits:
        return self.qubits[: self.control_qubit_nb]

    @property
    def control_qubit_nb(self) -> int:
        return self.cv_qubit_nb - self.cv_target.cv_qubit_nb

    @property
    def target(self) -> BaseStdGate:
        target_qubits = self.qubits[self.control_qubit_nb :]
        return self.cv_target(*self.args, *target_qubits)  # type: ignore


# end BaseStdCtrlGate
