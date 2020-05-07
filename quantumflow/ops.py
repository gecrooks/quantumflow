
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

We consider the elemental quantum operations, such as Gate, Channel, and Kraus,
as immutable. (Although immutability is not enforced in general.)
Transformations of these operations return new copies. On the other hand the
composite operations Circuit and DAGCircuit are mutable.

The main types of Operation's are Gate, Channel, Kraus, Circuit, DAGCircuit,
and Pauli.

.. autoclass:: Operation
    :members:

"""

# NOTE: This file contains the two main types of operations on
# Quantum states, Gate's and Channel's, and an abstract superclass
# Operation. These need to be defined in the same module since they
# reference each other. The class unit tests are currently located
# separately, in test_gates.py, and test_channels.py.


from typing import (
    Dict, Union, Any, Tuple, TypeVar, Sequence, Optional, Iterator, ClassVar)
from copy import copy
from abc import ABC  # Abstract Base Class

import numpy as np
from scipy.linalg import fractional_matrix_power as matpow
from scipy.linalg import logm
import sympy

# import quantumflow.backend as bk

from .qubits import Qubit, Qubits, QubitVector, qubits_count_tuple, asarray
from .states import State, Density
from .utils import symbolize
from .variables import Variable

from .backends import get_backend, BKTensor, TensorLike
bk = get_backend()

__all__ = ['Operation', 'Gate', 'StdGate', 'Unitary', 'Channel']


OperationType = TypeVar('OperationType', bound='Operation')
"""Generic type annotations for subtypes of Operation"""

GateType = TypeVar('GateType', bound='Gate')
"""Generic type annotations for subtypes of Gate"""


class Operation(ABC):
    """ An operation on a qubit state. An element of a quantum circuit.

    Abstract Base Class for Gate, Circuit, DAGCircuit, Channel, Kraus,
    and Pauli.
    """

    interchangeable: ClassVar[bool] = False
    """ Is this a multi-qubit operation that is known to be invariant under
    permutations of qubits?"""

    _diagram_labels: ClassVar[Optional[Sequence[str]]] = None
    """Override default labels for drawing text circuit diagrams.
    See visualizations.circuit_to_diagram()"""

    _diagram_noline: ClassVar[bool] = False
    """Override default to not draw a line between qubit wires for multi-qubit
    operations. See visualizations.circuit_to_diagram()"""

    # DOCME
    def __init__(self,
                 qubits: Qubits = (),
                 params: Dict[str, Variable] = None,
                 ) -> None:
        self._qubits: Qubits = tuple(qubits)
        self._params = params if params is not None else dict()

    @property
    def qubits(self) -> Qubits:
        """Return the qubits that this operation acts upon"""
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits"""
        return len(self.qubits)

    def on(self: OperationType, *qubits: Qubit) -> OperationType:
        """Return a copy of this Gate with new qubits"""
        if len(qubits) != self.qubit_nb:
            raise ValueError("Wrong number of qubits")
        op = copy(self)
        op._qubits = qubits
        return op

    # TODO: Rename to rewire? Or combine with .on()?
    def relabel(self: OperationType, labels: Dict[Qubit, Qubit]) \
            -> OperationType:
        """Relabel qubits and return copy of this gate"""
        qubits = tuple(labels[q] for q in self.qubits)
        op = copy(self)
        op._qubits = qubits
        return op

    @property
    def name(self) -> str:
        """Return the name of this operation"""
        return type(self).__name__

    @property
    def params(self) -> Dict[str, Variable]:
        """Return the parameters of this Operation"""
        return self._params

    def resolve(self, resolution: Dict[str, float]) -> 'Operation':
        """Resolve symbolic parameters"""
        op = copy(self)
        params = {k: float(sympy.N(v, subs=resolution))
                  for k, v in self._params.items()}
        op._params = params
        return op

    def parameters(self) -> Iterator[Variable]:
        """Iterate over all parameters of this Operation"""
        yield from self.params.values()

    def run(self, ket: State) -> State:
        """Apply the action of this operation upon a pure state"""
        raise NotImplementedError()

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this operation upon a mixed state"""
        raise NotImplementedError()

    def asgate(self) -> 'Gate':
        """Convert this quantum operation to a gate (if possible)"""
        raise NotImplementedError()

    def aschannel(self) -> 'Channel':
        """Convert this quantum operation to a channel (if possible)"""
        raise NotImplementedError()

    @property
    def H(self) -> 'Operation':
        """Return the Hermitian conjugate of this quantum operation.

        For unitary Gates (and Circuits composed of the same) the
        Hermitian conjugate returns the inverse Gate (or Circuit)"""
        raise NotImplementedError()

    @property
    def tensor(self) -> BKTensor:
        """
        Returns the tensor representation of this operation (if possible)
        """
        raise NotImplementedError()

    # So that we can ``extend`` Circuits with Operations
    def __iter__(self) -> Any:
        yield self

    # Make Operations sortable. (So we can use Operations in opt_einsum
    # axis labels.)
    def __lt__(self, other: Any) -> bool:
        return id(self) < id(other)

    def specialize(self) -> 'Operation':
        """For parameterized operations, return appropriate special cases
        for particular parameters. Else return the original Operation.

              e.g. RX(0).specialize() -> I()
        """
        return self

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

    Attributes:
        Gate.params: Optional keyword parameters used to create this gate
        Gate.name : The name of this gate

    """

    identity: ClassVar[bool] = False
    """Is this Gate type an identity?"""

    diagonal: ClassVar[bool] = False
    """Is the tensor representation of this Gate known to always be diagonal
    in the computation basis?"""

    permutation: ClassVar[bool] = False
    """Is the tensor representation of this Gate known to always be a permutation
    matrix in the computation basis?"""

    monomial: ClassVar[bool] = False
    """Is the tensor representation of this Gate known to always be a monomial
    matrix in the computation basis (but not diagonal or a permutation?).
    (A monomial matrix is a product of a diagonal and a permutation matrix.
    Only 1 entry in each row and column is non-zero.)"""

    hermitian: ClassVar[bool] = False
    """Is this Gate know to always be hermitian?"""

    # FIXME: Possible replacement for 2 separate boolean properties above.
    # Currently only 'diagonal' is actually exploited by run() method
    # Maybe also need qubit_permutation (Permutation of qubits, not just
    # states) and qubit_monomial. e.g. CNOT is a permutation matrix, but SWAP
    # is both a permutation matrix and a permutation of qubits
    tensor_structure: ClassVar[Optional[str]] = None
    """
    Is the tensor representation of this Operation known to have a particular
    structure in the computational basis?

    Options:
        identity
        diagonal
        permutation
        monomial
        swap

    A monomial matrix is a product of a diagonal and a permutation matrix.
    Only 1 entry in each row and column is non-zero.

    This property is used to optimize the run() and evolve
    state evolution methods.
    """

    # DOCME
    def __init__(self,
                 qubits: Qubits,
                 params: Dict[str, Variable] = None,
                 ) -> None:
        super().__init__(qubits=qubits, params=params)

    # TODO: Not overriding tensor makes Gate an abstract class?
    # @property
    # def tensor(self) -> BKTensor:
    #     """Returns the tensor representation of gate operator"""
    #     raise NotImplementedError()

    # Note: Circular import hell
    from .paulialgebra import Pauli

    @property
    def hamiltonian(self) -> 'Pauli':
        # DOCME
        # FIXME: Doesn't always get phase of unitary correct.
        # See test_gate_hamiltonians()
        from .paulialgebra import pauli_decompose_hermitian
        H = -logm(self.asoperator()) / 1.0j
        pauli = pauli_decompose_hermitian(H, self.qubits)
        return pauli

    def permute(self, qubits: Qubits) -> 'Unitary':
        """Permute the order of the qubits"""
        vec = self.vec.permute(qubits)
        return Unitary(vec.tensor, *vec.qubits)

    # DOCME
    @property
    def vec(self) -> QubitVector:
        return QubitVector(self.tensor, self.qubits)

    # FIXME: Should copy before flatten?
    def asoperator(self) -> BKTensor:
        """Return the gate tensor as a square array"""
        return bk.copy(self.vec.flatten())

    def run(self, ket: State) -> State:
        """Apply the action of this gate upon a state"""
        # if self.tensor_structure == 'identity':
        #     return ket

        diagonal = self.diagonal
        # diagonal = self.tensor_structure == 'diagonal'

        qubits = self.qubits
        indices = [ket.qubits.index(q) for q in qubits]
        tensor = bk.tensormul(self.tensor, ket.tensor, tuple(indices),
                              diagonal=diagonal)
        return State(tensor, ket.qubits, ket.memory)

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this gate upon a density"""
        # TODO: implement without explicit channel creation? With Kraus?
        chan = self.aschannel()
        return chan.evolve(rho)

    def __pow__(self, t: float) -> 'Gate':
        """Return this gate raised to the given power."""
        # Note: This operation cannot be performed within the tensorflow or
        # torch backends in general. Subclasses of Gate may override
        # for special cases.
        N = self.qubit_nb
        matrix = asarray(self.vec.flatten())
        matrix = matpow(matrix, t)
        matrix = np.reshape(matrix, ([2]*(2*N)))
        return Unitary(matrix, *self.qubits)

    @property
    def H(self) -> 'Gate':
        if self.hermitian:
            return self
        return Unitary(self.vec.H.tensor, *self.qubits)

    def __matmul__(self, other: 'Gate') -> 'Gate':
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
        indices = (gate1.qubits.index(q) for q in gate0.qubits)
        tensor = bk.tensormul(gate0.tensor, gate1.tensor, tuple(indices))
        return Unitary(tensor, *gate1.qubits)

    def __str__(self) -> str:
        # Implementation note: We don't want to eval tensor here.

        def _param_format(obj: Any) -> str:
            if isinstance(obj, float):
                try:
                    return str(symbolize(obj))
                except ValueError:
                    return f'{obj}'
            return str(obj)

        # # FIXME
        # if self.name == 'Gate':
        #     return super().__repr__()

        fqubits = " "+" ".join([str(qubit) for qubit in self.qubits])

        if self.params:
            fparams = "(" + ", ".join(_param_format(p)
                                      for p in self.params.values()) + ")"
        else:
            fparams = ""

        return f'{self.name}{fparams}{fqubits}'

    def asgate(self: OperationType) -> 'OperationType':
        return self

    def aschannel(self) -> 'Channel':
        """Converts a Gate into a Channel"""
        N = self.qubit_nb
        R = 4

        tensor = bk.outer(self.tensor, self.H.tensor)
        tensor = bk.reshape(tensor, [2**N]*R)
        tensor = bk.transpose(tensor, [0, 3, 1, 2])

        return Channel(tensor, self.qubits)

    def su(self) -> 'Unitary':
        """Convert gate tensor to the special unitary group."""
        rank = 2**self.qubit_nb
        U = asarray(self.asoperator())
        U /= np.linalg.det(U) ** (1/rank)
        return Unitary(U, *self.qubits)


class StdGate(Gate):
    """
    A standard gate. Standard gates have a name, have a fixed number of real
    parameters, and act upon a fixed number of qubits.
    """

    # DOCME
    # TESTME
    @classmethod
    def args(cls) -> Tuple[str, ...]:
        return tuple(s for s in getattr(cls, '__init__').__annotations__.keys()
                     if s[0] != 'q' and s != 'return')

    # # TESTME
    # @classmethod
    # def random(cls: Type[GateType], *qubits) -> GateType:
    #     """Return a random instance of this gate. If qubits are not given,
    #     then they are also picked randomly.
    #     """
    #     params = [random.uniform(4, 4) for arg in gatetype.args()]
    #     gate = gatetype(*params)

    #     if not qubits:
    #         qubits = list(range(0, 16))
    #         random.shuffle(qubits)
    #         qubits = qubits[0: gate.qubit_nb]

    #     gate = gate.on(*qubits)

    #     return gate

    # def specialize(self) -> StdGate:

# End class StdGate


class Unitary(Gate):
    """
    A quantum logic gate, specified by an explicit unitary operator.
    """

    def __init__(self,
                 tensor: TensorLike,
                 *qubits: Qubit,
                 name: str = None) -> None:

        tensor = bk.astensorproduct(tensor)

        N = bk.ndim(tensor) // 2
        if not qubits:
            qubits = tuple(range(N))

        if len(qubits) != N:
            raise ValueError('Wrong number of qubits for tensor')

        super().__init__(qubits=qubits)
        self._tensor = tensor
        self._name = type(self).__name__ if name is None else name

    @property
    def tensor(self) -> BKTensor:
        """Returns the tensor representation of gate operator"""
        return self._tensor

    @property
    def name(self) -> str:
        return self._name


# End class Unitary


class Channel(Operation):
    """A quantum channel"""
    def __init__(self, tensor: TensorLike,
                 qubits: Union[int, Qubits],
                 params: Dict[str, Variable] = None,
                 name: str = None) -> None:
        _, qubits = qubits_count_tuple(qubits)  # FIXME NEEDED?

        tensor = bk.astensorproduct(tensor)

        N = bk.ndim(tensor) // 4
        if len(qubits) != N:
            raise ValueError('Wrong number of qubits for tensor')

        super().__init__(qubits=qubits, params=params)
        self._tensor = tensor
        self._name = type(self).__name__ if name is None else name

    @property
    # @deprecated
    def vec(self) -> QubitVector:
        return QubitVector(self.tensor, self.qubits)

    @property
    def tensor(self) -> BKTensor:
        """Return the tensor representation of the channel's superoperator"""
        return self._tensor

    @property
    def name(self) -> str:
        return self._name

    def permute(self, qubits: Qubits) -> 'Channel':
        """Return a copy of this channel with qubits in new order"""
        vec = self.vec.permute(qubits)
        return Channel(vec.tensor, qubits=vec.qubits)

    @property
    def H(self) -> 'Channel':
        return Channel(tensor=self.vec.H.tensor, qubits=self.qubits)

    # TESTME
    @property
    def sharp(self) -> 'Channel':
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

        tensor = self.tensor
        tensor = bk.reshape(tensor, [2**N] * 4)
        tensor = bk.transpose(tensor, (0, 2, 1, 3))
        tensor = bk.reshape(tensor, [2] * 4 * N)
        return Channel(tensor, self.qubits)

    def choi(self) -> BKTensor:
        """Return the Choi matrix representation of this super
        operator"""
        # Put superop axes in the order [out_ket, in_bra, out_bra, in_ket]
        # and reshape to matrix
        N = self.qubit_nb
        return bk.reshape(self.sharp.tensor, [2**(N*2)] * 2)

    @classmethod
    def from_choi(cls,
                  tensor: TensorLike,
                  qubits: Union[int, Qubits]) -> 'Channel':
        """Return a Channel from a Choi matrix"""
        return cls(tensor, qubits).sharp

    # TESTME
    # FIXME: Can't be right, same as choi?
    def chi(self) -> BKTensor:
        """Return the chi (or process) matrix representation of this
        superoperator"""
        N = self.qubit_nb
        return bk.reshape(self.sharp.tensor, [2**(N*2)] * 2)

    def run(self, ket: State) -> 'State':
        raise TypeError()  # Not possible in general

    def evolve(self, rho: Density) -> Density:
        """Apply the action of this channel upon a density"""
        N = rho.qubit_nb
        qubits = rho.qubits

        indices = list([qubits.index(q) for q in self.qubits]) + \
            list([qubits.index(q) + N for q in self.qubits])

        tensor = bk.tensormul(self.tensor, rho.tensor, tuple(indices))
        return Density(tensor, qubits, rho.memory)

    def asgate(self) -> 'Gate':
        raise TypeError()  # Not possible in general

    def aschannel(self) -> 'Channel':
        return self

    # FIXME: Maybe not needed, too special a case. Remove?
    # Or make sure can do other operations, such as neg, plus ect
    # Move functionality to QubitVector
    def __add__(self, other: Any) -> 'Channel':
        if isinstance(other, Channel):
            if not self.qubits == other.qubits:
                raise ValueError("Qubits must be identical")
            return Channel(self.tensor + other.tensor, self.qubits)
        raise NotImplementedError()  # Or return NotImplemented?

    # FIXME: Maybe not needed, too special a case. Remove?
    def __mul__(self, other: Any) -> 'Channel':
        return Channel(self.tensor*other, self.qubits)

    def __matmul__(self, other: 'Channel') -> 'Channel':
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
        indices = list([chan1.qubits.index(q) for q in chan0.qubits]) + \
            list([chan1.qubits.index(q) + N for q in chan0.qubits])

        tensor = bk.tensormul(chan0.tensor, chan1.tensor, tuple(indices))

        return Channel(tensor, qubits)

    # TESTME
    def trace(self) -> BKTensor:
        """Return the trace of this super operator"""
        return self.vec.trace()

    # TESTME
    def partial_trace(self, qubits: Qubits) -> 'Channel':
        """Return the partial trace over the specified qubits"""
        vec = self.vec.partial_trace(qubits)
        return Channel(vec.tensor, vec.qubits)

# End class Channel
